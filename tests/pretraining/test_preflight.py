"""Hand-sized LoRA fixtures exercise the real parser and CLI, without ML libraries."""
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def fixture(root, name='guard', value=1.0):
    directory = root / 'adapters' / name
    directory.mkdir(parents=True)
    config = {'peft_type': 'LORA', 'task_type': 'CAUSAL_LM', 'r': 2,
              'lora_alpha': 4, 'target_modules': ['q_proj'],
              'base_model_name_or_path': '/historical/merged'}
    (directory / 'adapter_config.json').write_text(json.dumps(config))
    header = {}
    for side, shape, offsets in [('A', [2, 3], [0, 24]), ('B', [3, 2], [24, 48])]:
        header[f'base_model.model.model.layers.0.self_attn.q_proj.lora_{side}.weight'] = {
            'dtype': 'F32', 'shape': shape, 'data_offsets': offsets}
    write_tensor(directory, header, struct.pack('<12f', *([value] * 12)))
    return directory, header


def write_tensor(directory, header, payload=b'\0' * 48):
    raw = json.dumps(header).encode()
    (directory / 'adapter_model.safetensors').write_bytes(struct.pack('<Q', len(raw)) + raw + payload)


def run_audit(root, names=('guard',)):
    from ci.preflight import audit
    return audit(root, expected_adapters=names)


def test_valid_counts_and_historical_provenance(tmp_path):
    directory, _ = fixture(tmp_path)
    result = run_audit(tmp_path)
    assert result['artifact_preflight'] is True
    assert result['training_ready'] is False
    assert len(result['readiness_reasons']) == 6
    assert result['metrics']['tensor_count'] == 2
    assert result['metrics']['parameter_count'] == 12
    assert result['metrics']['checked_values'] == 12
    assert result['metrics']['nonfinite_values'] == 0
    adapter = result['adapters'][0]
    assert adapter['pair_count'] == 1 and adapter['layers'] == [0]
    assert adapter['rank'] == 2 and adapter['lora_alpha'] == 4
    assert adapter['base_model_provenance'] == '/historical/merged'
    assert adapter['files']['weights']['sha256'] == hashlib.sha256((directory / 'adapter_model.safetensors').read_bytes()).hexdigest()
    assert result['training_executed'] is False and result['model_downloads'] == 0 and result['gpu_used'] is False
    assert all(c['duration_seconds'] >= 0 for c in result['checks'])


@pytest.mark.parametrize('kind', ['rank', 'target', 'missing_pair', 'layer_gap', 'overlap', 'gap',
    'out_of_bounds', 'truncated', 'dtype', 'negative_shape', 'huge_shape', 'bool_offset',
    'bad_key', 'extra_dimension', 'metadata', 'nan', 'inf', 'negative_inf'])
def test_rejects_corrupt_tensor_before_success(tmp_path, kind):
    directory, header = fixture(tmp_path)
    first, second = header
    payload = b'\0' * 48
    if kind == 'rank': header[first]['shape'] = [3, 2]
    if kind == 'target': header = {k.replace('q_proj', 'v_proj'): v for k, v in header.items()}
    if kind == 'missing_pair': del header[second]; payload = payload[:24]
    if kind == 'layer_gap': header = {k.replace('layers.0', 'layers.1'): v for k, v in header.items()}
    if kind == 'overlap': header[second]['data_offsets'] = [0, 24]
    if kind == 'gap': header[second]['data_offsets'] = [28, 52]; payload += b'\0' * 4
    if kind == 'out_of_bounds': header[second]['data_offsets'] = [24, 100]
    if kind == 'truncated': payload = payload[:-1]
    if kind == 'dtype': header[first]['dtype'] = 'BOGUS'
    if kind == 'negative_shape': header[first]['shape'] = [-2, -3]
    if kind == 'huge_shape': header[first]['shape'] = [2, 10**100]
    if kind == 'bool_offset': header[first]['data_offsets'][0] = False
    if kind == 'bad_key': header['unexpected'] = header.pop(first)
    if kind == 'extra_dimension': header[first]['shape'] = [1, 2, 3]
    if kind == 'metadata': header['__metadata__'] = {'secret': 1}
    if kind in ('nan', 'inf', 'negative_inf'):
        value = {'nan': float('nan'), 'inf': float('inf'), 'negative_inf': -float('inf')}[kind]
        payload = struct.pack('<f', value) + payload[4:]
    write_tensor(directory, header, payload)
    result = run_audit(tmp_path)
    assert result['artifact_preflight'] is False
    assert any(c['errors'] for c in result['checks'])
    assert result['training_ready'] is False
    if kind not in ('nan', 'inf', 'negative_inf'):
        assert result['metrics']['checked_values'] == 0


@pytest.mark.parametrize('raw', [b'', b'123', struct.pack('<Q', 2**63),
    struct.pack('<Q', 10) + b'{}', struct.pack('<Q', 8) + b'{"x":NaN}',
    struct.pack('<Q', 13) + b'{"x":1,"x":2}', struct.pack('<Q', 2) + b'[]',
    struct.pack('<Q', 2) + b'{{', struct.pack('<Q', 2) + b'\xff\xff',
    struct.pack('<Q', 2) + b'{\xff'])
def test_malformed_header_is_bounded_failure(tmp_path, raw):
    directory, _ = fixture(tmp_path)
    (directory / 'adapter_model.safetensors').write_bytes(raw)
    result = run_audit(tmp_path)
    assert result['artifact_preflight'] is False
    assert result['metrics']['checked_values'] == 0


@pytest.mark.parametrize('change', [ {'r': True}, {'r': 0}, {'lora_alpha': float('nan')},
    {'target_modules': []}, {'target_modules': ['q_proj', 'q_proj']}, {'peft_type': 'OTHER'},
    {'rank_pattern': {'q_proj': 4}}, {'use_dora': True}, {'base_model_name_or_path': ''}])
def test_invalid_config_fails(tmp_path, change):
    directory, _ = fixture(tmp_path)
    path = directory / 'adapter_config.json'
    config = json.loads(path.read_text()); config.update(change); path.write_text(json.dumps(config))
    assert run_audit(tmp_path)['artifact_preflight'] is False


def test_missing_inputs_and_cross_adapter_mismatch(tmp_path):
    assert run_audit(tmp_path)['artifact_preflight'] is False
    fixture(tmp_path)
    directory, header = fixture(tmp_path, 'secure')
    header = {k.replace('self_attn', 'mlp'): v for k, v in header.items()}
    write_tensor(directory, header)
    result = run_audit(tmp_path, ('guard', 'secure'))
    assert result['artifact_preflight'] is False
    assert any(c['name'] == 'adapter_compatibility' and c['status'] == 'failed' for c in result['checks'])


def test_cli_writes_failure_reports_and_exit_code(tmp_path):
    output = tmp_path / 'reports'
    run = subprocess.run([sys.executable, str(REPO / 'ci/preflight.py'), '--root', str(tmp_path),
                          '--output', str(output)], capture_output=True, text=True)
    assert run.returncode == 1
    report = json.loads((output / 'preflight.json').read_text())
    assert report['artifact_preflight'] is False
    assert report['identity']['repository']
    assert 'Training ready: **false**' in (output / 'summary.md').read_text()
    assert 'failure' in (output / 'preflight.junit.xml').read_text()
    assert '/historical/merged' not in (output / 'summary.md').read_text()


def test_cli_success_reports_checks_and_counts(tmp_path):
    for name in ('guard', 'secure', 'anxiety', 'avoid'): fixture(tmp_path, name)
    run = subprocess.run([sys.executable, str(REPO / 'ci/preflight.py'), '--root', str(tmp_path),
        '--output', str(tmp_path / 'reports')], capture_output=True, text=True)
    assert run.returncode == 0
    report = json.loads((tmp_path / 'reports/preflight.json').read_text())
    assert report['metrics']['adapter_count'] == 4
    assert report['metrics']['checked_values'] == 48
    assert report['training_ready'] is False
    assert '<failure' not in (tmp_path / 'reports/preflight.junit.xml').read_text()


def test_oversized_alpha_is_reported_without_crashing(tmp_path):
    directory, _ = fixture(tmp_path)
    path = directory / 'adapter_config.json'
    config = json.loads(path.read_text()); config['lora_alpha'] = 10**400
    path.write_text(json.dumps(config))
    assert run_audit(tmp_path)['artifact_preflight'] is False


def test_huge_layer_index_is_reported_without_crashing(tmp_path):
    directory, header = fixture(tmp_path)
    header = {k.replace('layers.0', 'layers.' + '1'*5000): v for k, v in header.items()}
    write_tensor(directory, header)
    assert run_audit(tmp_path)['artifact_preflight'] is False


@pytest.mark.parametrize('dtype,fmt,values', [
    ('F16', 'e', [1.0]*12), ('F64', 'd', [1.0]*12), ('BF16', 'H', [0x3f80]*12),
    ('F16', 'e', [float('inf')]*12), ('BF16', 'H', [0x7f80]*12)])
def test_float_dtype_sizes_and_finiteness(tmp_path, dtype, fmt, values):
    directory, header = fixture(tmp_path)
    width = struct.calcsize('<' + fmt)
    for index, spec in enumerate(header.values()):
        spec['dtype'] = dtype; spec['data_offsets'] = [index*6*width, (index+1)*6*width]
    write_tensor(directory, header, struct.pack('<12' + fmt, *values))
    result = run_audit(tmp_path)
    assert result['metrics']['checked_values'] == 12
    assert result['artifact_preflight'] is (values[0] in (1.0, 0x3f80))


def test_extra_targets_missing_layer_pair_and_trailing_payload(tmp_path):
    directory, header = fixture(tmp_path)
    path = directory / 'adapter_config.json'
    config = json.loads(path.read_text()); config['target_modules'].append('v_proj')
    path.write_text(json.dumps(config))
    assert run_audit(tmp_path)['artifact_preflight'] is False
    config['target_modules'] = ['q_proj']; path.write_text(json.dumps(config))
    write_tensor(directory, header, b'\0'*52)
    assert run_audit(tmp_path)['artifact_preflight'] is False


def test_missing_config_skips_payload_and_oversized_config_is_bounded(tmp_path):
    directory, _ = fixture(tmp_path)
    config = directory / 'adapter_config.json'
    config.unlink()
    result = run_audit(tmp_path)
    assert any(c['name'] == 'guard/finite_payload' and c['status'] == 'skipped' for c in result['checks'])
    config.write_bytes(b' ' * (1024*1024+1))
    assert run_audit(tmp_path)['artifact_preflight'] is False


def test_sparse_layer_target_grid_fails_before_payload_scan(tmp_path):
    directory, source = fixture(tmp_path)
    config_path = directory / 'adapter_config.json'
    config = json.loads(config_path.read_text()); config['target_modules'].append('v_proj')
    config_path.write_text(json.dumps(config))
    header = {}
    for layer, target in [(0, 'q_proj'), (0, 'v_proj'), (1, 'q_proj')]:
        for key, spec in source.items():
            index = len(header)
            header[key.replace('layers.0', f'layers.{layer}').replace('q_proj', target)] = {
                **spec, 'data_offsets': [index * 24, (index+1) * 24]}
    write_tensor(directory, header, b'\0' * 144)
    result = run_audit(tmp_path)
    assert result['artifact_preflight'] is False
    assert result['metrics']['checked_values'] == 0


def test_split_module_a_b_pair_is_rejected_before_scan(tmp_path):
    directory, header = fixture(tmp_path)
    key = next(k for k in header if 'lora_B' in k)
    header[key.replace('self_attn', 'mlp')] = header.pop(key)
    write_tensor(directory, header)
    result = run_audit(tmp_path)
    assert result['artifact_preflight'] is False
    assert result['metrics']['checked_values'] == 0


@pytest.mark.parametrize('encoding,prefix', [('utf-16', b''), ('utf-8', b' ')])
def test_header_requires_utf8_and_initial_object_byte(tmp_path, encoding, prefix):
    directory, header = fixture(tmp_path)
    raw = prefix + json.dumps(header).encode(encoding)
    (directory / 'adapter_model.safetensors').write_bytes(struct.pack('<Q', len(raw)) + raw + b'\0'*48)
    result = run_audit(tmp_path)
    assert result['artifact_preflight'] is False
    assert result['metrics']['checked_values'] == 0


def test_explicit_partial_layer_zero_is_unsupported(tmp_path):
    directory, _ = fixture(tmp_path)
    path = directory / 'adapter_config.json'
    config = json.loads(path.read_text()); config['layers_to_transform'] = 0
    path.write_text(json.dumps(config))
    result = run_audit(tmp_path)
    assert result['artifact_preflight'] is False
    assert result['metrics']['checked_values'] == 0
