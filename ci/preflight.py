#!/usr/bin/env python3
"""Inspect local LoRA artifacts. Never imports ML code or downloads models."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import time
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

MAX_HEADER_BYTES = 8 * 1024 * 1024
MAX_CONFIG_BYTES = 1024 * 1024
CHUNK_BYTES = 1024 * 1024
MAX_TENSORS = 100_000
DTYPES = {'F64': (8, 'd'), 'F32': (4, 'f'), 'F16': (2, 'e'), 'BF16': (2, 'H')}
KEY = re.compile(r'^base_model\.model\.model\.layers\.(0|[1-9][0-9]{0,4})\.(self_attn|mlp)\.([a-z][a-z0-9_]*)\.lora_([AB])\.weight$')
DEFAULT_ADAPTERS = ('guard', 'secure', 'anxiety', 'avoid')
READINESS = {
    'training_entrypoint': 'No reviewed training entrypoint is defined in this repository.',
    'dataset_manifest': 'No versioned training dataset manifest is defined.',
    'training_hyperparameters': 'No reproducible training hyperparameter configuration is defined.',
    'dependency_lock': 'No training dependency lock is defined; CI dependencies are not a training lock.',
    'base_checkpoint_fingerprint': 'No exact immutable base checkpoint fingerprint is recorded.',
    'guard_checkpoint_fingerprint': 'No exact immutable merged guard checkpoint fingerprint is recorded.',
}


class InvalidArtifact(ValueError):
    """A sanitized validation error; never contains raw tensor or JSON content."""


def require(condition, message):
    if not condition:
        raise InvalidArtifact(message)


def integer(value):
    return type(value) is int


def strict_json(raw):
    def constant(_):
        raise InvalidArtifact('Non-standard JSON numeric constant is forbidden.')

    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON key is forbidden.')
            result[key] = value
        return result

    try:
        result = json.loads(raw, parse_constant=constant, object_pairs_hook=unique)
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, InvalidArtifact):
            raise
        raise InvalidArtifact('Malformed or excessively nested JSON.') from None
    require(isinstance(result, dict), 'JSON root must be an object.')
    return result


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def validate_config(path):
    require(path.stat().st_size <= MAX_CONFIG_BYTES, 'Adapter config exceeds size limit.')
    config = strict_json(path.read_bytes())
    require(config.get('peft_type') == 'LORA' and config.get('task_type') == 'CAUSAL_LM',
            'Only causal language model LoRA adapters are supported.')
    rank = config.get('r')
    require(integer(rank) and 0 < rank <= 65536, 'LoRA rank must be a bounded positive integer.')
    alpha = config.get('lora_alpha')
    require(type(alpha) in (int, float) and 0 < alpha <= 1e9 and math.isfinite(alpha),
            'LoRA alpha must be finite and positive.')
    targets = config.get('target_modules')
    require(isinstance(targets, list) and 0 < len(targets) <= 1024 and
            all(isinstance(t, str) and re.fullmatch(r'[a-z][a-z0-9_]{0,127}', t) for t in targets),
            'Target modules must be a nonempty bounded list of module names.')
    require(len(targets) == len(set(targets)), 'Duplicate target module names are forbidden.')
    base = config.get('base_model_name_or_path')
    require(isinstance(base, str) and 0 < len(base) <= 4096, 'Base model provenance must be a nonempty string.')
    for field in ('rank_pattern', 'alpha_pattern'):
        require(not config.get(field), 'Per-module rank or alpha overrides are unsupported.')
    for field in ('use_dora', 'use_qalora', 'use_rslora', 'fan_in_fan_out', 'lora_bias'):
        require(not config.get(field), 'This adapter variation is unsupported by this preflight.')
    require(config.get('bias', 'none') == 'none' and not config.get('modules_to_save') and
            config.get('layers_to_transform') is None and not config.get('layer_replication'),
            'Additional saved modules or partial/replicated layers are unsupported.')
    return config


def validate_header(path, config):
    size = path.stat().st_size
    require(size >= 8, 'Safetensors length prefix is truncated.')
    with path.open('rb') as handle:
        length = struct.unpack('<Q', handle.read(8))[0]
        require(2 <= length <= MAX_HEADER_BYTES, 'Safetensors header exceeds bounded size limits.')
        require(length <= size - 8, 'Safetensors header is truncated.')
        raw_header = handle.read(length)
        require(raw_header.startswith(b'{'), 'Safetensors header must begin with an object byte.')
        try:
            header_text = raw_header.decode('utf-8', errors='strict')
        except UnicodeError:
            raise InvalidArtifact('Safetensors header must use UTF-8 encoding.') from None
        header = strict_json(header_text)
    metadata = header.pop('__metadata__', {})
    require(isinstance(metadata, dict) and all(isinstance(v, str) for v in metadata.values()),
            'Safetensors metadata must contain string values.')
    require(0 < len(header) <= MAX_TENSORS, 'Safetensors tensor count is outside bounded limits.')
    payload_bytes = size - 8 - length
    ranges, pairs, layers, targets, signature = [], {}, set(), set(), []
    dtype_counts = {}
    parameters = 0
    for key, spec in header.items():
        match = KEY.fullmatch(key)
        require(match is not None, 'Unsupported LoRA tensor key.')
        layer, module, target, side = int(match[1]), match[2], match[3], match[4]
        require(layer < 65536, 'Layer index exceeds supported limit.')
        require(isinstance(spec, dict) and set(spec) == {'dtype', 'shape', 'data_offsets'},
                'Tensor descriptor must contain dtype, shape and data offsets only.')
        dtype, shape, offsets = spec['dtype'], spec['shape'], spec['data_offsets']
        require(isinstance(dtype, str) and dtype in DTYPES, 'Unsupported tensor dtype; expected floating-point LoRA weights.')
        require(isinstance(shape, list) and len(shape) == 2 and all(integer(d) and 0 < d <= 2**31 for d in shape),
                'LoRA tensor shape must contain two bounded positive dimensions.')
        require(isinstance(offsets, list) and len(offsets) == 2 and all(integer(o) for o in offsets),
                'Tensor offsets must contain two integers.')
        start, end = offsets
        require(0 <= start <= end <= payload_bytes, 'Tensor data offsets exceed payload bounds.')
        count = math.prod(shape)
        require(count * DTYPES[dtype][0] == end - start, 'Tensor byte size does not match dtype and shape.')
        require((shape[0] if side == 'A' else shape[1]) == config['r'], 'Tensor rank differs from adapter config.')
        require(target in config['target_modules'], 'Tensor target differs from adapter config.')
        pair = pairs.setdefault((layer, module, target), {})
        require(side not in pair, 'Duplicate LoRA A/B pair side.')
        pair[side] = key
        ranges.append((start, end, dtype))
        layers.add(layer); targets.add(target)
        signature.append((key, dtype, tuple(shape)))
        parameters += count
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    require(all(set(pair) == {'A', 'B'} for pair in pairs.values()), 'A LoRA A/B pair is missing.')
    require(layers == set(range(max(layers) + 1)), 'Layer indices must be contiguous from zero.')
    require(targets == set(config['target_modules']), 'Configured targets do not match tensor targets.')
    require(len(pairs) == len(layers) * len(targets) and
            len({(layer, target) for layer, _, target in pairs}) == len(pairs),
            'Every layer must contain every configured target pair.')
    cursor = 0
    for start, end, _ in sorted(ranges):
        require(start == cursor, 'Tensor payload has a gap or overlapping offsets.')
        cursor = end
    require(cursor == payload_bytes, 'Payload has trailing or unaccounted bytes.')
    metrics = {'tensor_count': len(header), 'parameter_count': parameters, 'pair_count': len(pairs),
               'layers': sorted(layers), 'targets': sorted(targets), 'dtype_tensor_counts': dtype_counts,
               'header_bytes': length, 'payload_bytes': payload_bytes}
    return metrics, sorted(signature), sorted(ranges), 8 + length


def scan_payload(path, ranges, payload_start):
    checked = nonfinite = 0
    with path.open('rb') as handle:
        for start, end, dtype in ranges:
            handle.seek(payload_start + start)
            width, code = DTYPES[dtype]
            remaining = end - start
            while remaining:
                block = handle.read(min(CHUNK_BYTES, remaining))
                require(len(block) == min(CHUNK_BYTES, remaining), 'Tensor payload became truncated during scan.')
                for (value,) in struct.iter_unpack('<' + code, block):
                    checked += 1
                    nonfinite += ((value & 0x7f80) == 0x7f80) if dtype == 'BF16' else not math.isfinite(value)
                remaining -= len(block)
    return {'checked_values': checked, 'nonfinite_values': nonfinite, 'artifact_bytes_scanned': sum(b-a for a,b,_ in ranges)}


def identity(root):
    commit = os.environ.get('GITHUB_SHA')
    if not commit:
        try:
            commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True,
                                    text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            commit = 'unknown'
    return {'repository': os.environ.get('GITHUB_REPOSITORY', '100-hours-a-week/KTB4-17th-AI-ML'),
            'commit': commit, 'run_id': os.environ.get('GITHUB_RUN_ID', 'local'),
            'run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT', '1'),
            'workflow': os.environ.get('GITHUB_WORKFLOW', 'local preflight')}


def audit(root, expected_adapters=DEFAULT_ADAPTERS):
    root = Path(root)
    report = {'schema_version': '1.0', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'identity': identity(root), 'training_executed': False, 'gpu_used': False,
              'model_downloads': 0, 'external_model_downloads': 0,
              'checks': [], 'adapters': [], 'training_ready': False,
              'readiness_reasons': [{'code': key, 'status': 'missing', 'message': value} for key, value in READINESS.items()]}

    def check(name, action=None, skip=None):
        started = time.perf_counter()
        entry = {'name': name, 'status': 'skipped' if skip else 'passed', 'errors': [],
                 'warnings': [], 'skips': [skip] if skip else [], 'metrics': {}}
        value = None
        if action and not skip:
            try:
                value = action()
            except (InvalidArtifact, OSError) as exc:
                entry['status'] = 'failed'
                entry['errors'] = [str(exc) if isinstance(exc, InvalidArtifact) else 'Required artifact is missing or unreadable.']
        entry['duration_seconds'] = round(time.perf_counter() - started, 6)
        report['checks'].append(entry)
        return entry, value

    signatures = []
    for name in expected_adapters:
        directory = root / 'adapters' / name
        adapter = {'name': name, 'files': {}}
        report['adapters'].append(adapter)
        config_check, config = check(name + '/config', lambda: validate_config(directory / 'adapter_config.json'))
        if config is not None:
            adapter.update(rank=config['r'], lora_alpha=config['lora_alpha'],
                           base_model_provenance=config['base_model_name_or_path'])
            config_check['warnings'].append('Base model name/path is historical provenance; existence and checkpoint identity are not verified.')
        for label, filename in [('config', 'adapter_config.json'), ('weights', 'adapter_model.safetensors')]:
            entry, details = check(name + '/' + label + '_fingerprint', lambda p=directory / filename: fingerprint(p))
            if details is not None:
                adapter['files'][label] = details
                entry['metrics'] = details
        header_check, parsed = check(name + '/header', lambda: validate_header(directory / 'adapter_model.safetensors', config),
                                     skip='Valid adapter config is required.' if config is None else None)
        if parsed is None:
            check(name + '/finite_payload', skip='Valid config and complete structural header validation are required.')
            continue
        metrics, signature, ranges, start = parsed
        adapter.update(metrics)
        header_check['metrics'] = metrics.copy()
        signatures.append((signature, config['r'], config['lora_alpha'], tuple(sorted(config['target_modules']))))
        finite_check, finite = check(name + '/finite_payload', lambda: scan_payload(directory / 'adapter_model.safetensors', ranges, start))
        if finite is not None:
            adapter.update(finite)
            finite_check['metrics'] = finite
            if finite['nonfinite_values']:
                finite_check['status'] = 'failed'
                finite_check['errors'] = ['Payload contains non-finite scalar values.']

    def compatible():
        require(all(signature == signatures[0] for signature in signatures[1:]),
                'Adapters differ in tensor names/shapes/dtypes, rank, alpha or targets.')
    check('adapter_compatibility', compatible,
          skip='Every expected adapter must have a valid header.' if len(signatures) != len(expected_adapters) or not signatures else None)
    readiness, _ = check('training_prerequisites', skip='Training prerequisites are unavailable; see readiness_reasons.')
    readiness['warnings'] = ['Artifact preflight does not establish model quality or training readiness.']
    report['metrics'] = {
        'adapter_count': len(report['adapters']),
        'valid_header_count': len(signatures),
        'weight_bytes': sum(a['files'].get('weights', {}).get('bytes', 0) for a in report['adapters']),
        **{field: sum(a.get(field, 0) for a in report['adapters']) for field in
           ('tensor_count', 'parameter_count', 'checked_values', 'nonfinite_values', 'artifact_bytes_scanned')},
    }
    report['artifact_preflight'] = bool(expected_adapters) and not any(c['status'] == 'failed' for c in report['checks']) and len(signatures) == len(expected_adapters)
    return report


def write_reports(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'preflight.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    checks = report['checks']
    suite = ET.Element('testsuite', name='adapter_artifact_preflight', tests=str(len(checks)),
                       failures=str(sum(c['status'] == 'failed' for c in checks)),
                       skipped=str(sum(c['status'] == 'skipped' for c in checks)),
                       time=str(sum(c['duration_seconds'] for c in checks)))
    for check in checks:
        case = ET.SubElement(suite, 'testcase', name=check['name'], classname='pretraining.artifacts', time=str(check['duration_seconds']))
        if check['status'] == 'failed':
            ET.SubElement(case, 'failure', message='; '.join(check['errors'])).text = '\n'.join(check['errors'])
        if check['status'] == 'skipped':
            ET.SubElement(case, 'skipped', message='; '.join(check['skips']))
    ET.ElementTree(suite).write(output / 'preflight.junit.xml', encoding='unicode', xml_declaration=True)
    metrics = report['metrics']
    lines = ['# Model pre-training artifact preflight', '',
             f"Artifact preflight: **{str(report['artifact_preflight']).lower()}**",
             'Training ready: **false**', '',
             'This measures local artifact integrity and compatibility, not model quality.', '',
             '| Metric | Value |', '| --- | ---: |']
    lines.extend(f'| {key} | {value} |' for key, value in metrics.items())
    lines.extend(['', '| Check | Status | Seconds | Diagnostics |', '| --- | --- | ---: | --- |'])
    for check in checks:
        diagnostics = check['errors'] + check['warnings'] + check['skips']
        lines.append(f"| {check['name']} | {check['status']} | {check['duration_seconds']} | {'; '.join(diagnostics)} |")
    lines.extend(['', '## Missing training prerequisites', ''])
    lines.extend('- ' + reason['message'] for reason in report['readiness_reasons'])
    lines.extend(['', 'No training, inference, external model/tokenizer downloads or GPU use occurred.',
                  'The download counters refer to external model downloads; tracked adapter weights arrive with the source checkout.',
                  'JSON contains UTC time, repository/commit/run identity, hashes and measured adapter metrics.', ''])
    (output / 'summary.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('reports'))
    args = parser.parse_args()
    report = audit(args.root)
    write_reports(report, args.output)
    print(f"artifact_preflight={str(report['artifact_preflight']).lower()} training_ready=false")
    return 0 if report['artifact_preflight'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
