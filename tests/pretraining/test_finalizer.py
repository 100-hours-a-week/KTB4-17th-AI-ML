"""Complete tiny adapter evidence and explicit corrupt report-consumer regressions."""
import json
import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
from test_preflight import fixture

REPO = Path(__file__).resolve().parents[2]
FINALIZER = REPO / 'ci/finalize_reports.py'


def finalize(root, script=FINALIZER, **outcomes):
    command = [sys.executable, str(script), '--output', str(root)]
    for step, outcome in outcomes.items(): command.extend(['--' + step + '-outcome', outcome])
    return subprocess.run(command, capture_output=True, text=True, cwd=script.parent.parent)


def successful_outcomes():
    return dict(install='success', compile='success', tests='success', audit='success')


def coverage_fixture(output, covered=80):
    """Hand-counted, complete JSON/XML consumer fixtures; not actual CI measurements."""
    files = {}
    xml = ET.Element('coverage', {'lines-valid': '80', 'lines-covered': str(covered)})
    sources = ET.SubElement(xml, 'sources'); ET.SubElement(sources, 'source').text = 'ci'
    classes = ET.SubElement(ET.SubElement(ET.SubElement(xml, 'packages'), 'package'), 'classes')
    for path in ('ci/__init__.py', 'ci/finalize_reports.py', 'ci/preflight.py'):
        count = 80 if path.endswith('preflight.py') else 0
        hit = covered if count else 0
        summary = {'covered_lines': hit, 'num_statements': count, 'missing_lines': count-hit,
                   'excluded_lines': 0, 'percent_covered': 100*hit/count if count else 100.0}
        files[path] = {'executed_lines': list(range(1,hit+1)), 'missing_lines': list(range(hit+1,count+1)),
                       'excluded_lines': [], 'summary': summary}
        node = ET.SubElement(classes, 'class', filename=Path(path).name)
        lines = ET.SubElement(node, 'lines')
        for line in range(1,count+1): ET.SubElement(lines, 'line', number=str(line), hits='1' if line <= hit else '0')
    totals = {'covered_lines': covered, 'num_statements': 80, 'missing_lines': 80-covered,
              'excluded_lines': 0, 'percent_covered': 100*covered/80}
    (output / 'coverage.json').write_text(json.dumps({'files': files, 'totals': totals}))
    ET.ElementTree(xml).write(output / 'coverage.xml', encoding='unicode')


def passing_evidence(tmp_path):
    from ci.preflight import audit, write_reports
    checkout = tmp_path / 'checkout'
    for name in ('guard', 'secure', 'anxiety', 'avoid'): fixture(checkout, name)
    report = audit(checkout)
    assert report['artifact_preflight'] is True
    output = tmp_path / 'reports'; write_reports(report, output / 'preflight')
    suite = ET.Element('testsuite', tests='3', failures='0', errors='0', skipped='0')
    for name in ('tiny_counts', 'malformed_input', 'report_contract'): ET.SubElement(suite, 'testcase', name=name)
    ET.ElementTree(suite).write(output / 'tests.junit.xml', encoding='unicode')
    coverage_fixture(output)
    return output


@pytest.mark.parametrize('failure', ['compile', 'install'])
def test_unavailable_validation_materializes_failed_artifacts(tmp_path, failure):
    isolated_ci = tmp_path / 'ci'; isolated_ci.mkdir()
    broken = isolated_ci / 'preflight.py'; broken.write_text('def invalid(:\n')
    isolated = isolated_ci / 'finalize_reports.py'; isolated.write_bytes(FINALIZER.read_bytes())
    assert subprocess.run([sys.executable, str(broken)], capture_output=True).returncode != 0
    output = tmp_path / 'reports'
    outcomes = dict(install='success', compile='success', tests='skipped', audit='failure'); outcomes[failure] = 'failure'
    assert finalize(output, script=isolated, **outcomes).returncode == 1
    report = json.loads((output / 'preflight/preflight.json').read_text())
    assert report['artifact_preflight'] is False and report['training_ready'] is False
    assert report['measurements_available'] is False and report['metrics'] == {}
    assert report['identity']['commit']
    for filename in ('preflight/preflight.junit.xml', 'tests.junit.xml'):
        assert ET.parse(output / filename).find('.//failure') is not None
    coverage = json.loads((output / 'coverage-availability.json').read_text())
    assert coverage['status'] == 'unavailable' and coverage['percent_covered'] is None
    assert not (output / 'coverage.json').exists()
    summary = (output / 'summary.md').read_text()
    assert 'unavailable' in summary and 'Training ready: **false**' in summary and '0.00%' not in summary


def test_valid_failed_evidence_is_retained_without_masking_failure(tmp_path):
    from ci.preflight import audit, write_reports
    output = passing_evidence(tmp_path)
    (tmp_path / 'checkout/adapters/guard/adapter_config.json').unlink()
    write_reports(audit(tmp_path / 'checkout'), output / 'preflight')
    original_json = (output / 'preflight/preflight.json').read_bytes()
    original_xml = (output / 'preflight/preflight.junit.xml').read_bytes()
    coverage_fixture(output, covered=77)
    result = finalize(output, **{**successful_outcomes(), 'audit': 'failure'})
    assert result.returncode == 1
    assert (output / 'preflight/preflight.json').read_bytes() == original_json
    assert (output / 'preflight/preflight.junit.xml').read_bytes() == original_xml
    assert '96.25%' in (output / 'summary.md').read_text()
    assert 'tests: 3' in (output / 'summary.md').read_text()


def test_complete_genuinely_passing_adapter_evidence_can_finalize(tmp_path):
    output = passing_evidence(tmp_path)
    assert finalize(output, **successful_outcomes()).returncode == 0
    assert json.loads((output / 'finalization.json').read_text())['status'] == 'passed'


@pytest.mark.parametrize('unavailable', ['missing_junit', 'malformed_junit', 'missing_markdown'])
def test_missing_original_artifact_evidence_cannot_finalize_as_passed(tmp_path, unavailable):
    output = passing_evidence(tmp_path)
    junit = output / 'preflight/preflight.junit.xml'; markdown = output / 'preflight/summary.md'
    if unavailable == 'missing_junit': junit.unlink()
    elif unavailable == 'malformed_junit': junit.write_text('<broken')
    else: markdown.unlink()
    assert finalize(output, **successful_outcomes()).returncode == 1
    record = json.loads((output / 'finalization.json').read_text())
    assert record['status'] == 'failed'
    assert record['artifact_junit_available'] is (unavailable == 'missing_markdown')
    assert record['artifact_markdown_available'] is (unavailable != 'missing_markdown')
    assert 'CI result: **failed**' in (output / 'summary.md').read_text()
    if unavailable != 'missing_markdown': assert ET.parse(junit).find('.//failure') is not None
    else: assert 'unavailable' in markdown.read_text()


@pytest.mark.parametrize('location', ['tests.junit.xml', 'preflight/preflight.junit.xml'])
@pytest.mark.parametrize('corrupt', ['root', 'nonnumeric', 'negative', 'wrong_count', 'hidden_failure'])
def test_invalid_junit_is_failed_unavailable_evidence_not_green_or_crash(tmp_path, location, corrupt):
    output = passing_evidence(tmp_path)
    path = output / location
    tree = ET.parse(path); root = tree.getroot()
    if corrupt == 'root': root.tag = 'arbitrary'
    if corrupt == 'nonnumeric': root.set('tests', 'oops')
    if corrupt == 'negative': root.set('failures', '-1')
    if corrupt == 'wrong_count': root.set('tests', '999')
    if corrupt == 'hidden_failure': ET.SubElement(root.find('testcase'), 'failure')
    tree.write(path, encoding='unicode')
    result = finalize(output, **successful_outcomes())
    assert result.returncode == 1 and 'Traceback' not in result.stderr
    assert json.loads((output / 'finalization.json').read_text())['status'] == 'failed'
    assert 'CI result: **failed**' in (output / 'summary.md').read_text()
    assert ET.parse(path).find('.//failure') is not None


@pytest.mark.parametrize('contradiction', ['failed_check', 'passed_with_error', 'missing_check', 'failed_audit_junit'])
def test_artifact_boolean_cannot_override_failed_or_incomplete_checks(tmp_path, contradiction):
    output = passing_evidence(tmp_path)
    path = output / 'preflight/preflight.json'; report = json.loads(path.read_text())
    if contradiction == 'failed_check': report['checks'][0].update(status='failed', errors=['Sanitized fixture error.'])
    if contradiction == 'passed_with_error': report['checks'][0]['errors'] = ['Contradictory sanitized error.']
    if contradiction == 'missing_check': report['checks'].pop(0)
    if contradiction == 'failed_audit_junit':
        junit = output / 'preflight/preflight.junit.xml'; tree = ET.parse(junit)
        tree.getroot().set('failures', '1'); ET.SubElement(tree.getroot().find('testcase'), 'failure')
        tree.write(junit, encoding='unicode')
    path.write_text(json.dumps(report))
    assert finalize(output, **successful_outcomes()).returncode == 1
    assert json.loads(path.read_text())['artifact_preflight'] is False
    assert json.loads((output / 'finalization.json').read_text())['status'] == 'failed'
    assert 'unavailable' in (output / 'summary.md').read_text()


@pytest.mark.parametrize('corrupt', ['percentage_only', 'outside_scope', 'bad_totals', 'bad_percent', 'missing_xml', 'bad_xml_totals'])
def test_coverage_requires_real_source_entries_and_coherent_totals(tmp_path, corrupt):
    output = passing_evidence(tmp_path)
    path = output / 'coverage.json'; data = json.loads(path.read_text())
    if corrupt == 'percentage_only': data = {'totals': {'percent_covered': 100.0}}
    if corrupt == 'outside_scope': data['files']['../unrelated.py'] = data['files'].pop('ci/preflight.py')
    if corrupt == 'bad_totals': data['totals']['covered_lines'] = 999
    if corrupt == 'bad_percent': data['totals']['percent_covered'] = 2.0
    if corrupt == 'missing_xml': (output / 'coverage.xml').unlink()
    if corrupt == 'bad_xml_totals':
        xml = output / 'coverage.xml'; tree = ET.parse(xml); tree.getroot().set('lines-covered', '999'); tree.write(xml)
    path.write_text(json.dumps(data))
    assert finalize(output, **successful_outcomes()).returncode == 1
    coverage = json.loads((output / 'coverage-availability.json').read_text())
    assert coverage['status'] == 'unavailable' and coverage['percent_covered'] is None
    assert 'Scoped coverage unavailable' in (output / 'summary.md').read_text()


def test_valid_failed_fixture_junit_is_retained_and_blocks_success(tmp_path):
    output = passing_evidence(tmp_path)
    path = output / 'tests.junit.xml'; tree = ET.parse(path)
    tree.getroot().set('failures', '1'); ET.SubElement(tree.getroot().find('testcase'), 'failure')
    tree.write(path, encoding='unicode'); original = path.read_bytes()
    assert finalize(output, **successful_outcomes()).returncode == 1
    assert path.read_bytes() == original
    assert json.loads((output / 'finalization.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('field', ['weight_bytes', 'tensor_count', 'parameter_count', 'checked_values',
                                  'nonfinite_values', 'artifact_bytes_scanned'])
def test_successful_aggregate_metrics_must_match_measured_adapter_sums(tmp_path, field):
    output = passing_evidence(tmp_path)
    path = output / 'preflight/preflight.json'; report = json.loads(path.read_text())
    report['metrics'][field] += 1
    path.write_text(json.dumps(report))
    assert finalize(output, **successful_outcomes()).returncode == 1
    assert json.loads((output / 'finalization.json').read_text())['status'] == 'failed'
    assert json.loads(path.read_text())['artifact_preflight'] is False


@pytest.mark.parametrize('mutation', ['adapter_header', 'check_header', 'adapter_finite', 'check_finite',
                                      'fingerprint', 'hidden_payload_size', 'hidden_nonfinite'])
def test_adapter_percheck_and_successful_numerical_invariants_are_coherent(tmp_path, mutation):
    output = passing_evidence(tmp_path)
    path = output / 'preflight/preflight.json'; report = json.loads(path.read_text())
    adapter = report['adapters'][0]
    checks = {c['name']: c for c in report['checks']}
    if mutation == 'adapter_header': adapter['tensor_count'] += 1
    if mutation == 'check_header': checks['guard/header']['metrics']['tensor_count'] += 1
    if mutation == 'adapter_finite': adapter['checked_values'] += 1
    if mutation == 'check_finite': checks['guard/finite_payload']['metrics']['checked_values'] += 1
    if mutation == 'fingerprint': checks['guard/weights_fingerprint']['metrics']['bytes'] += 1
    if mutation == 'hidden_payload_size':
        adapter['payload_bytes'] += 4; adapter['artifact_bytes_scanned'] += 4
        checks['guard/header']['metrics']['payload_bytes'] += 4
        checks['guard/finite_payload']['metrics']['artifact_bytes_scanned'] += 4
        report['metrics']['artifact_bytes_scanned'] += 4
    if mutation == 'hidden_nonfinite':
        adapter['nonfinite_values'] = 1; checks['guard/finite_payload']['metrics']['nonfinite_values'] = 1
        report['metrics']['nonfinite_values'] = 1
    path.write_text(json.dumps(report))
    assert finalize(output, **successful_outcomes()).returncode == 1
    assert json.loads((output / 'finalization.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('location', ['totals', 'file'])
def test_huge_coverage_percentage_is_unavailable_without_suppressing_reports(tmp_path, location):
    output = passing_evidence(tmp_path)
    path = output / 'coverage.json'; coverage = json.loads(path.read_text())
    summary = coverage['totals'] if location == 'totals' else coverage['files']['ci/preflight.py']['summary']
    summary['percent_covered'] = 10**309
    path.write_text(json.dumps(coverage))
    result = finalize(output, **successful_outcomes())
    assert result.returncode == 1 and 'Traceback' not in result.stderr
    assert json.loads((output / 'coverage-availability.json').read_text())['percent_covered'] is None
    assert json.loads((output / 'finalization.json').read_text())['status'] == 'failed'
    assert 'Scoped coverage unavailable' in (output / 'summary.md').read_text()


def test_all_skipped_fixture_cases_cannot_certify_execution(tmp_path):
    output = passing_evidence(tmp_path)
    path = output / 'tests.junit.xml'; tree = ET.parse(path); suite = tree.getroot()
    for case in suite.findall('testcase'): ET.SubElement(case, 'skipped')
    suite.set('skipped', suite.get('tests')); tree.write(path, encoding='unicode')
    original = path.read_bytes()
    assert finalize(output, **successful_outcomes()).returncode == 1
    assert path.read_bytes() == original
    assert json.loads((output / 'finalization.json').read_text())['status'] == 'failed'
    assert 'tests: 3' in (output / 'summary.md').read_text()
