"""Independent finalization must survive an unimportable validator and absent dependencies."""
import json
import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

REPO = Path(__file__).resolve().parents[2]
FINALIZER = REPO / 'ci/finalize_reports.py'


def finalize(root, script=FINALIZER, **outcomes):
    command = [sys.executable, str(script), '--output', str(root)]
    for step, outcome in outcomes.items():
        command.extend(['--' + step + '-outcome', outcome])
    return subprocess.run(command, capture_output=True, text=True, cwd=script.parent.parent)


@pytest.mark.parametrize('failure', ['compile', 'install'])
def test_unavailable_validation_materializes_failed_artifacts(tmp_path, failure):
    isolated_ci = tmp_path / 'ci'
    isolated_ci.mkdir()
    broken = isolated_ci / 'preflight.py'
    broken.write_text('def invalid(:\n')
    isolated_finalizer = isolated_ci / 'finalize_reports.py'
    isolated_finalizer.write_bytes(FINALIZER.read_bytes())
    assert subprocess.run([sys.executable, str(broken)], capture_output=True).returncode != 0
    output = tmp_path / 'reports'
    outcomes = {'install': 'success', 'compile': 'success', 'tests': 'skipped', 'audit': 'failure'}
    outcomes[failure] = 'failure'
    result = finalize(output, script=isolated_finalizer, **outcomes)
    assert result.returncode == 1
    report = json.loads((output / 'preflight/preflight.json').read_text())
    assert report['artifact_preflight'] is False
    assert report['training_ready'] is False
    assert report['measurements_available'] is False
    assert report['metrics'] == {}
    assert report['identity']['commit']
    assert ET.parse(output / 'preflight/preflight.junit.xml').find('.//failure') is not None
    assert ET.parse(output / 'tests.junit.xml').find('.//failure') is not None
    coverage = json.loads((output / 'coverage-availability.json').read_text())
    assert coverage['status'] == 'unavailable'
    assert coverage['percent_covered'] is None
    assert not (output / 'coverage.json').exists()
    summary = (output / 'summary.md').read_text()
    assert 'unavailable' in summary and 'Training ready: **false**' in summary
    assert '0.00%' not in summary
    assert (output / 'preflight/summary.md').exists()


def test_preserves_actual_reports_and_marks_dependency_failure(tmp_path):
    sys.path.insert(0, str(REPO))
    from ci.preflight import audit, write_reports
    output = tmp_path / 'reports'
    write_reports(audit(tmp_path), output / 'preflight')
    original = (output / 'preflight/preflight.json').read_bytes()
    result = finalize(output, install='failure', compile='skipped', tests='skipped', audit='failure')
    assert result.returncode == 1
    assert (output / 'preflight/preflight.json').read_bytes() == original
    assert json.loads((output / 'coverage-availability.json').read_text())['status'] == 'unavailable'


def test_finalizer_retains_measured_coverage_and_fixture_results(tmp_path):
    from ci.preflight import audit, write_reports
    output = tmp_path / 'reports'
    write_reports(audit(tmp_path), output / 'preflight')
    (output / 'tests.junit.xml').write_text('<testsuites><testsuite tests="3" failures="0" errors="0" skipped="0"/></testsuites>')
    (output / 'coverage.json').write_text(json.dumps({'totals': {'percent_covered': 96.25}}))
    result = finalize(output, install='success', compile='success', tests='success', audit='failure')
    assert result.returncode == 1  # The real artifact check failed; publishing must preserve failure.
    coverage = json.loads((output / 'coverage-availability.json').read_text())
    assert coverage['status'] == 'available' and coverage['percent_covered'] == 96.25
    assert '96.25%' in (output / 'summary.md').read_text()
    assert 'tests: 3' in (output / 'summary.md').read_text()


def test_finalizer_can_publish_successful_existing_reports(tmp_path):
    from ci.preflight import audit, write_reports
    output = tmp_path / 'reports'
    report = audit(tmp_path)
    # Controlled report consumer input: the finalizer must preserve the recorded result.
    report['artifact_preflight'] = True
    write_reports(report, output / 'preflight')
    (output / 'tests.junit.xml').write_text('<testsuite tests="1" failures="0" errors="0" skipped="0"/>')
    (output / 'coverage.json').write_text(json.dumps({'totals': {'percent_covered': 100.0}}))
    result = finalize(output, install='success', compile='success', tests='success', audit='success')
    assert result.returncode == 0
    assert json.loads((output / 'finalization.json').read_text())['status'] == 'passed'


def test_malformed_existing_reports_are_replaced_with_failure_evidence(tmp_path):
    output = tmp_path / 'reports'
    (output / 'preflight').mkdir(parents=True)
    for filename in ('preflight/preflight.json', 'preflight/preflight.junit.xml', 'tests.junit.xml', 'coverage.json'):
        (output / filename).write_text('broken')
    result = finalize(output, install='success', compile='failure', tests='skipped', audit='failure')
    assert result.returncode == 1
    assert json.loads((output / 'preflight/preflight.json').read_text())['measurements_available'] is False
    assert ET.parse(output / 'preflight/preflight.junit.xml').find('.//failure') is not None
    assert json.loads((output / 'coverage-availability.json').read_text())['percent_covered'] is None


@pytest.mark.parametrize('unavailable', ['missing_junit', 'malformed_junit', 'missing_markdown'])
def test_missing_original_artifact_evidence_cannot_finalize_as_passed(tmp_path, unavailable):
    from ci.preflight import audit, write_reports
    output = tmp_path / 'reports'
    report = audit(tmp_path)
    report['artifact_preflight'] = True  # Controlled, passing report-consumer input.
    write_reports(report, output / 'preflight')
    (output / 'tests.junit.xml').write_text('<testsuite tests="1" failures="0" errors="0" skipped="0"/>')
    (output / 'coverage.json').write_text(json.dumps({'totals': {'percent_covered': 100.0}}))
    junit = output / 'preflight/preflight.junit.xml'
    markdown = output / 'preflight/summary.md'
    if unavailable == 'missing_junit':
        junit.unlink()
    elif unavailable == 'malformed_junit':
        junit.write_text('<broken')
    else:
        markdown.unlink()
    result = finalize(output, install='success', compile='success', tests='success', audit='success')
    assert result.returncode == 1
    finalization = json.loads((output / 'finalization.json').read_text())
    assert finalization['status'] == 'failed'
    assert finalization['artifact_junit_available'] is (unavailable == 'missing_markdown')
    assert finalization['artifact_markdown_available'] is (unavailable != 'missing_markdown')
    assert 'CI result: **failed**' in (output / 'summary.md').read_text()
    assert json.loads((output / 'preflight/preflight.json').read_text())['artifact_preflight'] is True
    if unavailable != 'missing_markdown':
        assert ET.parse(junit).find('.//failure') is not None
    else:
        assert 'unavailable' in markdown.read_text()
    assert json.loads((output / 'coverage-availability.json').read_text())['percent_covered'] == 100.0
