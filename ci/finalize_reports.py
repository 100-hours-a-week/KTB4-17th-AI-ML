#!/usr/bin/env python3
"""Publish explicit availability evidence independently of validator imports/dependencies."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def failure_junit(path, name, message):
    suite = ET.Element('testsuite', name='pretraining.reporting', tests='1', failures='1', errors='0', skipped='0')
    case = ET.SubElement(suite, 'testcase', classname='pretraining.reporting', name=name)
    ET.SubElement(case, 'failure', message=message)
    ET.ElementTree(suite).write(path, encoding='unicode', xml_declaration=True)


def valid_xml(path):
    try:
        return ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None


def finalize(output, outcomes):
    output = Path(output)
    preflight = output / 'preflight'
    preflight.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat()
    try:
        report = json.loads((preflight / 'preflight.json').read_text())
        if not isinstance(report, dict) or type(report.get('artifact_preflight')) is not bool:
            raise ValueError('Report contract unavailable')
    except (OSError, ValueError):
        commit = os.environ.get('GITHUB_SHA')
        if not commit:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
            commit = result.stdout.strip() if result.returncode == 0 else 'unknown'
        message = 'Validator report unavailable; artifact measurements were not established.'
        report = {'schema_version': '1.0', 'created_at_utc': created,
                  'identity': {'repository': os.environ.get('GITHUB_REPOSITORY', '100-hours-a-week/KTB4-17th-AI-ML'),
                               'commit': commit, 'run_id': os.environ.get('GITHUB_RUN_ID', 'local'),
                               'run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT', '1')},
                  'artifact_preflight': False, 'training_ready': False, 'measurements_available': False,
                  'training_executed': False, 'gpu_used': False, 'external_model_downloads': 0,
                  'model_downloads': 0, 'metrics': {},
                  'checks': [{'name': 'validator_execution', 'status': 'failed', 'duration_seconds': None,
                              'errors': [message], 'warnings': [], 'skips': [], 'metrics': {}}],
                  'readiness_reasons': [{'code': 'validation_unavailable', 'status': 'unavailable', 'message': message}]}
        write_json(preflight / 'preflight.json', report)
    artifact_junit_available = valid_xml(preflight / 'preflight.junit.xml') is not None
    artifact_markdown_available = (preflight / 'summary.md').is_file()
    if not artifact_junit_available:
        failure_junit(preflight / 'preflight.junit.xml', 'validator_execution',
                      'Artifact JUnit unavailable; validation is not confirmed.')
    if not artifact_markdown_available:
        (preflight / 'summary.md').write_text('# Model pre-training artifact preflight\n\n'
            'Artifact preflight: **unavailable**\nTraining ready: **false**\n\n'
            'Validator output is unavailable. No artifact measurements or passing status are inferred.\n')
    fixture_root = valid_xml(output / 'tests.junit.xml')
    fixtures_available = fixture_root is not None
    if fixture_root is None:
        failure_junit(output / 'tests.junit.xml', 'fixture_execution',
                      'Fixture JUnit unavailable; fixture tests are not confirmed.')
    coverage = {'status': 'unavailable', 'source_scope': ['ci/'], 'percent_covered': None,
                'reason': 'Scoped coverage was not established; no percentage is inferred.'}
    try:
        percent = json.loads((output / 'coverage.json').read_text())['totals']['percent_covered']
        if type(percent) not in (int, float) or not 0 <= percent <= 100 or not math.isfinite(percent):
            raise ValueError('Invalid measured coverage')
        coverage.update(status='available', percent_covered=percent, reason='Measured ci/ coverage report exists.')
    except (OSError, ValueError, KeyError, TypeError):
        pass
    write_json(output / 'coverage-availability.json', coverage)
    passed = (all(value == 'success' for value in outcomes.values()) and report['artifact_preflight']
              and artifact_junit_available and artifact_markdown_available
              and fixtures_available and coverage['status'] == 'available')
    write_json(output / 'finalization.json', {'schema_version': '1.0', 'created_at_utc': created,
               'status': 'passed' if passed else 'failed', 'step_outcomes': outcomes,
               'artifact_junit_available': artifact_junit_available,
               'artifact_markdown_available': artifact_markdown_available,
               'fixture_results_available': fixtures_available, 'coverage': coverage})
    summary = (preflight / 'summary.md').read_text() + '\n## CI execution and report availability\n\n'
    summary += 'CI result: **' + ('passed' if passed else 'failed') + '**\n\n'
    summary += ('Original artifact JUnit available: **' + str(artifact_junit_available).lower() + '**; '
                'original artifact Markdown available: **' + str(artifact_markdown_available).lower() + '**.\n\n')
    summary += '\n'.join('- ' + step + ': ' + outcome for step, outcome in outcomes.items()) + '\n\n'
    if fixtures_available:
        fields = ('tests', 'failures', 'errors', 'skipped')
        totals = {key: sum(int(s.get(key, 0)) for s in fixture_root.iter('testsuite')) for key in fields}
        summary += 'Fixture JUnit: ' + ', '.join(f'{key}: {value}' for key, value in totals.items()) + '\n\n'
    else:
        summary += 'Fixture results unavailable; retained JUnit records a reporting failure.\n\n'
    summary += (f"Coverage of ci/ only: {coverage['percent_covered']:.2f}% (measured).\n"
                if coverage['status'] == 'available' else 'Scoped coverage unavailable; no measured percentage exists.\n')
    (output / 'summary.md').write_text(summary)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports'))
    for step in ('install', 'compile', 'tests', 'audit'):
        parser.add_argument('--' + step + '-outcome', choices=('success', 'failure', 'skipped', 'cancelled'), default='skipped')
    args = parser.parse_args()
    return finalize(args.output, {step: getattr(args, step + '_outcome') for step in ('install', 'compile', 'tests', 'audit')})


if __name__ == '__main__':
    raise SystemExit(main())
