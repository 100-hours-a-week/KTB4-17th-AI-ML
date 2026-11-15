#!/usr/bin/env python3
"""Publish coherent evidence independently of validator imports/dependencies."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('tests', 'failures', 'errors', 'skipped')
ADAPTERS = ('guard', 'secure', 'anxiety', 'avoid')
CHECK_NAMES = {name + '/' + check for name in ADAPTERS for check in
               ('config', 'config_fingerprint', 'weights_fingerprint', 'header', 'finite_payload')}
CHECK_NAMES.update(('adapter_compatibility', 'training_prerequisites'))


def require(condition):
    if not condition:
        raise ValueError('Evidence contract is unavailable or inconsistent.')


def count(value):
    require(isinstance(value, str) and re.fullmatch(r'0|[1-9][0-9]{0,11}', value) is not None)
    return int(value)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def failure_junit(path, name, message):
    suite = ET.Element('testsuite', name='pretraining.reporting', tests='1', failures='1', errors='0', skipped='0')
    case = ET.SubElement(suite, 'testcase', classname='pretraining.reporting', name=name)
    ET.SubElement(case, 'failure', message=message)
    ET.ElementTree(suite).write(path, encoding='unicode', xml_declaration=True)


def junit_evidence(path):
    """Check real testcase counts before treating any parseable XML as JUnit."""
    try:
        root = ET.parse(path).getroot()
        require(root.tag in ('testsuite', 'testsuites'))
        suites = [root] if root.tag == 'testsuite' else list(root)
        require(suites and all(s.tag == 'testsuite' for s in suites))
        totals = dict.fromkeys(FIELDS, 0)
        cases = []
        for suite in suites:
            require(not suite.findall('testsuite'))
            recorded = {field: count(suite.get(field)) for field in FIELDS}
            actual = dict.fromkeys(FIELDS, 0)
            for case in suite.findall('testcase'):
                require(bool(case.get('name')))
                states = [state for state in ('failure', 'error', 'skipped') if case.find(state) is not None]
                require(len(states) <= 1)
                status = 'failed' if states and states[0] in ('failure', 'error') else 'skipped' if states else 'passed'
                actual['tests'] += 1
                if states: actual[{'failure': 'failures', 'error': 'errors', 'skipped': 'skipped'}[states[0]]] += 1
                cases.append((case.get('name'), status))
            require(recorded == actual)
            for field in FIELDS: totals[field] += actual[field]
        if root.tag == 'testsuites' and any(field in root.attrib for field in FIELDS):
            require({field: count(root.get(field)) for field in FIELDS} == totals)
        require(totals['tests'] > 0)
        return {'totals': totals, 'cases': cases,
                'passed': not (totals['failures'] or totals['errors'])}
    except (OSError, ValueError, ET.ParseError):
        return None


def artifact_evidence(path, junit):
    try:
        report = json.loads(path.read_text())
        require(isinstance(report, dict) and type(report.get('artifact_preflight')) is bool)
        require(report.get('training_ready') is False)
        checks = report.get('checks')
        require(isinstance(checks, list) and len(checks) == len(CHECK_NAMES))
        require(all(isinstance(c, dict) and isinstance(c.get('name'), str) for c in checks))
        statuses = {c['name']: c.get('status') for c in checks}
        require(set(statuses) == CHECK_NAMES and all(status in ('passed', 'failed', 'skipped') for status in statuses.values()))
        for check in checks:
            require(all(isinstance(check.get(field), list) and all(isinstance(item, str) for item in check[field])
                        for field in ('errors', 'warnings', 'skips')))
            require((bool(check['errors']) if check['status'] == 'failed' else not check['errors']) and
                    (bool(check['skips']) if check['status'] == 'skipped' else not check['skips']))
        require(statuses['training_prerequisites'] == 'skipped')
        measured_pass = all(status == 'passed' for name, status in statuses.items() if name != 'training_prerequisites')
        require(report['artifact_preflight'] == measured_pass)
        if junit is not None:
            require(len(junit['cases']) == len(statuses) and dict(junit['cases']) == statuses)
        metrics = report.get('metrics')
        require(isinstance(metrics, dict) and all(type(value) is int and value >= 0 for value in metrics.values()))
        require(metrics.get('adapter_count') == 4)
        if measured_pass: require(metrics.get('valid_header_count') == 4)
        return report
    except (OSError, ValueError, KeyError, TypeError):
        return None


def coverage_evidence(output):
    unavailable = {'status': 'unavailable', 'source_scope': ['ci/'], 'percent_covered': None,
                   'reason': 'Scoped coverage files/totals were not established; no percentage is inferred.'}
    try:
        data = json.loads((output / 'coverage.json').read_text())
        expected = {p.relative_to(ROOT).as_posix() for p in (ROOT / 'ci').glob('*.py') if p.is_file() and not p.is_symlink()}
        require(isinstance(data, dict) and isinstance(data.get('files'), dict) and set(data['files']) == expected)
        require(expected)
        fields = ('covered_lines', 'num_statements', 'missing_lines', 'excluded_lines')
        totals = dict.fromkeys(fields, 0)
        summaries = {}
        for filename, details in data['files'].items():
            summary = details['summary']
            require(all(type(summary.get(key)) is int and summary[key] >= 0 for key in fields))
            require(summary['covered_lines'] + summary['missing_lines'] == summary['num_statements'])
            arrays = [details[key] for key in ('executed_lines', 'missing_lines', 'excluded_lines')]
            line_limit = len((ROOT / filename).read_text().splitlines())
            require(all(isinstance(lines, list) and len(lines) == len(set(lines)) and
                        all(type(line) is int and 0 < line <= line_limit for line in lines) for lines in arrays))
            require(len(arrays[0]) >= summary['covered_lines'] and len(arrays[1]) == summary['missing_lines'] and
                    len(arrays[2]) == summary['excluded_lines'] and not set(arrays[0]) & set(arrays[1]))
            ratio = 100 * summary['covered_lines'] / summary['num_statements'] if summary['num_statements'] else 100.0
            require(type(summary.get('percent_covered')) in (int, float) and math.isclose(summary['percent_covered'], ratio, abs_tol=0.01))
            summaries[filename] = summary
            for key in fields: totals[key] += summary[key]
        require(totals['num_statements'] > 0 and all(type(data['totals'].get(key)) is int and data['totals'][key] == totals[key] for key in fields))
        percent = data['totals']['percent_covered']
        require(type(percent) in (int, float) and math.isfinite(percent) and
                math.isclose(percent, 100 * totals['covered_lines'] / totals['num_statements'], abs_tol=0.01))
        xml = ET.parse(output / 'coverage.xml').getroot()
        require(xml.tag == 'coverage' and [s.text for s in xml.findall('./sources/source')] == ['ci'])
        require(count(xml.get('lines-valid')) == totals['num_statements'] and count(xml.get('lines-covered')) == totals['covered_lines'])
        classes = xml.findall('./packages/package/classes/class')
        require(len(classes) == len(expected) and {'ci/' + c.get('filename', '') for c in classes} == expected)
        for node in classes:
            filename = 'ci/' + node.get('filename')
            lines = [(count(line.get('number')), count(line.get('hits'))) for line in node.findall('./lines/line')]
            summary = summaries[filename]
            require(len(lines) == summary['num_statements'] and len({number for number, _ in lines}) == len(lines))
            require(sum(hits > 0 for _, hits in lines) == summary['covered_lines'])
            require({number for number, hits in lines if not hits} == set(data['files'][filename]['missing_lines']))
        return {'status': 'available', 'source_scope': ['ci/'], 'percent_covered': percent,
                'reason': 'Measured per-file coverage JSON/XML sources and totals agree.'}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ET.ParseError):
        return unavailable


def unavailable_report(created):
    commit = os.environ.get('GITHUB_SHA')
    if not commit:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
        commit = result.stdout.strip() if result.returncode == 0 else 'unknown'
    message = 'Validator evidence unavailable or contradictory; artifact measurements are not confirmed.'
    return {'schema_version': '1.0', 'created_at_utc': created,
            'identity': {'repository': os.environ.get('GITHUB_REPOSITORY', '100-hours-a-week/KTB4-17th-AI-ML'),
                         'commit': commit, 'run_id': os.environ.get('GITHUB_RUN_ID', 'local'),
                         'run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT', '1')},
            'artifact_preflight': False, 'training_ready': False, 'measurements_available': False,
            'training_executed': False, 'gpu_used': False, 'external_model_downloads': 0, 'model_downloads': 0,
            'metrics': {}, 'checks': [{'name': 'validator_execution', 'status': 'failed', 'duration_seconds': None,
                                     'errors': [message], 'warnings': [], 'skips': [], 'metrics': {}}],
            'readiness_reasons': [{'code': 'validation_unavailable', 'status': 'unavailable', 'message': message}]}


def finalize(output, outcomes):
    output = Path(output); preflight = output / 'preflight'; preflight.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat()
    artifact_junit = junit_evidence(preflight / 'preflight.junit.xml')
    artifact_junit_available = artifact_junit is not None
    report = artifact_evidence(preflight / 'preflight.json', artifact_junit)
    report_available = report is not None
    if not report_available:
        report = unavailable_report(created)
        write_json(preflight / 'preflight.json', report)
    if not artifact_junit_available:
        failure_junit(preflight / 'preflight.junit.xml', 'validator_execution',
                      'Artifact JUnit unavailable; validation is not confirmed.')
    try:
        artifact_summary = (preflight / 'summary.md').read_text()
        artifact_markdown_available = bool(artifact_summary.strip())
    except (OSError, UnicodeError):
        artifact_markdown_available = False
    if not artifact_markdown_available or not report_available:
        artifact_summary = ('# Model pre-training artifact preflight\n\nArtifact preflight: **unavailable**\n'
                            'Training ready: **false**\n\nValidator evidence is unavailable or contradictory. '
                            'No passing status or artifact measurements are inferred.\n')
        (preflight / 'summary.md').write_text(artifact_summary)
    fixtures = junit_evidence(output / 'tests.junit.xml')
    fixtures_available = fixtures is not None
    if not fixtures_available:
        failure_junit(output / 'tests.junit.xml', 'fixture_execution', 'Fixture JUnit unavailable; tests are not confirmed.')
    coverage = coverage_evidence(output)
    write_json(output / 'coverage-availability.json', coverage)
    passed = (all(value == 'success' for value in outcomes.values()) and report_available and report['artifact_preflight']
              and artifact_junit_available and artifact_markdown_available and fixtures_available and fixtures['passed']
              and coverage['status'] == 'available' and coverage['percent_covered'] >= 95)
    record = {'schema_version': '1.0', 'created_at_utc': created, 'status': 'passed' if passed else 'failed',
              'step_outcomes': outcomes, 'artifact_report_available': report_available,
              'artifact_junit_available': artifact_junit_available, 'artifact_markdown_available': artifact_markdown_available,
              'fixture_results_available': fixtures_available, 'coverage': coverage}
    summary = artifact_summary + '\n## CI execution and report availability\n\nCI result: **' + record['status'] + '**\n\n'
    summary += ('Original artifact JUnit: ' + ('available' if artifact_junit_available else 'unavailable') + '; '
                'original artifact Markdown: ' + ('available' if artifact_markdown_available else 'unavailable') + '.\n\n')
    summary += '\n'.join('- ' + step + ': ' + outcome for step, outcome in outcomes.items()) + '\n\n'
    summary += ('Fixture JUnit: ' + ', '.join(f'{key}: {value}' for key, value in fixtures['totals'].items()) + '\n\n'
                if fixtures_available else 'Fixture results unavailable; retained JUnit records a reporting failure.\n\n')
    summary += (f"Coverage of ci/ only: {coverage['percent_covered']:.2f}% (measured).\n" if coverage['status'] == 'available'
                else 'Scoped coverage unavailable; no measured percentage exists.\n')
    (output / 'summary.md').write_text(summary)
    write_json(output / 'finalization.json', record)
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
