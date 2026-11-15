# Model pre-training CI

The `Model pre-training CI` workflow checks the four tracked adapters in `dev` (`guard`, `secure`, `anxiety`, `avoid`) on a hosted Ubuntu CPU runner with Python 3.12. It runs for relevant pull requests, relevant pushes to `dev` and `codex/**`, and manual dispatch. Existing app CI/CD is outside this workflow. Official actions are pinned to reviewed commit SHAs. The job has read-only contents permission, a 20-minute timeout and cancellation of superseded runs. It needs no secrets.

Run locally with an isolated environment:

```sh
python3.12 -m venv .venv-ci
.venv-ci/bin/python -m pip install -r ci/requirements.txt
.venv-ci/bin/python -m compileall -q ci tests/pretraining
.venv-ci/bin/python -m pytest tests/pretraining -q --junitxml=reports/tests.junit.xml --cov=ci --cov-config=ci/coverage.ini --cov-fail-under=95 --cov-report=term-missing --cov-report=xml:reports/coverage.xml --cov-report=json:reports/coverage.json
.venv-ci/bin/python ci/preflight.py --root . --output reports/preflight
```

The validator is standard-library only. Pytest and coverage are CI tools; this workflow does not install training dependencies, torch or Hugging Face clients. All direct and transitive CI dependencies are pinned in `ci/requirements.txt`.

## What the checks establish

Each expected adapter requires `adapter_config.json` and `adapter_model.safetensors`. Configuration checks validate causal LM LoRA type, positive finite alpha and rank, unique target modules, and supported adapter conventions. Per-module rank/alpha overrides, partial layers, additional saved modules and adapter variations such as DoRA are rejected rather than assumed compatible.

The safetensors parser bounds config files to 1 MiB, headers to 8 MiB and tensor count to 100,000. It rejects duplicate JSON keys and non-standard NaN/Infinity JSON constants. Before any scalar scan, it checks descriptor keys, floating dtype, two positive bounded dimensions, exact shape byte size, offsets inside the actual payload, contiguous non-overlapping coverage with no trailing bytes, A/B rank and pairs, contiguous layer indices, and configured target coverage for every layer. Supported scalar dtypes are F32, F64, F16 and BF16. Unsupported dtypes fail closed. Scalar scans and SHA256 hashing use chunks of at most 1 MiB, without allocating full tensors or importing model code. Layers are checked for contiguity and consistency; without an immutable base config the validator cannot prove that the adapter covers the intended base architecture's entire layer set.

Across adapters, tensor names, shapes, dtypes, ranks, alpha values and targets must match. Base names and historical absolute paths are recorded as provenance. The path `/root/ws/guard5_merged` is not required to exist and is not treated as an exact checkpoint fingerprint. A matching schema does not prove that all adapters were trained from compatible checkpoint contents.

A successful exit means `artifact_preflight=true`: local artifact structure, compatibility and finiteness passed. It does not measure accuracy, safety, model quality or training readiness. `training_ready=false` remains explicit because this repository does not define a reviewed training entrypoint, dataset manifest, training hyperparameters, training dependency lock, exact base checkpoint fingerprint or exact merged guard checkpoint fingerprint. `readiness_reasons` records those six gaps. These prerequisites require a future reviewed contract; adding an arbitrary file does not automatically enable training.

## Reports and failure behavior

`reports/preflight/preflight.json` includes schema version, UTC timestamp, repository/commit/run identity, per-check status/duration/errors/warnings/skips, file bytes/SHA256, tensor and parameter counts, dtype counts, layers, targets, rank, alpha, checked scalar counts and nonfinite counts. Aggregate `weight_bytes` measures complete tracked weight files; `artifact_bytes_scanned` measures scalar payload bytes and excludes headers. Missing or corrupt artifacts produce a nonzero exit and failed checks; dependent payload scans are explicitly skipped. Valid structure with nonfinite values is a failed preflight.

`preflight.junit.xml` represents actual artifact checks. `tests.junit.xml` represents synthetic fixture regressions. Coverage XML/JSON measures only `ci/`, including the CLI subprocesses; the 95% threshold is validator coverage, not model or training coverage. Markdown summarizes measured counts, check diagnostics, prerequisite gaps, fixture results and scoped coverage. Actions always attempts the actual audit, summary publication and report upload, including after failed tests, and retains reports for 14 days. Failure exits are preserved.

Reports contain no raw tensors, dataset examples, prompts, completions, environment dumps or credentials. Identity reads only the named GitHub repository/commit/run/workflow variables. `training_executed=false`, `gpu_used=false` and `external_model_downloads=0` describe this preflight's fixed scope. The legacy `model_downloads=0` field also means **external** downloads: tracked adapter weight files arrive with the source checkout and their measured bytes are reported. The workflow never invokes training, model inference, teacher APIs, tokenizer/model download APIs, adapter composition, checkpoint writes, AWS or CD.

The initial local baseline at dev commit `8bd60424a4b55725d627353bdeb8ad0df61edc67` measured 4 weight files totaling 299,082,336 bytes, 448 F32 tensors / 224 A/B pairs / 18,677,760 parameters per adapter, 32 layers, 7 targets, rank 16 and alpha 32. A full scan found 0 nonfinite values across 74,711,040 scalars. These are observations, not hardcoded pass conditions; future reports derive metrics from current files.
