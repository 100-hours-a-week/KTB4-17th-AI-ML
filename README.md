# KTB4-17th-AI-ML

소개팅 대화 연습용 **페르소나 모델(Kanana-2-1.3b-persona-attach-v1)**의 LoRA 어댑터와, 서버 배포용 9칸 어댑터를 만드는 코드.

**Powered by Kanana.** 가중치는 git 에 두지 않고 RunPod S3 로 배포한다. 이 저장소는 **어떻게 만들고 어떻게 실행되는지**를 관리한다.

## 1. 폴더

```
adapters/                원본 LoRA 어댑터 4개 (rank 16, 사용법: adapters/README.md)
├─ guard/                가드레일 v5 (SFT)        — 원본 Kanana 1.3B 위에서 학습
├─ secure/               안정형 (회피 낮춤, DPO)   — 가드레일 병합 모델 위에서 학습
├─ avoid/                회피 (회피 높임, DPO)     — 〃
└─ anxiety/              불안 (불안 높임, DPO)     — 〃
scripts/build_cells.py   위 4개로 서버 배포용 9칸 LoRA 만들기
```

## 2. 모델 구성: 9칸

회피 3단계 × 불안 3단계 = 9가지 성향. 칸마다 **원본 Kanana 1.3B 위에 바로 얹는 LoRA 하나**로 배포한다.

| 회피 \ 불안 | 낮음 (불안 α 0) | 중간 (α 0.75) | 높음 (α 1) |
| --- | --- | --- | --- |
| 낮음 — 안정형 | `attach9-low-anx0` | `attach9-low-anx075` | `attach9-low-anx1` |
| 중간 — 기본 | `attach9-mid-anx0` | `attach9-mid-anx075` | `attach9-mid-anx1` |
| 높음 — 거리 두기 | `attach9-high-anx0` | `attach9-high-anx075` | `attach9-high-anx1` |

칸 하나는 원본 어댑터들을 계수(α)로 더한 것이다.

```
ΔW_칸 = 1·ΔW_guard + α_회피·ΔW_(secure 또는 avoid) + α_불안·ΔW_anxiety

      = [B_guard | α·B_회피 | α·B_불안] × [A_guard ; A_회피 ; A_불안]
         B 는 옆으로 이어 붙임             A 는 아래로 이어 붙임
```

- 이어 붙인 행렬의 곱은 원래 변화량의 합과 **같다** (버리는 방향이 없음, bf16 저장 반올림 0.23% 만 있음).
- 그래서 rank 는 들어간 어댑터 수 × 16 이다: `mid-anx0` 16 / 어댑터 2개 칸 32 / 3개 칸 48.
- **가드레일을 칸마다 넣은 이유**: 원래 애착 어댑터는 가드레일 병합 모델 위에서 학습해서, 원본 Kanana 에 애착 어댑터만 얹으면 가드레일(AI 정체 숨기기·지어내지 않기·기억 처리)이 빠진다.
- 예전 방식(가드레일 병합 모델 + 애착 LoRA)과 비교: 9칸 × 질문 6개에서 1등 토큰 일치 98.5~100%, 토큰 log 확률 차 평균 0.005~0.007 (vLLM 과 transformers 차이와 같은 크기). 같은 모델로 본다.

### 칸별 성향 (2026-11-06, 학습에 안 쓴 질문 84개 × 3답, Claude Sonnet 블라인드 판정)

| 칸 | 회피 | 불안 | 판정 탈락 | 가드레일 규칙 | 특징 |
| --- | --- | --- | --- | --- | --- |
| `low-anx0` | 2.18 | 1.19 | 10% | 100% | 안정형. 마음을 편하게 열고 연락 공백에도 흔들리지 않음 |
| `low-anx075` | 2.06 | 1.83 | 14% | 98.0% | 다정하고 관계를 챙김 |
| `low-anx1` | 2.04 | 2.08 | 24% | 97.0% | 다가가며 확인을 구함 (몰입형). 탈락이 가장 높음 |
| `mid-anx0` | 2.94 | 1.23 | 7% | 99.7% | **기본**. 가드레일만. 탈락이 가장 낮음 |
| `mid-anx075` | 2.58 | 1.42 | 11% | 97.0% | 기본 + 가끔 "부담 드린 건 아니죠?" |
| `mid-anx1` | 2.44 | 2.14 | 12% | 94.4% | 불안형. 관계 확인이 잦음 |
| `high-anx0` | 3.33 | 1.03 | 12% | 98.7% | 회피형. 짧게 답하고 개인 얘기를 피함 |
| `high-anx075` | 2.89 | 1.36 | 8% | 97.0% | 거리 두되 미안해함 |
| `high-anx1` | 2.71 | 2.04 | 17% | 95.1% | 두려움형. 거리 두면서 연락은 붙잡음 |

점수는 1~5, 칸끼리 비교하는 상대값이다. 판정 탈락은 맥락 오해·깨진 글처럼 대화 상대로 부적절한 답의 비율, 가드레일 규칙은 규칙 평가 304문제 통과율.

## 3. 배포 흐름

```
① 이 저장소 adapters/ (원본 4개)
      │  python scripts/build_cells.py --out build
      ▼
② build/attach9-*/  칸마다 서빙 파일 5개
      │  publish.py --adapter (RunPod 네트워크 볼륨)
      ▼
③ RunPod S3   releases/<모델 이름>/<릴리스ID>/  +  release-markers/<모델 이름>.<릴리스ID>._READY
      │  AWS 가 _READY 를 감지 (자동)
      ▼
④ AWS EC2 · GPU 추론 서버 (vLLM)
      ├─ Kanana 베이스 모델      ← 서버가 직접 띄움 (S3 로 오지 않음)
      └─ 요청의 model 이름으로 LoRA 선택 → 베이스에 얹어 추론
      ▼
⑤ FastAPI 가 model=attach9-high-anx075 처럼 칸 이름으로 호출
```

### 3-1. 만들기

```bash
pip install torch safetensors huggingface_hub
python scripts/build_cells.py --out build --verify     # 9칸, CPU 로 수십 초
python scripts/build_cells.py --out build --cells attach9-mid-anx0,attach9-high-anx1   # 일부만
```

칸마다 `adapter_model.safetensors`, `adapter_config.json`(베이스 `kakaocorp/kanana-2-1.3b-instruct`), `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` 가 생긴다. 토크나이저 3개는 원본 Kanana 1.3B 저장소에서 받는다. `build/` 는 git 에 넣지 않는다.

### 3-2. 올리기 (RunPod 네트워크 볼륨을 붙인 Pod 에서)

```bash
for d in build/attach9-*; do
  python /workspace/mlops/bin/publish.py "$(basename "$d")" "$d" --adapter
done
```

`_READY` 마커가 생기는 순간 AWS 쪽 자동 배포가 시작된다. **아래 4장 조건을 서버가 갖춘 뒤에 올린다.**

## 4. 추론 서버 조건 (배포 전 확인)

| 항목 | 필요한 값 | 이유 |
| --- | --- | --- |
| 베이스 모델 | `kakaocorp/kanana-2-1.3b-instruct` | LoRA 는 학습한 베이스 위에서만 동작한다. 지금 서버의 다른 어댑터(extraction-v1, tagging-v1)는 **`kanana-2-3b-instruct`** 기준이라, 9칸은 1.3B 베이스를 띄운 vLLM 이 따로 필요하다 (1.3B bf16 은 약 3GB) |
| `--max-lora-rank` | 48 이상 | 칸 rank 가 16·32·48 |
| `--trust-remote-code` | 켬 | Kanana-2 1.3B 는 모델 구조 코드(`modeling_kanana2_tiny.py`)를 저장소에서 불러온다 |

## 5. 호출

OpenAI 호환 API. `model` 값에 칸 이름을 넣으면 그 칸 성향으로 답한다.

```python
resp = client.chat.completions.create(
    model="attach9-mid-anx0",
    messages=[{"role": "system", "content": system_prompt}, *history],  # 마지막은 user
    temperature=0.7,
    max_tokens=160,
)
```

| 항목 | 값 | 이유 |
| --- | --- | --- |
| `temperature` | 0.7 | 평가와 같은 값. 0 이면 불안 높은 칸에서 같은 말을 끝없이 반복하는 경우가 드물게 생김 |
| `max_tokens` | 160 | 칸 평균 답은 20~37 토큰 |
| `repetition_penalty` | 넣지 않음 | 넣으면 답이 16~29% 길어져 칸 성향이 바뀜 |
| 시스템 프롬프트 | 학습 때의 compact 프롬프트 (프로필 + 규칙) | 애착 성향 줄은 넣지 않는다. 성향은 칸으로 고른다 |
| 최대 입력 | 8,192 토큰 | 시스템 프롬프트 + 대화 10턴 + 과거 대화 검색 결과 |

## 6. 한계

- 위기 발언(자해·자살)과 만남·통화·연락처 요청은 학습하지 않았다. 서비스 프롬프트와 서버 감지로 처리한다.
- 불안 α 1 칸은 판정 탈락이 12~24% 로 높다 (`low-anx1` 이 가장 높음).
- 불안 α 가 클수록 상대가 틀리게 말한 세부를 바로잡지 못하는 경우가 늘어난다.

## 7. 라이선스

Kanana Open License(`kakaocorp/kanana-2-1.3b-instruct`)를 따른다. 파생 모델로서 "Kanana" 이름 접두사와 "Powered by Kanana" 표시를 유지한다. 온디바이스에 넣어 제3자에게 제공하려면 카카오의 별도 상업 라이선스가 필요하다 (4.1(iii)).
