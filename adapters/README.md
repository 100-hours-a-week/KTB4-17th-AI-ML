# Kanana-2-1.3b-persona-attach-v1 LoRA 어댑터

**Powered by Kanana.** 소개팅 대화 연습용 페르소나 모델 1차 버전(2026-11-06)의 LoRA 어댑터다.

`kakaocorp/kanana-2-1.3b-instruct`에 가드레일 LoRA(SFT)를 병합한 모델 위에 애착 2축(회피·불안) DPO LoRA를 계수(α)로 합쳐 9가지 성향(3 × 3)을 만든다.

## 구성

| 폴더 | 내용 | 얹는 모델 |
| --- | --- | --- |
| `guard/` | 가드레일 v5 LoRA (SFT) | `kakaocorp/kanana-2-1.3b-instruct` |
| `secure/` | 안정형 DPO LoRA (회피 낮음) | 가드레일 병합 모델 |
| `avoid/` | 회피 DPO LoRA (회피 높음) | 가드레일 병합 모델 |
| `anxiety/` | 불안 DPO LoRA | 가드레일 병합 모델 |

모두 rank 16, lora_alpha 32이고, 대상 모듈은 q·k·v·o·gate·up·down이다.

`secure`·`avoid`·`anxiety`의 `adapter_config.json`에는 `base_model_name_or_path`가 학습 서버 경로(`/root/ws/guard5_merged`)로 적혀 있다. 아래처럼 가드레일 병합 모델을 먼저 만든 뒤 그 위에 얹으면 된다.

## 사용법

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "kakaocorp/kanana-2-1.3b-instruct"
tok = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype="bfloat16")

# 1) 가드레일 병합 (9칸 모두의 기본 모델)
merged = PeftModel.from_pretrained(base, "adapters/guard").merge_and_unload()
merged.save_pretrained("guard5_merged"); tok.save_pretrained("guard5_merged")

# 2) 애착 어댑터를 α로 합쳐 칸 만들기 (예: 회피 높음 × 불안 0.75)
m = PeftModel.from_pretrained(merged, "adapters/avoid", adapter_name="avoid")
m.load_adapter("adapters/anxiety", adapter_name="anxiety")
m.add_weighted_adapter(["avoid", "anxiety"], [1.0, 0.75], "high_anx0.75", combination_type="cat")
m.set_adapter("high_anx0.75")
```

프롬프트는 학습 때와 같은 compact 시스템 프롬프트(프로필 + 규칙)를 쓴다. 애착 관련 프로필 줄은 넣지 않는다(성향은 칸으로 고른다).

## 9칸

| 회피 \ 불안 | 낮음 (α 0) | 중간 (α 0.75) | 높음 (α 1) |
| --- | --- | --- | --- |
| 낮음 (`secure` α 1) | `low_anx0` | `low_anx0.75` | `low_anx1` |
| 중간 (어댑터 없음) | 가드레일 병합 모델 그대로 | `mid_anx0.75` | `mid_anx1` |
| 높음 (`avoid` α 1) | `high_anx0` | `high_anx0.75` | `high_anx1` |

## 평가 (2026-11-06)

학습에 쓰지 않은 질문 84개 × 답 3개를 Claude Sonnet이 블라인드로 판정했다. 회피·불안 점수는 1~5이며, 높을수록 그 성향이 강하다. 판정 탈락은 맥락 오해나 깨진 글처럼 대화 상대로 부적절한 답이다(만남 제안은 서비스 프롬프트에서 처리하므로 제외). 규칙은 가드레일 규칙 평가 304문제 통과율이다. 기억은 진짜 기억 / 틀린 세부 바로잡기 결과다(각 18문제).

| 칸 | 회피 | 불안 | 판정 탈락 | 규칙 | 기억 |
| --- | --- | --- | --- | --- | --- |
| 낮음 × 0 | 2.18 | 1.19 | 10% | 100% | 17 / 17 |
| 낮음 × 0.75 | 2.06 | 1.83 | 14% | 98.0% | 17 / 16 |
| 낮음 × 1 | 2.04 | 2.08 | 24% | 97.0% | 18 / 15 |
| 중간 × 0 (기본) | 2.94 | 1.23 | 7% | 99.7% | 17 / 15 |
| 중간 × 0.75 | 2.58 | 1.42 | 11% | 97.0% | 18 / 12 |
| 중간 × 1 | 2.44 | 2.14 | 12% | 94.4% | 17 / 10 |
| 높음 × 0 | 3.33 | 1.03 | 12% | 98.7% | 15 / 18 |
| 높음 × 0.75 | 2.89 | 1.36 | 8% | 97.0% | 17 / 10 |
| 높음 × 1 | 2.71 | 2.04 | 17% | 95.1% | 16 / 8 |

## 한계

- 위기 발언(자해·자살)과 만남·통화·연락처 요청은 학습하지 않았다. 서비스 프롬프트와 서버 감지로 처리해야 한다.
- 불안 α 1 칸은 판정 탈락이 12~24%다. 특히 낮음 × 1에서 많다.
- 불안 α가 클수록 상대가 틀리게 말한 세부를 바로잡지 못하는 경우가 늘어난다.
- 회피 높음(α 1)은 중간과 차이가 작다(+0.3~0.4).

## 라이선스

Kanana Open License(`kakaocorp/kanana-2-1.3b-instruct`)를 따른다. 파생 모델로서 "Kanana" 이름 접두사와 "Powered by Kanana" 표시를 유지한다. 온디바이스에 넣어 제3자에게 제공하려면 카카오의 별도 상업 라이선스가 필요하다(4.1(iii)). 학습 데이터 생성에는 kanana-2-30b-a3b-instruct(Kanana License)를 교사 모델로 썼다.
