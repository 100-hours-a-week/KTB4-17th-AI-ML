"""애착 9칸 서빙용 LoRA 만들기.

저장소의 원본 어댑터 4개(adapters/guard·secure·avoid·anxiety)로, 원본 Kanana 1.3B
(kakaocorp/kanana-2-1.3b-instruct) 위에 바로 얹는 칸별 LoRA 하나씩을 만든다.

칸 하나의 가중치 변화량:
    ΔW = 1·ΔW_guard + α_avoid·ΔW_(secure|avoid) + α_anx·ΔW_anxiety
    (각 ΔW = lora_alpha / r · B @ A)
A 는 세로로, (α·scaling 을 곱한) B 는 가로로 이어 붙여 LoRA 하나로 만든다.
→ rank = 16 × 들어간 어댑터 수 (16·32·48), lora_alpha = rank (scaling 1).
원래 어댑터들의 합과 같다 (bf16 저장 반올림만 있음).

출력: <out>/<모델 이름>/ 에 RunPod S3 publish.py --adapter 가 받는 5개 파일
    adapter_model.safetensors, adapter_config.json,
    tokenizer.json, tokenizer_config.json, chat_template.jinja

    pip install torch safetensors huggingface_hub
    python scripts/build_cells.py --out build            # 9칸
    python scripts/build_cells.py --out build --verify   # + 모든 모듈 검산
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
BASE = "kakaocorp/kanana-2-1.3b-instruct"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")

# 칸 이름: 서빙 모델 이름 (publish.py 규칙: 소문자·숫자·-·_ 만, 점 불가)
# 값: (회피 쪽 어댑터, 회피 α, 불안 α)
CELLS = {
    "attach9-low-anx0": ("secure", 1.0, 0.0),
    "attach9-low-anx075": ("secure", 1.0, 0.75),
    "attach9-low-anx1": ("secure", 1.0, 1.0),
    "attach9-mid-anx0": (None, 0.0, 0.0),
    "attach9-mid-anx075": (None, 0.0, 0.75),
    "attach9-mid-anx1": (None, 0.0, 1.0),
    "attach9-high-anx0": ("avoid", 1.0, 0.0),
    "attach9-high-anx075": ("avoid", 1.0, 0.75),
    "attach9-high-anx1": ("avoid", 1.0, 1.0),
}


def read_adapter(name: str) -> tuple[dict, dict[str, torch.Tensor]]:
    d = ROOT / "adapters" / name
    cfg = json.loads((d / "adapter_config.json").read_text())
    return cfg, load_file(str(d / "adapter_model.safetensors"))


def modules(w: dict) -> list[str]:
    return sorted({k.rsplit(".lora_", 1)[0] for k in w})


def delta(cfg: dict, w: dict, mod: str, coef: float = 1.0) -> torch.Tensor:
    a = w[f"{mod}.lora_A.weight"].float()
    b = w[f"{mod}.lora_B.weight"].float()
    return b @ a * (cfg["lora_alpha"] / cfg["r"] * coef)


def combine(parts: list[tuple[dict, dict, float]]) -> dict[str, torch.Tensor]:
    """[(config, weights, α)] → LoRA 하나. α·scaling 은 B 에 곱해 둔다."""
    out = {}
    for mod in modules(parts[0][1]):
        a_list, b_list = [], []
        for cfg, w, coef in parts:
            s = coef * cfg["lora_alpha"] / cfg["r"]
            a_list.append(w[f"{mod}.lora_A.weight"].float())
            b_list.append(w[f"{mod}.lora_B.weight"].float() * s)
        out[f"{mod}.lora_A.weight"] = torch.cat(a_list, 0).to(torch.bfloat16)
        out[f"{mod}.lora_B.weight"] = torch.cat(b_list, 1).to(torch.bfloat16)
    return {k: v.contiguous() for k, v in out.items()}


def max_rel_err(parts, cfg: dict, w: dict) -> float:
    errs = []
    for m in modules(w):
        want = sum(delta(c, x, m, a) for c, x, a in parts)
        errs.append(float((delta(cfg, w, m) - want).norm() / want.norm()))
    return max(errs)


def tokenizer_dir() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(BASE, allow_patterns=list(TOKENIZER_FILES)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cells", default="", help="쉼표로 구분한 칸 이름 (기본: 9칸)")
    ap.add_argument("--verify", action="store_true", help="모든 모듈에서 ΔW 검산")
    args = ap.parse_args()

    names = [c for c in args.cells.split(",") if c] or list(CELLS)
    adapters = {n: read_adapter(n) for n in ("guard", "secure", "avoid", "anxiety")}
    guard_cfg = adapters["guard"][0]
    assert guard_cfg["base_model_name_or_path"] == BASE
    tok = tokenizer_dir()

    for name in names:
        avoid, a_avoid, a_anx = CELLS[name]
        used = [("guard", 1.0)]
        used += [(avoid, a_avoid)] if avoid else []
        used += [("anxiety", a_anx)] if a_anx else []
        parts = [(*adapters[n], a) for n, a in used]
        weights = combine(parts)
        r = 16 * len(parts)
        cfg = {**guard_cfg, "r": r, "lora_alpha": r, "lora_dropout": 0.0}
        cfg.pop("alpha_pattern", None)
        cfg.pop("rank_pattern", None)

        d = args.out / name
        d.mkdir(parents=True, exist_ok=True)
        save_file(weights, str(d / "adapter_model.safetensors"), {"format": "pt"})
        (d / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
        for f in TOKENIZER_FILES:
            shutil.copy(tok / f, d / f)

        info = f"{name}: rank {r}, " + " + ".join(f"{n}×{a:g}" for n, a in used)
        if args.verify:
            info += f", 검산 최대 상대 오차 {max_rel_err(parts, cfg, weights):.1e}"
        print(info)


if __name__ == "__main__":
    main()
