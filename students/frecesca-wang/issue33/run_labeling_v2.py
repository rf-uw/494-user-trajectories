#!/usr/bin/env python3
"""
Issue #33: LLM Annotation v2
- Step 2: Predict hand labels using FULL POST text
- Step 3: Predict hand labels using ONLY COMMUNITY NOTE text (summary)
- Estimate average cost per tweet using tiktoken token counts.

Usage (from repo root):
  python students/frecesca-wang/issue33/run_labeling_v2.py

Outputs:
  students/frecesca-wang/issue33/outputs/
    set1_full_predictions.csv
    set1_note_predictions.csv
    set2_full_predictions.csv
    set2_note_predictions.csv
    summary_metrics.json
    failure_cases_set1_full.csv
    failure_cases_set1_note.csv
    failure_cases_set2_full.csv
    failure_cases_set2_note.csv
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from openai import OpenAI

# ---- Optional (cost estimation) ----
try:
    import tiktoken
except Exception:
    tiktoken = None


# -----------------------------
# Config
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]  # .../students/frecesca-wang/issue33/run_labeling_v2.py -> repo root
DATA_DIR = REPO_ROOT / "data"
ISSUE_DIR = REPO_ROOT / "students" / "frecesca-wang" / "issue33"
OUT_DIR = ISSUE_DIR / "outputs"

SET1_DATA = DATA_DIR / "cn_sample_1.csv"
SET2_DATA = DATA_DIR / "cn_sample_2.csv"
SET1_HAND = ISSUE_DIR / "hand_labels_set1.csv"
SET2_HAND = ISSUE_DIR / "hand_labels_set2.csv"

# Model: choose one you’re allowed to use. Keep it consistent across runs.
MODEL_NAME = "gpt-4.1-mini"

# Pricing changes over time; DO NOT hardcode as “truth”.
# Put your best-known prices here OR leave 0 and report token counts only.
# Units: dollars per 1M tokens
PRICE_INPUT_PER_1M = 0.0
PRICE_OUTPUT_PER_1M = 0.0

# How many failure cases to save per run
MAX_FAILURES_TO_SAVE = 40


LABELS = ["LEFT", "RIGHT", "CENTER", "MIXED", "NONE"]


# -----------------------------
# Prompt
# -----------------------------
PROMPT_TEMPLATE_FULL = """You are labeling the partisan lean of a social media post.

LABELS (choose exactly ONE):
- LEFT: supports progressive/left positions OR criticizes conservatives/Republicans/right-wing institutions.
- RIGHT: supports conservative/right positions OR criticizes liberals/Democrats/left-wing institutions.
- CENTER: political but mostly neutral reporting / informational update with no clear endorsement or attack.
- MIXED: contains clear left-leaning AND right-leaning signals.
- NONE: not about politics/ideology, or only a passing mention with no stance.

Decision rules:
1) If not political -> NONE.
2) If political: identify the target of praise/attack.
3) Use MIXED only when both sides are clearly present (not just uncertain).
4) If mostly factual/no stance -> CENTER.

POST:
{post_text}

Return ONLY one label from: LEFT, RIGHT, CENTER, MIXED, NONE.
"""

PROMPT_TEMPLATE_NOTE_ONLY = """You are labeling the partisan lean of a social media post, but you may ONLY use the text of the community note (not the post itself).

LABELS (choose exactly ONE):
- LEFT: the note frames the post in a way aligned with progressive/left critique OR criticizes conservatives/Republicans/right-wing institutions.
- RIGHT: the note frames the post in a way aligned with conservative/right critique OR criticizes liberals/Democrats/left-wing institutions.
- CENTER: neutral fact-checking or informational tone with no partisan framing.
- MIXED: contains clear left and right partisan framing signals.
- NONE: no political content.

COMMUNITY NOTE TEXT:
{note_text}

Return ONLY one label from: LEFT, RIGHT, CENTER, MIXED, NONE.
"""


# -----------------------------
# Helpers
# -----------------------------
def load_api_key() -> None:
    """
    Loads secrets/OPENAIKEY.txt into OPENAI_API_KEY if not already set.
    """
    if os.getenv("OPENAI_API_KEY"):
        return
    key_path = REPO_ROOT / "secrets" / "OPENAIKEY.txt"
    if not key_path.exists():
        raise FileNotFoundError(f"Missing API key file: {key_path}\nCreate it from secrets/OPENAIKEY.txt.template")
    os.environ["OPENAI_API_KEY"] = key_path.read_text(encoding="utf-8").strip()


def normalize_label(x: str) -> str:
    x = (x or "").strip().upper()
    # Extract first valid label if the model adds text
    for lab in LABELS:
        if re.search(rf"\b{lab}\b", x):
            return lab
    return "INVALID"


def token_count(model: str, text: str) -> int:
    if tiktoken is None:
        return 0
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text or ""))


@dataclass
class RunMetrics:
    n: int
    accuracy: float
    invalid_rate: float
    avg_input_tokens: float
    avg_output_tokens: float
    avg_total_tokens: float
    avg_cost_usd: float


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    # dollars
    return (input_tokens / 1_000_000.0) * PRICE_INPUT_PER_1M + (output_tokens / 1_000_000.0) * PRICE_OUTPUT_PER_1M


def read_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect columns: post_id, full_text, summary
    need = {"post_id", "full_text", "summary"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}. Found: {list(df.columns)}")
    return df


def read_handlabels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"post_id", "hand_label"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}. Found: {list(df.columns)}")
    df["hand_label"] = df["hand_label"].astype(str).str.strip().str.upper()
    return df


def merge_labels(data_df: pd.DataFrame, hand_df: pd.DataFrame) -> pd.DataFrame:
    merged = data_df.merge(hand_df, on="post_id", how="inner")
    if len(merged) != len(hand_df):
        # This warns if IDs don't match 1:1
        print(f"[WARN] merged rows={len(merged)} != hand rows={len(hand_df)}. Some post_id may not match.")
    return merged


def llm_label(client: OpenAI, model: str, prompt: str) -> Tuple[str, int, int]:
    """
    Returns: (label, input_tokens, output_tokens)
    """
    in_tok = token_count(model, prompt)
    resp = client.responses.create(
        model=model,
        input=prompt,
    )
    out_text = getattr(resp, "output_text", "") or ""
    out_tok = token_count(model, out_text)
    return normalize_label(out_text), in_tok, out_tok


def run_one_mode(
    client: OpenAI,
    df: pd.DataFrame,
    mode: str,
    out_csv: Path,
    failures_csv: Path,
) -> RunMetrics:
    """
    mode: "full" uses df.full_text; "note" uses df.summary
    """
    rows: List[Dict] = []
    failures: List[Dict] = []

    total = len(df)
    invalid = 0
    correct = 0

    sum_in = 0
    sum_out = 0
    sum_cost = 0.0

    for _, r in tqdm(df.iterrows(), total=total, desc=f"Labeling ({mode})"):
        post_id = r["post_id"]
        gold = str(r["hand_label"]).strip().upper()

        if mode == "full":
            content = str(r["full_text"])
            prompt = PROMPT_TEMPLATE_FULL.format(post_text=content)
        elif mode == "note":
            content = str(r["summary"])
            prompt = PROMPT_TEMPLATE_NOTE_ONLY.format(note_text=content)
        else:
            raise ValueError("mode must be 'full' or 'note'")

        pred, in_tok, out_tok = llm_label(client, MODEL_NAME, prompt)
        cost = estimate_cost_usd(in_tok, out_tok)

        sum_in += in_tok
        sum_out += out_tok
        sum_cost += cost

        if pred == "INVALID":
            invalid += 1
        if pred == gold:
            correct += 1
        else:
            if len(failures) < MAX_FAILURES_TO_SAVE:
                failures.append(
                    {
                        "post_id": post_id,
                        "gold": gold,
                        "pred": pred,
                        "mode": mode,
                        "text_used": content,
                    }
                )

        rows.append(
            {
                "post_id": post_id,
                "hand_label": gold,
                "pred_label": pred,
                "mode": mode,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": in_tok + out_tok,
                "estimated_cost_usd": cost,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)

    if failures:
        pd.DataFrame(failures).to_csv(failures_csv, index=False)
    else:
        pd.DataFrame(columns=["post_id", "gold", "pred", "mode", "text_used"]).to_csv(failures_csv, index=False)

    acc = correct / total if total else 0.0
    invalid_rate = invalid / total if total else 0.0

    avg_in = (sum_in / total) if total else 0.0
    avg_out = (sum_out / total) if total else 0.0
    avg_total = avg_in + avg_out
    avg_cost = (sum_cost / total) if total else 0.0

    return RunMetrics(
        n=total,
        accuracy=acc,
        invalid_rate=invalid_rate,
        avg_input_tokens=avg_in,
        avg_output_tokens=avg_out,
        avg_total_tokens=avg_total,
        avg_cost_usd=avg_cost,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    load_api_key()
    client = OpenAI()

    # Load + merge
    set1 = merge_labels(read_dataset(SET1_DATA), read_handlabels(SET1_HAND))
    set2 = merge_labels(read_dataset(SET2_DATA), read_handlabels(SET2_HAND))

    metrics_all: Dict[str, Dict] = {}

    def do_set(set_name: str, df: pd.DataFrame):
        # Step 2: FULL
        m_full = run_one_mode(
            client=client,
            df=df,
            mode="full",
            out_csv=OUT_DIR / f"{set_name}_full_predictions.csv",
            failures_csv=OUT_DIR / f"failure_cases_{set_name}_full.csv",
        )
        # Step 3: NOTE ONLY
        m_note = run_one_mode(
            client=client,
            df=df,
            mode="note",
            out_csv=OUT_DIR / f"{set_name}_note_predictions.csv",
            failures_csv=OUT_DIR / f"failure_cases_{set_name}_note.csv",
        )

        metrics_all[set_name] = {
            "full": m_full.__dict__,
            "note_only": m_note.__dict__,
        }

        print(f"\n=== {set_name.upper()} RESULTS ===")
        print(f"FULL:      n={m_full.n}  acc={m_full.accuracy:.3f}  invalid={m_full.invalid_rate:.3f}  "
              f"avg_tokens(in/out/total)={m_full.avg_input_tokens:.1f}/{m_full.avg_output_tokens:.1f}/{m_full.avg_total_tokens:.1f}  "
              f"avg_cost=${m_full.avg_cost_usd:.6f}")
        print(f"NOTE ONLY: n={m_note.n}  acc={m_note.accuracy:.3f}  invalid={m_note.invalid_rate:.3f}  "
              f"avg_tokens(in/out/total)={m_note.avg_input_tokens:.1f}/{m_note.avg_output_tokens:.1f}/{m_note.avg_total_tokens:.1f}  "
              f"avg_cost=${m_note.avg_cost_usd:.6f}")

    do_set("set1", set1)
    do_set("set2", set2)

    # Save summary
    summary_path = OUT_DIR / "summary_metrics.json"
    summary_path.write_text(json.dumps({
        "model": MODEL_NAME,
        "price_input_per_1m": PRICE_INPUT_PER_1M,
        "price_output_per_1m": PRICE_OUTPUT_PER_1M,
        "note": "If prices are 0, report token counts and explain pricing not filled.",
        "metrics": metrics_all,
    }, indent=2), encoding="utf-8")

    print(f"\nWrote summary: {summary_path}")
    print(f"Predictions + failures in: {OUT_DIR}")


if __name__ == "__main__":
    main()