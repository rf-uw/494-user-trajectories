import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import re
    import os
    import polars as pl
    from openai import OpenAI
    from tqdm import tqdm
    return OpenAI, mo, os, pl, re, tqdm


@app.cell
def _(OpenAI, os):
    with open("secrets/OPENAIKEY.txt", "r") as f:
        os.environ["OPENAI_API_KEY"] = f.read().strip()
    client = OpenAI()
    return (client,)


@app.cell
def _(pl):
    # Load BOTH labeled datasets
    # (make sure you put cn_sample_2_labeled.csv into students/emmazhang/)
    df1 = pl.read_csv("students/emmazhang/cn_sample_1_labeled.csv")
    df2 = pl.read_csv("students/emmazhang/cn_sample_2_labeled.csv")
    return df1, df2


@app.cell
def _(df1, df2):
    # Peek at data
    df1.head(3), df2.head(3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Labeling
    """)
    return


@app.cell
def _():
    SIMPLE_PROMPT_TEMPLATE_FULL = SIMPLE_PROMPT_TEMPLATE_FULL = """You are labeling the partisan lean of a social media post.

# DECISION PROCESS (FOLLOW IN ORDER)

1) Is this post about politics, public policy, elections, political ideology, or political actors?
   - If NO → output NONE.

2) If YES:
   Does the post express support, praise, criticism, blame, sarcasm, or attack toward a political side?
   - If YES → choose LEFT or RIGHT.
   - If it attacks both sides → MIXED.
   - If it is purely factual or neutral → CENTER.

# LABEL DEFINITIONS

- LEFT: supports progressive/left positions OR criticizes conservatives/Republicans/right-wing actors.
- RIGHT: supports conservative/right positions OR criticizes liberals/Democrats/left-wing actors.
- CENTER: about politics but purely neutral/factual with no endorsement or attack.
- MIXED: contains BOTH left and right framing.
- NONE: not about politics or ideology.

# IMPORTANT
Do NOT guess.
Do NOT default to CENTER.
If unsure whether political → choose NONE.

# TWEET
{content}

# OUTPUT (no extra text)
<output>
LEFT|CENTER|RIGHT|MIXED|NONE
</output>
"""


    SIMPLE_PROMPT_TEMPLATE_NOTE = """You are labeling the partisan lean of a social media post.
    If the community note is purely a factual correction without partisan framing → output CENTER.
    If it is not political → output NONE.

    # LABELS (choose exactly ONE)
    - LEFT: supports progressive/left positions OR criticizes conservatives/Republicans/right-wing institutions.
    - RIGHT: supports conservative/right positions OR criticizes liberals/Democrats/left-wing institutions.
    - CENTER: mainly factual reporting, neutral informational updates, or balanced non-partisan wording with no clear endorsement/attack.
    - MIXED: contains BOTH left-leaning and right-leaning signals.
    - NONE: not about politics or ideology.

    # IMPORTANT
    You MUST label using ONLY the community note text below.
    Do NOT assume anything about the original post.

    # COMMUNITY NOTE TEXT
    {content}

    # OUTPUT (no extra text)
    <output>
    LEFT|CENTER|RIGHT|MIXED|NONE
    </output>
    """
    return SIMPLE_PROMPT_TEMPLATE_FULL, SIMPLE_PROMPT_TEMPLATE_NOTE


@app.cell
def _(
    SIMPLE_PROMPT_TEMPLATE_FULL,
    SIMPLE_PROMPT_TEMPLATE_NOTE,
    client,
    pl,
    re,
    tqdm,
):
    MODEL = "gpt-4.1-mini"

    def _parse_output(output_text: str) -> str:
        text = (output_text or "").strip()
        m = re.search(r"<output>\s*(.*?)\s*</output>", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()
        text = text.strip().upper()

        allowed = {"LEFT", "RIGHT", "CENTER", "MIXED", "NONE"}
        if text not in allowed:
            for lab in ["LEFT", "RIGHT", "CENTER", "MIXED", "NONE"]:
                if re.search(rf"\b{lab}\b", text):
                    return lab
            return "NONE"
        return text

    def _query_llm(prompt: str) -> str:
        resp = client.responses.create(model=MODEL, input=prompt, temperature=0)
        return getattr(resp, "output_text", "") or ""

    def run_labeling(df):
        results = []
        for row in tqdm(df.iter_rows(named=True), total=df.height):
            full_text = row["full_text"]
            note_text = row["summary"]

            # Step 2: Full text
            prompt_full = SIMPLE_PROMPT_TEMPLATE_FULL.format(content=full_text)
            out_full = _query_llm(prompt_full)
            pred_full = _parse_output(out_full)

            # Step 3: Note only
            prompt_note = SIMPLE_PROMPT_TEMPLATE_NOTE.format(content=note_text)
            out_note = _query_llm(prompt_note)
            pred_note = _parse_output(out_note)

            result_row = {
                **row,
                "model_label_fulltext": pred_full,
                "model_label_noteonly": pred_note,
            }
            results.append(result_row)

        return pl.DataFrame(results)
    return (run_labeling,)


@app.cell
def _(df1, df2, run_labeling):
    scored1 = run_labeling(df1)
    scored2 = run_labeling(df2)
    return scored1, scored2


@app.cell
def _(scored1, scored2):
    scored1, scored2
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Evaluation
    """)
    return


@app.cell
def _(scored1, scored2):
    def _acc(scored, gold_col: str, pred_col: str) -> float:
        tmp = scored.filter(scored[gold_col].is_not_null() & (scored[gold_col].str.strip_chars() != ""))
        if tmp.height == 0:
            return float("nan")
        correct = (tmp[gold_col].str.to_uppercase() == tmp[pred_col].str.to_uppercase()).sum()
        return float(correct) / float(tmp.height)

    out = {
        "sample1_accuracy_step2_full_text": _acc(scored1, "human_label_fulltext", "model_label_fulltext"),
        "sample1_accuracy_step3_note_only": _acc(scored1, "human_label_noteonly", "model_label_noteonly"),
        "sample2_accuracy_step2_full_text": _acc(scored2, "human_label_fulltext", "model_label_fulltext"),
        "sample2_accuracy_step3_note_only": _acc(scored2, "human_label_noteonly", "model_label_noteonly"),
    }
    out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Confusion Matrix (Sample 1: Full Text)
    """)
    return


@app.cell
def _(scored1):
    crosstab1_full = (
        scored1
        .group_by("human_label_fulltext", "model_label_fulltext")
        .len()
        .pivot(index="human_label_fulltext", on="model_label_fulltext", values="len")
    )
    crosstab1_full
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Confusion Matrix (Sample 1: Note Only)
    """)
    return


@app.cell
def _(scored1):
    crosstab1_note = (
        scored1
        .group_by("human_label_noteonly", "model_label_noteonly")
        .len()
        .pivot(index="human_label_noteonly", on="model_label_noteonly", values="len")
    )
    crosstab1_note
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Confusion Matrix (Sample 2: Full Text)
    """)
    return


@app.cell
def _(scored2):
    crosstab2_full = (
        scored2
        .group_by("human_label_fulltext", "model_label_fulltext")
        .len()
        .pivot(index="human_label_fulltext", on="model_label_fulltext", values="len")
    )
    crosstab2_full
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Confusion Matrix (Sample 2: Note Only)
    """)
    return


@app.cell
def _(scored2):
    crosstab2_note = (
        scored2
        .group_by("human_label_noteonly", "model_label_noteonly")
        .len()
        .pivot(index="human_label_noteonly", on="model_label_noteonly", values="len")
    )
    crosstab2_note
    return


@app.cell
def _(scored1, scored2):
    scored1.write_csv("students/emmazhang/cn_sample_1_scored.csv")
    scored2.write_csv("students/emmazhang/cn_sample_2_scored.csv")
    ("students/emmazhang/cn_sample_1_scored.csv", "students/emmazhang/cn_sample_2_scored.csv")
    return


if __name__ == "__main__":
    app.run()
