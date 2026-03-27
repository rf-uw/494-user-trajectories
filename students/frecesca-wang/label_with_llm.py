import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Setup
    """)
    return


@app.cell
def _():
    import marimo as mo
    import re 
    import os
    import time  # NOTE: used to measure total labeling runtime for Issue #6 submission
    import polars as pl
    import textwrap  # NOTE: normalize prompt indentation for cleaner model outputs
    from openai import OpenAI
    from tqdm import tqdm
    return OpenAI, mo, os, pl, re, time, tqdm, textwrap



@app.cell
def _(OpenAI, os):
    # Set your API key
    # Make sure to create the file OPENAIKEY.txt before running this
    # (You can use the OPENAIKEY.txt.template file as a template)
    with open("secrets/OPENAIKEY.txt", "r") as f:
        os.environ["OPENAI_API_KEY"] = f.read().strip()
    client = OpenAI()
    return (client,)

print("Key prefix:", os.environ["OPENAI_API_KEY"][:8])

@app.cell
def _(pl):
    # Load MITweet dataset
    df = pl.read_csv("data/mitweet_sample.csv")
    return (df,)


@app.cell
def _(df):
    # Look at data
    df
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Labeling
    """)
    return


@app.cell
def _(textwrap):
    SIMPLE_PROMPT_TEMPLATE = textwrap.dedent("""\
    You are labeling the partisan lean of a tweet.

    Choose ONE label that best fits the tweet.

    Labels:
    - LEFT: aligns with mainstream US Democratic / progressive positions
    - RIGHT: aligns with mainstream US Republican / conservative positions
    - CENTER: explicitly neutral or bipartisan framing (use sparingly)
    - MIXED: contains clear cues for both LEFT and RIGHT
    - NONE: not about US politics, policy, or political actors

    Guidelines:
    - Base your decision only on the tweet text. Do not add outside context.
    - First decide whether the tweet is political at all. If not, use NONE.
    - If the tweet is political but mostly descriptive or unclear, prefer CENTER.
    - Only use MIXED when there are strong signals from both sides.
    - Sarcasm or irony may reduce confidence.

    Return the result in the following format only:

    <output>
    label: LEFT|CENTER|RIGHT|MIXED|NONE
    confidence: 0|1|2
    tags: short keywords describing the stance, or "none"
    </output>

    Tweet:
    {tweet}
    """)
    return (SIMPLE_PROMPT_TEMPLATE,)




@app.cell
def _(SIMPLE_PROMPT_TEMPLATE, client, df, pl, re, time, tqdm):
    def _parse_output(output_text: str) -> str:
        text = (output_text or "").strip()
        # Prefer the explicit <output> block if present
        m = re.search(r"<output>\s*(.*?)\s*</output>", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()

        # NOTE (changed): The prompt now returns a structured <output> block that includes:
        #   label: LEFT|CENTER|RIGHT|MIXED|NONE
        #   confidence: 0|1|2
        #   tags: ...
        # We only want the label for evaluation, so we extract "label:" specifically.
        m2 = re.search(r"label:\s*(LEFT|CENTER|RIGHT|MIXED|NONE)", text, flags=re.IGNORECASE)
        if m2:
            return m2.group(1).strip().upper()

        # NOTE (changed): Fallback if formatting is slightly off—grab any bare label token.
        m3 = re.search(r"\b(LEFT|CENTER|RIGHT|MIXED|NONE)\b", text, flags=re.IGNORECASE)
        return (m3.group(1).strip().upper() if m3 else "")
    
    def _query_llm(row: dict) -> str:
        tweet = row["tweet"]
        prompt = SIMPLE_PROMPT_TEMPLATE.format(tweet=tweet)

        # NOTE (changed): temperature=0 improves consistency/reproducibility
        # NOTE (added): lightweight retry logic for transient API failures
        for attempt in range(3):
            try:
                resp = client.responses.create(
                    model="gpt-4.1-mini",
                    input=prompt,
                    temperature=0,
                )
                output_text = getattr(resp, "output_text", "") or ""
                return output_text

            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))





    # Process rows with a for-loop
    start_time = time.time()  # NOTE: measure end-to-end labeling runtime

    results = []
    for row in tqdm(df.iter_rows(named=True), total=df.height):
        output_text = _query_llm(row)
        prediction = _parse_output(output_text)
        # Combine original row data with classification results
        result_row = {**row, **{"llm_output": output_text, "prediction": prediction}}
        results.append(result_row)

    elapsed_s = time.time() - start_time  # NOTE: total seconds to label the dataset
    print(
        f"Labeling finished: {df.height} tweets in {elapsed_s:.1f}s "
        f"({elapsed_s/df.height:.3f}s per tweet)"
    )

    # Convert results back to a DataFrame
    simple_predictions = (
    pl.DataFrame(results)
    .with_columns(
        # NOTE (added): Explicitly mark cases where the model output could not be parsed.
        # This helps distinguish true misclassifications from formatting failures.
        pl.when(pl.col("prediction") == "")
          .then(pl.lit("PARSE_FAIL"))
          .otherwise(pl.col("prediction"))
          .alias("prediction")
    )
)

    # NOTE: return both raw results and the cleaned DataFrame
    return (results, simple_predictions, elapsed_s)





@app.cell
def _(mo):
    mo.md(r"""
    # Evaluation
    """)
    return

@app.cell
def _(pl, simple_predictions):
    # Overall accuracy (fraction correct)
    accuracy = simple_predictions.select(
        (pl.col("prediction") == pl.col("partisan_lean")).mean().alias("accuracy")
    )
    accuracy
    return

@app.cell
def _(df, elapsed_s):
    # Total runtime + per-record runtime for reporting
    per_record_s = elapsed_s / df.height if df.height else float("nan")
    {
        "num_records": df.height,
        "elapsed_seconds": round(elapsed_s, 2),
        "seconds_per_record": round(per_record_s, 4),
    }
    return


@app.cell
def _(pl, simple_predictions):
    # Look at results
    simple_predictions
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Confusion Matrix
    """)
    return


@app.cell
def _(pl, simple_predictions):
    crosstab = (
        simple_predictions
        .group_by('partisan_lean', 'prediction')
        .len()
        .pivot(index="partisan_lean", on="prediction", values="len")
    )

    # Get prediction columns (everything except the index)
    prediction_columns = [col for col in crosstab.columns if col != "partisan_lean"]

    crosstab = (
        crosstab
        .with_columns(
            pl.concat_str([pl.lit("actually_"), pl.col("partisan_lean")]).alias("partisan_lean")
        )
        .rename({
            "partisan_lean": "actual_label",
            **{col: f"predicted_{col}" for col in prediction_columns}
        })
    )

    crosstab
    return


if __name__ == "__main__":
    app.run()

