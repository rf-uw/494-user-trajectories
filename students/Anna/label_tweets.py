import marimo

__generated_with = "0.17.6"
app = marimo.App()


@app.cell
def _():
    import os
    import re
    import time
    import polars as pl
    from openai import OpenAI
    from tqdm import tqdm

    with open("secrets/OPENAIKEY.txt.template", "r") as f:
        os.environ["OPENAI_API_KEY"] = f.read().strip()

    client = OpenAI()
    return client, pl, re, time, tqdm


@app.cell
def _(pl):
    df = pl.read_csv("../../../data/cn_sample_1.csv")
    df.head()
    return (df,)


@app.cell
def _():
    PROMPT = """
     You are labeling the partisan lean of the ORIGINAL POST,
        using ONLY the text of a Community Note written about it.

        The Community Note is usually correcting misinformation.
        Your task is to infer the likely partisan lean of the ORIGINAL POST
        that required correction.

        Step 1: Determine if the topic is political.
        - If the note is about sports, music, memes, tech, quotes, etc. → NONE.

        Step 2: If political, identify WHO the original post was supporting
        or attacking (based on what is being corrected).

        Rules:

        - If the note corrects misinformation attacking Democrats/liberals,
          or corrects a pro-Republican/anti-Democratic claim → RIGHT

        - If the note corrects misinformation attacking Republicans/conservatives,
          or corrects a pro-Democratic/anti-Republican claim → LEFT

        - If the topic is political but the note is neutral, descriptive,
          policy-based, international, or not clearly partisan → CENTER

        - If the note clearly indicates the original post attacked BOTH sides → MIXED

        Important:
        - Do NOT classify based on emotional tone.
        - Focus on who the ORIGINAL POST was likely targeting.
        - Many Community Notes are neutral in tone — you must infer the direction
          from what is being corrected.

        Community Note:
        \"\"\"{tweet}\"\"\"

        Return ONLY ONE word:
        LEFT | CENTER | RIGHT | MIXED | NONE
    """
    return (PROMPT,)


@app.cell
def _(PROMPT, client, re):
    def classify(tweet: str) -> str:
        prompt = PROMPT.format(tweet=tweet)

        resp = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            max_output_tokens=64,
        )

        output = resp.output_text.strip().upper()

        match = re.search(r"(LEFT|CENTER|RIGHT|MIXED|NONE)", output)
        return match.group(1) if match else "ERROR"
    return (classify,)


@app.cell
def _(classify, df, pl, time, tqdm):
    TEXT_COL = "summary"

    predictions = []
    start = time.time()

    for text in tqdm(df[TEXT_COL]):
        predictions.append(classify(text))

    end = time.time()

    results = df.with_columns(
        pl.Series("prediction", predictions)
    )

    print(f"Time: {end-start:.2f} seconds")
    results.head()
    return (results,)


@app.cell
def _(results):
    results.write_csv("labeled_summary_output.csv")
    print("Saved to labeled_summary_output.csv")
    return


if __name__ == "__main__":
    app.run()
