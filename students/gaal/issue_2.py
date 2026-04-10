import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    return (pl,)


@app.cell
def _(pl):
    notes = pl.read_csv("/Users/gaaljaylaani/494-user-trajectories/data/filtered/2023-10/notes/notes-00000.tsv", separator="\t")
    scored = pl.read_csv("/Users/gaaljaylaani/494-user-trajectories/students/gaal/output/scored_notes.tsv", separator="\t")
    ratings = pl.read_csv("/Users/gaaljaylaani/494-user-trajectories/students/gaal/output/merged_ratings.tsv", separator="\t")
    return notes, ratings, scored


@app.cell
def _(notes, pl, scored):
    # Cast noteId to Int64 in both dataframes before joining
    notes_cast = notes.with_columns(pl.col("noteId").cast(pl.Int64))
    scored_cast = scored.with_columns(pl.col("noteId").cast(pl.Int64))
    notes_with_scores = notes_cast.join(scored_cast, on="noteId", how="inner")
    return (notes_with_scores,)


@app.cell
def _(notes_with_scores):
    notes_with_scores.select([
        "noteId", 
        "summary",
        "classification",
        "coreNoteIntercept",
        "finalRatingStatus"
    ]).head(1000)
    return


@app.cell
def _(pl, ratings):
    # Cast noteId to Int64 to match notes_with_scores type
    rating_counts = ratings.with_columns(
        pl.col("noteId").cast(pl.Int64)
    ).group_by("noteId").agg([
        (pl.col("helpfulnessLevel") == "HELPFUL").sum().alias("helpful_votes"),
        (pl.col("helpfulnessLevel") == "NOT_HELPFUL").sum().alias("not_helpful_votes"),
        (pl.col("helpfulnessLevel") == "SOMEWHAT_HELPFUL").sum().alias("somewhat_helpful_votes"),
        pl.len().alias("total_ratings")
    ])
    return


if __name__ == "__main__":
    app.run()
