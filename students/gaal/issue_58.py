import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    from pathlib import Path

    ARCHIVE = Path("/Users/gaaljaylaani/494-user-trajectories/local-data/Archive (1)")
    rating_traj  = pl.read_parquet(ARCHIVE / "sample_user_rating_traj.parquet")
    note_traj    = pl.read_parquet(ARCHIVE / "sample_user_note_traj.parquet")
    request_traj = pl.read_parquet(ARCHIVE / "sample_user_request_traj.parquet")

    mo.vstack([
        mo.md(f"**rating_traj**: {len(rating_traj):,} rows"),
        mo.md(f"**note_traj**: {len(note_traj):,} rows"),
        mo.md(f"**request_traj**: {len(request_traj):,} rows"),
    ])
    return alt, mo, note_traj, pl, rating_traj, request_traj


@app.cell
def _(alt, pl):
    def _sample_users(df, user_col, n=100, seed=42):
        users = df[user_col].unique().sample(n=min(n, df[user_col].n_unique()), seed=seed)
        return df.filter(pl.col(user_col).is_in(users.implode()))

    def _summarize(df, y_col, group_col="userMonth"):
        """Precompute median and IQR per group in Python."""
        return (
            df.drop_nulls(y_col)
            .group_by(group_col)
            .agg([
                pl.col(y_col).median().alias("median"),
                pl.col(y_col).quantile(0.25).alias("q25"),
                pl.col(y_col).quantile(0.75).alias("q75"),
            ])
            .sort(group_col)
            .to_pandas()
        )

    def traj_plot(df, user_col, y_col, y_title, color="steelblue", n=100, seed=42):
        """Thin lines for ~100 sampled users + red LOESS."""
        sample = _sample_users(df, user_col, n, seed).drop_nulls(y_col).to_pandas()

        lines = (
            alt.Chart(sample)
            .mark_line(opacity=0.08, color=color)
            .encode(
                x=alt.X("userMonth:Q", title="Month in User Life"),
                y=alt.Y(f"{y_col}:Q", title=y_title),
                detail=f"{user_col}:N",
            )
        )
        loess = (
            alt.Chart(sample)
            .mark_line(color="red", strokeWidth=2.5)
            .transform_loess("userMonth", y_col, bandwidth=0.4)
            .encode(x="userMonth:Q", y=f"{y_col}:Q")
        )
        return (lines + loess).properties(width=380, height=240, title=y_title)

    def dist_plot(df, y_col, y_title, color="steelblue"):
        """Median line + IQR shaded band, precomputed in Python."""
        summary = _summarize(df, y_col)

        band = (
            alt.Chart(summary)
            .mark_area(color=color, opacity=0.3)
            .encode(
                x=alt.X("userMonth:Q", title="Month in User Life"),
                y=alt.Y("q25:Q", title=y_title),
                y2="q75:Q",
            )
        )
        median = (
            alt.Chart(summary)
            .mark_line(color=color, strokeWidth=2)
            .encode(x="userMonth:Q", y="median:Q")
        )
        return (band + median).properties(width=380, height=240, title=y_title)

    def melt_traj(df, user_col, cols, labels=None):
        """Melt multiple columns to long format for multi-series plots."""
        melted = (
            df.select([user_col, "userMonth"] + cols)
            .unpivot(on=cols, index=[user_col, "userMonth"],
                     variable_name="series", value_name="value")
        )
        if labels:
            mapping = pl.DataFrame({"series": cols, "label": labels})
            melted = melted.join(mapping, on="series", how="left").drop("series").rename({"label": "series"})
        return melted

    def traj_plot_multi(df, user_col, y_col, y_title, n=100, seed=42):
        """Multi-series trajectory plot (df already melted, y_col='value')."""
        sample = _sample_users(df, user_col, n, seed).drop_nulls(y_col).to_pandas()

        lines = (
            alt.Chart(sample)
            .mark_line(opacity=0.06)
            .encode(
                x=alt.X("userMonth:Q", title="Month in User Life"),
                y=alt.Y(f"{y_col}:Q", title=y_title),
                color="series:N",
                detail=f"{user_col}:N",
            )
        )
        loess = (
            alt.Chart(sample)
            .mark_line(strokeWidth=2.5)
            .transform_loess("userMonth", y_col, groupby=["series"], bandwidth=0.4)
            .encode(x="userMonth:Q", y=f"{y_col}:Q", color="series:N")
        )
        return (lines + loess).resolve_scale(color="shared").properties(width=380, height=240, title=y_title)

    def dist_plot_multi(df, y_col, y_title):
        """Multi-series distribution plot — median per series precomputed."""
        summary = (
            df.drop_nulls(y_col)
            .group_by(["userMonth", "series"])
            .agg(pl.col(y_col).median().alias("median"))
            .sort("userMonth")
            .to_pandas()
        )
        median = (
            alt.Chart(summary)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("userMonth:Q", title="Month in User Life"),
                y=alt.Y("median:Q", title=y_title),
                color="series:N",
            )
        )
        return median.properties(width=380, height=240, title=y_title)
    return dist_plot, dist_plot_multi, melt_traj, traj_plot, traj_plot_multi


@app.cell
def _(mo):
    mo.md("""
    ## Raters
    """)
    return


@app.cell
def _(dist_plot_multi, melt_traj, mo, rating_traj, traj_plot_multi):
    _melted = melt_traj(
        rating_traj, "raterParticipantId",
        ["avgHelpfulFactor", "avgNotHelpfulFactor"],
        ["Helpful", "Not Helpful"],
    )
    mo.hstack([
        traj_plot_multi(_melted, "raterParticipantId", "value", "Avg Note Factor"),
        dist_plot_multi(_melted, "value", "Avg Note Factor (distribution)"),
    ])
    return


@app.cell
def _(dist_plot_multi, melt_traj, mo, rating_traj, traj_plot_multi):
    _melted = melt_traj(
        rating_traj, "raterParticipantId",
        ["posFactorRatedHelpful", "negFactorRatedHelpful",
         "posFactorRatedNotHelpful", "negFactorRatedNotHelpful"],
        ["+factor Helpful", "-factor Helpful", "+factor NotHelpful", "-factor NotHelpful"],
    )
    mo.hstack([
        traj_plot_multi(_melted, "raterParticipantId", "value", "Count of Notes Rated"),
        dist_plot_multi(_melted, "value", "Count of Notes Rated (distribution)"),
    ])
    return


@app.cell
def _(dist_plot_multi, melt_traj, mo, rating_traj, traj_plot_multi):
    _melted = melt_traj(
        rating_traj, "raterParticipantId",
        ["pctHelpfulRatingsCorrect", "pctNotHelpfulRatingsCorrect"],
        ["Helpful ratings correct", "Not Helpful ratings correct"],
    )
    mo.hstack([
        traj_plot_multi(_melted, "raterParticipantId", "value", "Pct Ratings Correct"),
        dist_plot_multi(_melted, "value", "Pct Ratings Correct (distribution)"),
    ])
    return


@app.cell
def _(dist_plot_multi, melt_traj, mo, rating_traj, traj_plot_multi):
    _melted = melt_traj(
        rating_traj, "raterParticipantId",
        ["avgHelpfulIntercept", "avgNotHelpfulIntercept"],
        ["Helpful", "Not Helpful"],
    )
    mo.hstack([
        traj_plot_multi(_melted, "raterParticipantId", "value", "Avg Note Intercept"),
        dist_plot_multi(_melted, "value", "Avg Note Intercept (distribution)"),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Writers
    """)
    return


@app.cell
def _(dist_plot, mo, note_traj, traj_plot):
    mo.hstack([
        traj_plot(note_traj, "noteAuthorParticipantId", "avgNoteFactor", "Avg Note Factor"),
        dist_plot(note_traj, "avgNoteFactor", "Avg Note Factor (distribution)"),
    ])
    return


@app.cell
def _(dist_plot, mo, note_traj, traj_plot):
    mo.hstack([
        traj_plot(note_traj, "noteAuthorParticipantId", "avgNoteIntercept", "Avg Note Intercept"),
        dist_plot(note_traj, "avgNoteIntercept", "Avg Note Intercept (distribution)"),
    ])
    return


@app.cell
def _(dist_plot, mo, note_traj, traj_plot):
    mo.hstack([
        traj_plot(note_traj, "noteAuthorParticipantId", "hits", "CRH Notes Written", color="darkorange"),
        dist_plot(note_traj, "hits", "CRH Notes Written (distribution)", color="darkorange"),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Requesters
    """)
    return


@app.cell
def _(dist_plot_multi, melt_traj, mo, request_traj, traj_plot_multi):
    _melted = melt_traj(
        request_traj, "requesterParticipantId",
        ["numRequestsResultingInNote", "numRequestsResultingInCrh"],
        ["Resulted in Note", "Resulted in CRH"],
    )
    mo.hstack([
        traj_plot_multi(_melted, "requesterParticipantId", "value", "Requests Over Time"),
        dist_plot_multi(_melted, "value", "Requests Over Time (distribution)"),
    ])
    return


if __name__ == "__main__":
    app.run()
