import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return (pl,)


@app.cell
def _(pl):
    writing_traj = (
        pl.read_parquet("data/Archive/sample_user_note_traj.parquet")
        .rename({"noteAuthorParticipantId": "userId"})
        .select(["userId", "userMonth", "calendarMonth", "notesCreated"])
    )

    rating_traj = (
        pl.read_parquet("data/Archive/sample_user_rating_traj.parquet")
        .rename({"raterParticipantId": "userId"})
        .select(["userId", "userMonth", "calendarMonth", "notesRated"])
    )

    requesting_traj = (
        pl.read_parquet("data/Archive/sample_user_request_traj.parquet")
        .rename({"requesterParticipantId": "userId"})
        .select(["userId", "userMonth", "calendarMonth", "requestsMade"])
    )
    return rating_traj, requesting_traj, writing_traj


@app.cell
def _(pl, rating_traj, requesting_traj, writing_traj):
    traj = (
        writing_traj
        .join(
            rating_traj,
            on=["userId", "userMonth", "calendarMonth"],
            how="full",
            coalesce=True,
            validate="1:1"
        )
        .join(
            requesting_traj,
            on=["userId", "userMonth", "calendarMonth"],
            how="full",
            coalesce=True,
            validate="1:1"
        )
        .with_columns(
            pl.col("notesCreated").fill_null(0),
            pl.col("notesRated").fill_null(0),
            pl.col("requestsMade").fill_null(0),
        )
        .select(
            "userId", "userMonth", "calendarMonth",
            "notesCreated", "notesRated", "requestsMade"
        )
        .sort(["userId", "userMonth"])
    )

    users = traj.group_by("userId").agg(
        minUserMonth=pl.col("userMonth").min(),
        maxUserMonth=pl.col("userMonth").max(),
        minCalendarMonth=pl.col("calendarMonth").min(),
        maxCalendarMonth=pl.col("calendarMonth").max(),
    )
    return traj, users


@app.cell
def _(traj, users):
    traj, users
    return


@app.cell
def _(pl, users):
    user_month_panel = (
        users
        .with_columns(
            pl.int_ranges(
                0,
                pl.col("maxUserMonth") + 1
            ).alias("userMonth")
        )
        .explode("userMonth")
        .select(["userId", "userMonth"])
        .sort(["userId", "userMonth"])
    )
    return (user_month_panel,)


@app.cell
def _(user_month_panel):
    user_month_panel
    return


@app.cell
def _(full_panel_with_month, pl, traj):
    full_traj = (
        full_panel_with_month
        .join(
            traj.select(["userId", "userMonth", "notesCreated", "notesRated", "requestsMade"]),
            on=["userId", "userMonth"],
            how="left"
        )
        .with_columns(
            pl.col("notesCreated").fill_null(0),
            pl.col("notesRated").fill_null(0),
            pl.col("requestsMade").fill_null(0),
        )
        .sort(["userId", "userMonth"])
    )
    return (full_traj,)


@app.cell
def _(full_traj):
    full_traj
    return


@app.cell
def _(full_traj, pl):
    classified = (
        full_traj
        .with_columns(
            pl.when(pl.col("notesCreated") == 1)
            .then(pl.lit("single-note writer"))

            .when((pl.col("notesCreated") >= 2) & (pl.col("notesCreated") <= 9))
            .then(pl.lit("single-digit writer"))

            .when(pl.col("notesCreated") >= 10)
            .then(pl.lit("double-digit writer"))

            .when((pl.col("notesCreated") == 0) & (pl.col("notesRated") == 1))
            .then(pl.lit("single-note rater"))

            .when((pl.col("notesCreated") == 0) & (pl.col("notesRated") >= 2) & (pl.col("notesRated") <= 9))
            .then(pl.lit("single-digit rater"))

            .when((pl.col("notesCreated") == 0) & (pl.col("notesRated") >= 10))
            .then(pl.lit("double-digit rater"))

            .when((pl.col("notesCreated") == 0) & (pl.col("notesRated") == 0) & (pl.col("requestsMade") == 1))
            .then(pl.lit("single-post requestor"))

            .when((pl.col("notesCreated") == 0) & (pl.col("notesRated") == 0) & (pl.col("requestsMade") >= 2) & (pl.col("requestsMade") <= 9))
            .then(pl.lit("single-digit requestor"))

            .when((pl.col("notesCreated") == 0) & (pl.col("notesRated") == 0) & (pl.col("requestsMade") >= 10))
            .then(pl.lit("double-digit requestor"))

            .otherwise(pl.lit("Not active"))
            .alias("activityClass")
        )
    )
    return (classified,)


@app.cell
def _(pl, traj, user_month_panel):
    user_starts = (
        traj
        .group_by("userId")
        .agg(
            startCalendarMonth=pl.col("calendarMonth").min()
        )
    )

    full_panel_with_month = (
        user_month_panel
        .join(user_starts, on="userId", how="left")
        .with_columns(
            pl.col("startCalendarMonth")
            .str.strptime(pl.Date, "%Y-%m")
            .alias("startDate")
        )
        .with_columns(
            pl.col("startDate")
            .dt.offset_by(pl.format("{}mo", pl.col("userMonth")))
            .dt.strftime("%Y-%m")
            .alias("calendarMonth")
        )
        .select(["userId", "userMonth", "calendarMonth"])
        .sort(["userId", "userMonth"])
    )
    return (full_panel_with_month,)


@app.cell
def _(classified):
    classified
    return


@app.cell
def _():
    return


@app.cell
def _(classified):
    monthly_counts = (
        classified
        .group_by(["calendarMonth", "activityClass"])
        .len()
        .rename({"len": "n_users"})
        .sort(["calendarMonth", "activityClass"])
    )
    return (monthly_counts,)


@app.cell
def _(monthly_counts, pl):
    monthly_pct = (
        monthly_counts
        .with_columns(
            pl.col("n_users").sum().over("calendarMonth").alias("month_total")
        )
        .with_columns(
            (pl.col("n_users") / pl.col("month_total") * 100).alias("pct_users")
        )
        .sort(["calendarMonth", "activityClass"])
    )
    return (monthly_pct,)


@app.cell
def _(monthly_pct):
    monthly_wide = (
        monthly_pct
        .select(["calendarMonth", "activityClass", "pct_users"])
        .pivot(
            values="pct_users",
            index="calendarMonth",
            on="activityClass"
        )
        .fill_null(0)
        .sort("calendarMonth")
    )
    return (monthly_wide,)


@app.cell
def _(monthly_wide):
    import matplotlib.pyplot as plt
    import numpy as np

    df = monthly_wide.to_pandas()
    x = np.arange(len(df))
    classes = [c for c in df.columns if c != "calendarMonth"]

    plt.figure(figsize=(14, 7))

    bottom = np.zeros(len(df))
    for cls in classes:
        plt.bar(x, df[cls], bottom=bottom, label=cls)
        bottom += df[cls].values

    plt.xticks(x, df["calendarMonth"], rotation=45, ha="right")
    plt.ylabel("Percentage of users")
    plt.xlabel("Calendar month")
    plt.title("Community Notes user makeup by month")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
