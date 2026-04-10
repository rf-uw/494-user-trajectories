import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    return (pl,)


@app.cell
def _(pl):
    writing_traj = pl.read_parquet("../Archive-2/sample_user_note_traj.parquet")
    rating_traj = pl.read_parquet("../Archive-2/sample_user_rating_traj.parquet")
    requesting_traj = pl.read_parquet("../Archive-2/sample_user_request_traj.parquet")
    return rating_traj, requesting_traj, writing_traj


@app.cell
def _(pl, rating_traj, requesting_traj, writing_traj):
    traj = (
        writing_traj
        .join(
            rating_traj, 
            left_on=["noteAuthorParticipantId", "userMonth", "calendarMonth"], 
            right_on=["raterParticipantId", "userMonth", "calendarMonth"], 
            how="full",
            coalesce=True,
            validate="1:1"
        )
        .join(
            requesting_traj, 
            left_on=["noteAuthorParticipantId", "userMonth", "calendarMonth"], 
            right_on=["requesterParticipantId", "userMonth", "calendarMonth"], 
            how="full",
            coalesce=True,
            validate="1:1"
        )
        .with_columns(
            pl.selectors.ends_with("Count").fill_null(0),
            pl.selectors.ends_with("hits").fill_null(0),
            pl.selectors.ends_with("Created").fill_null(0),
            pl.selectors.ends_with("Rated").fill_null(0),
            pl.selectors.ends_with("Made").fill_null(0),
            pl.selectors.ends_with("Targeted").fill_null(0),
        )
        .select(
            "noteAuthorParticipantId","userMonth", "calendarMonth",
            "notesCreated", "notesRated", "requestsMade"
        )
        .sort(["noteAuthorParticipantId", "userMonth"])
    )

    users = traj.group_by("noteAuthorParticipantId").agg(
        minUserMonth=pl.col("userMonth").min(),
        maxUserMonth=pl.col("userMonth").max(),
        minCalendarMonth=pl.col("calendarMonth").min(),
        maxCalendarMonth=pl.col("calendarMonth").max(),
    )

    zero_activity = pl.DataFrame(
        {"notesRated":[0], "notesCreated":[0], "requestsMade":[0]}
    )
    return (traj,)


@app.cell
def _(pl, traj):
    user_range = (
        traj
        .group_by("noteAuthorParticipantId")
        .agg(
            pl.col("calendarMonth").min().alias("min_month"),
            pl.col("calendarMonth").max().alias("max_month"),
        )
    )

    all_months = traj.select("calendarMonth").unique()

    full_grid = (
        user_range
        .join(all_months, how="cross")
        .filter(
            (pl.col("calendarMonth") >= pl.col("min_month")) &
            (pl.col("calendarMonth") <= pl.col("max_month"))
        )
        .select(["noteAuthorParticipantId", "calendarMonth"])
    )

    traj_full = (
        full_grid.join(
            traj,
            on=["noteAuthorParticipantId", "calendarMonth"],
            how="left"
        )
        .with_columns(
            pl.col("userMonth").fill_null(-1),
            pl.col("notesCreated").fill_null(0),
            pl.col("notesRated").fill_null(0),
            pl.col("requestsMade").fill_null(0),
        )
        .sort(["noteAuthorParticipantId", "calendarMonth"])
    )

    traj_full
    return (traj_full,)


@app.cell
def _(pl, traj_full):
    users_month_stat = (
        traj_full
        .with_columns(
            pl.when(pl.col("notesCreated") == 1)
            .then(pl.lit("single-note writer"))
            .when((pl.col("notesCreated") >= 2) & (pl.col("notesCreated") <= 9))
            .then(pl.lit("single-digit writer"))
            .when(pl.col("notesCreated") >= 10)
            .then(pl.lit("double-digit writer"))

            # rater
            .when((pl.col("notesCreated") == 0) & (pl.col("notesRated") == 1))
            .then(pl.lit("single-note rater"))
            .when(
                (pl.col("notesCreated") == 0)
                & (pl.col("notesRated") >= 2)
                & (pl.col("notesRated") <= 9)
            )
            .then(pl.lit("single-digit rater"))
            .when((pl.col("notesCreated") == 0) & (pl.col("notesRated") >= 10))
            .then(pl.lit("double-digit rater"))

            # requestor
            .when(
                (pl.col("notesCreated") == 0)
                & (pl.col("notesRated") == 0)
                & (pl.col("requestsMade") == 1)
            )
            .then(pl.lit("single-post requestor"))
            .when(
                (pl.col("notesCreated") == 0)
                & (pl.col("notesRated") == 0)
                & (pl.col("requestsMade") >= 2)
                & (pl.col("requestsMade") <= 9)
            )
            .then(pl.lit("single-digit requestor"))
            .when(
                (pl.col("notesCreated") == 0)
                & (pl.col("notesRated") == 0)
                & (pl.col("requestsMade") >= 10)
            )
            .then(pl.lit("double-digit requestor"))

            .otherwise(pl.lit("not active"))
            .alias("contribution_type")
        )
        .group_by(["calendarMonth", "contribution_type"])
        .len()
        .rename({"len": "num_users"})
        .with_columns(
            pl.col("num_users").sum().over("calendarMonth").alias("month_total")
        )
        .with_columns(
            (pl.col("num_users") / pl.col("month_total")).alias("prop_users")
        )
        .sort(["calendarMonth", "contribution_type"])
    )

    users_month_stat
    return (users_month_stat,)


@app.cell
def _(users_month_stat):
    users_month_stat
    return


@app.cell
def _(users_month_stat):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick


    pivot_df = (
        users_month_stat
        .to_pandas()
        .pivot(
            index="calendarMonth",
            columns="contribution_type",
            values="prop_users",
        )
        .fillna(0)
        .sort_index()
    )

    category_order = [
        "double-digit writer",
        "single-digit writer",
        "single-note writer",
        "double-digit rater",
        "single-digit rater",
        "single-note rater",
        "double-digit requestor",
        "single-digit requestor",
        "single-post requestor",
        "not active",
    ]

    cols = [c for c in category_order if c in pivot_df.columns]
    pivot_df = pivot_df[cols]

    ax = pivot_df.plot(
        kind="bar",
        stacked=True,
        figsize=(15, 6),
        width=0.9,
    )

    ax.set_title("User pool makeup over time")
    ax.set_xlabel("Calendar month")
    ax.set_ylabel("Percent of users")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax.legend(
        title="Contribution type",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
