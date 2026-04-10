import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return (pl,)


@app.cell
def _(pl):
    writing_traj = pl.read_parquet("data/Archive/sample_user_note_traj.parquet")
    rating_traj = pl.read_parquet("data/Archive/sample_user_rating_traj.parquet")
    requesting_traj = pl.read_parquet("data/Archive/sample_user_request_traj.parquet")
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
    return (users,)


@app.cell
def _(users):
    users
    return


if __name__ == "__main__":
    app.run()
