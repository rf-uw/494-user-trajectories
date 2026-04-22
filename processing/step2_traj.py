from hashlib import md5
import os

import polars as pl
from loguru import logger

from processing.constants import _top_5_topics, _null_means_0

logger.add("logs/step2.log", rotation="10 MB", level="DEBUG", serialize=True)

# Reusable filter expressions for rating aggregations
_rated_helpful = pl.col("helpfulnessLevel") == "HELPFUL"
_rated_not_helpful = pl.col("helpfulnessLevel") != "HELPFUL"
_pos_factor = pl.col("noteFinalFactor") > 0
_neg_factor = pl.col("noteFinalFactor") < 0
_posted_by_dem = pl.col("tweet_author_party") == "democrat"
_posted_by_rep = pl.col("tweet_author_party") == "republican"
_note_claims_misinfo = pl.col("classification") == "MISINFORMED_OR_POTENTIALLY_MISLEADING"
_note_claims_not_misinfo = pl.col("classification") == "NOT_MISLEADING"
_ever_crh = pl.col("noteEverCrh")
_never_crh = ~pl.col("noteEverCrh")

activity_levels = [
    # NB: Order matters; first match takes precedence.
    ("4_digit_writer",      pl.col("notesWritten") >= 1000),
    ("triple_digit_writer", pl.col("notesWritten") >= 100),
    ("double_digit_writer", pl.col("notesWritten") >= 10),
    ("single_digit_writer", pl.col("notesWritten") >= 2),
    ("single_note_writer",  pl.col("notesWritten") == 1),

    ("4_digit_rater",       pl.col("notesRated") >= 1000),
    ("triple_digit_rater",  pl.col("notesRated") >= 100),
    ("double_digit_rater",  pl.col("notesRated") >= 10),
    ("single_digit_rater",  pl.col("notesRated") >= 2),
    ("single_note_rater",   pl.col("notesRated") == 1),

    ("4_digit_requestor",      pl.col("notesRequested") >= 1000),
    ("triple_digit_requestor", pl.col("notesRequested") >= 100),
    ("double_digit_requestor", pl.col("notesRequested") >= 10),
    ("single_digit_requestor", pl.col("notesRequested") >= 2),
    ("single_post_requestor",  pl.col("notesRequested") == 1),

    ("not_active", pl.lit(True)),
]

# Extract ordered labels from rules
activity_level_labels = [label for label, _ in activity_levels]

# Build the classification expression from rules
def apply_rules() -> pl.Expr:
    # Apply rules in reverse order to ensure first match takes precedence
    expr = pl.lit(None, dtype=pl.String)
    for label, condition in reversed(activity_levels):
        expr = pl.when(condition).then(pl.lit(label)).otherwise(expr)

    # Make the column an ordered categorical with the specified levels
    expr = expr.cast(pl.Enum(activity_level_labels))

    return expr


if __name__ == "__main__":
    logger.info("Loading data...")
    notes = pl.read_parquet("data/intermediate/notes_enriched.parquet")
    ratings = pl.read_parquet("data/intermediate/ratings_enriched.parquet")
    requests = pl.read_parquet("data/intermediate/requests_enriched.parquet")
    logger.info("Loaded data")

    # Aggregate all users' notes per month
    user_notes = notes.group_by(["noteAuthorParticipantId", "userMonth"]).agg(
        calendarMonth=pl.col("calendarMonth").first(),
        notesWritten=pl.len(),
        hitRate=pl.col("noteEverCrh").mean(),
        hits=pl.col("noteEverCrh").sum(),
        avgNoteFactor=pl.col("noteFinalFactor").mean(),
        avgNoteIntercept=pl.col("noteFinalIntercept").mean(),
        topicsTargeted=pl.col("topic").filter(pl.col("topic").is_not_null()).n_unique(),
        avgRatingsEarned=pl.col("numRatings").mean(),

        antiDemNotes    =(_posted_by_dem & _note_claims_misinfo).sum(),
        proDemNotes     =(_posted_by_dem & _note_claims_not_misinfo).sum(),
        antiRepNotes    =(_posted_by_rep & _note_claims_misinfo).sum(),
        proRepNotes     =(_posted_by_rep & _note_claims_not_misinfo).sum(),

        *[
            pl.col("condensed_topic")
            .filter(pl.col("condensed_topic") == topic)
            .count()
            .alias(f"{topic}Count")
            for topic in _top_5_topics + ["other"]
        ]
    ).with_columns(
        notesOnDems=pl.col("antiDemNotes") + pl.col("proDemNotes"),
        notesOnReps=pl.col("antiRepNotes") + pl.col("proRepNotes"),
        demAlignedNotes=pl.col("proDemNotes") + pl.col("antiRepNotes"),
        repAlignedNotes=pl.col("proRepNotes") + pl.col("antiDemNotes"),
    ).with_columns(
        demAlignedLessRepAlignedNotes=pl.col("demAlignedNotes") - pl.col("repAlignedNotes"),
    ).sort("noteAuthorParticipantId", "userMonth")
    logger.info(f"Aggregated user notes: {len(user_notes):,} rows")


    # Aggregate all users' ratings per month
    user_ratings = ratings.group_by(["raterParticipantId", "userMonth"]).agg(
        calendarMonth=pl.col("calendarMonth").first(),
        notesRated=pl.len(),
        avgHelpfulFactor=pl.col("noteFinalFactor").filter(_rated_helpful).mean(),
        avgNotHelpfulFactor=pl.col("noteFinalFactor").filter(_rated_not_helpful).mean(),
        avgHelpfulIntercept=pl.col("noteFinalIntercept").filter(_rated_helpful).mean(),
        avgNotHelpfulIntercept=pl.col("noteFinalIntercept").filter(_rated_not_helpful).mean(),
        correctHelpfuls=_ever_crh.filter(_rated_helpful).sum(),
        correctNotHelpfuls=_never_crh.filter(_rated_not_helpful).sum(),

        # Counts by factor sign x helpfulness
        posFactorRatedHelpful=(_pos_factor & _rated_helpful).sum(),
        posFactorRatedNotHelpful=(_pos_factor & _rated_not_helpful).sum(),
        negFactorRatedHelpful=(_neg_factor & _rated_helpful).sum(),
        negFactorRatedNotHelpful=(_neg_factor & _rated_not_helpful).sum(),

        # % correct among +/- factor notes rated helpful/not helpful
        pctCorrectPosFactorHelpful=_ever_crh.filter(_pos_factor & _rated_helpful).mean(),
        pctCorrectPosFactorNotHelpful=_never_crh.filter(_pos_factor & _rated_not_helpful).mean(),
        pctCorrectNegFactorHelpful=_ever_crh.filter(_neg_factor & _rated_helpful).mean(),
        pctCorrectNegFactorNotHelpful=_never_crh.filter(_neg_factor & _rated_not_helpful).mean(),

        # % correct among helpful/not-helpful ratings overall
        pctHelpfulRatingsCorrect=_ever_crh.filter(_rated_helpful).mean(),
        pctNotHelpfulRatingsCorrect=_never_crh.filter(_rated_not_helpful).mean(),

        uniqueDaysRated=pl.col("ratingDate").n_unique(),
        avgPostsRatedPerDay=pl.len() / pl.col("ratingDate").n_unique(),
        uniqueTopicsRated=pl.col("topic").filter(pl.col("topic").is_not_null()).n_unique(),

        # Classifications from "Hyperactive Minority Alter the Stability of Community Notes" by Nudo et al.
        antiDemRatings    = (
              (_posted_by_dem & _note_claims_misinfo     & _rated_helpful)
            | (_posted_by_dem & _note_claims_not_misinfo & _rated_not_helpful)
        ).sum(),
        antiRepRatings    = (
                (_posted_by_rep & _note_claims_misinfo     & _rated_helpful)
            | (_posted_by_rep & _note_claims_not_misinfo & _rated_not_helpful)
        ).sum(),
        proDemRatings     = (
                (_posted_by_dem & _note_claims_misinfo     & _rated_not_helpful)
            | (_posted_by_dem & _note_claims_not_misinfo & _rated_helpful)
        ).sum(),
        proRepRatings     = (
                (_posted_by_rep & _note_claims_misinfo     & _rated_not_helpful)
            | (_posted_by_rep & _note_claims_not_misinfo & _rated_helpful)
        ).sum(),
        *[
            pl.col("condensed_topic")
            .filter(pl.col("condensed_topic") == topic)
            .count()
            .alias(f"{topic}RatedCount")
            for topic in _top_5_topics + ["other"]
        ],
    ).with_columns(
        overallAccuracy=(pl.col("correctHelpfuls") + pl.col("correctNotHelpfuls")) / pl.col("notesRated"),
        helpfulNotHelpfulFactorDiff=pl.col("avgHelpfulFactor") - pl.col("avgNotHelpfulFactor"),
        helpfulNotHelpfulInterceptDiff=pl.col("avgHelpfulIntercept") - pl.col("avgNotHelpfulIntercept"),
        ratingsOnDems=pl.col("antiDemRatings") + pl.col("proDemRatings"),
        ratingsOnReps=pl.col("antiRepRatings") + pl.col("proRepRatings"),
        demAlignedRatings=pl.col("proDemRatings") + pl.col("antiRepRatings"),
        repAlignedRatings=pl.col("proRepRatings") + pl.col("antiDemRatings"),
    ).with_columns(
        demAlignedLessRepAlignedRatings=pl.col("demAlignedRatings") - pl.col("repAlignedRatings"),
    ).sort("raterParticipantId", "userMonth")
    logger.info(f"Aggregated user ratings: {len(user_ratings):,} rows")

    user_requests = requests.group_by(["requesterParticipantId", "userMonth"]).agg(
        calendarMonth=pl.col("calendarMonth").first(),
        notesRequested=pl.len(),
        numRequestsResultingInCrh   = pl.col("requestResultedInCrh") .sum(),
        numRequestsResultingInNote  = pl.col("requestResultedInNote").sum(),
        pctRequestResultedInNote    = pl.col("requestResultedInNote").mean(),
        pctRequestResultedInCrh     = pl.col("requestResultedInCrh") .mean(),
    ).sort("requesterParticipantId", "userMonth")
    logger.info(f"Aggregated user requests: {len(user_requests):,} rows")

    # TODO: Number of ratings sessions + Average number of posts rated per session

    user_months = (
        user_notes
        .rename({"noteAuthorParticipantId": "participantId"})
        .join(
            user_ratings, how="full", coalesce=True,
            left_on=["participantId", "userMonth", "calendarMonth"],
            right_on=["raterParticipantId", "userMonth", "calendarMonth"]
        )
        .join(
            user_requests, how="full", coalesce=True,
            left_on=["participantId", "userMonth", "calendarMonth"],
            right_on=["requesterParticipantId", "userMonth", "calendarMonth"]
        )
        .with_columns(
            calendarDate=pl.col("calendarMonth").str.strptime(pl.Date, "%Y-%m"),
            activeMonth=(pl.col("notesWritten").fill_null(0) + pl.col("notesRated").fill_null(0) + pl.col("notesRequested").fill_null(0)) > 0))

    # Build a df from users' first observed month to the last possible month
    _when_users_joined = user_months.group_by("participantId").agg(
        userFirstCalendarMonth=pl.col("calendarDate").min(),
        userLastActiveCalendarMonth=pl.col("calendarDate").max())

    calendar_max = user_months.select(pl.col("calendarDate").max()).row(0)[0]
    calendar_min = user_months.select(pl.col("calendarDate").min()).row(0)[0]

    _all_months = pl.DataFrame({
            "calendarDate": pl.date_range(
                start=calendar_min,
                end=calendar_max,
                interval="1mo",
                eager=True)})

    empty_user_months = (
        _when_users_joined
        .join(_all_months, how="cross")
        .filter(pl.col("calendarDate") >= pl.col("userFirstCalendarMonth"))
        .with_columns(
            yearsSinceJoining = pl.col("calendarDate").dt.year() - pl.col("userFirstCalendarMonth").dt.year(),
            monthsSinceJoining = pl.col("calendarDate").dt.month() - pl.col("userFirstCalendarMonth").dt.month(),
        )
        .with_columns(
            userMonth = pl.col("yearsSinceJoining") * 12 + pl.col("monthsSinceJoining"),
            calendarMonth = pl.col("calendarDate").dt.strftime("%Y-%m"),
        )
        .drop("userFirstCalendarMonth", "userLastActiveCalendarMonth", "yearsSinceJoining", "monthsSinceJoining")
    )

    user_months = (
        user_months
        .join(
            empty_user_months, how="full", coalesce=True,
            on=["participantId", "userMonth", "calendarMonth", "calendarDate"])
        .with_columns(
            [pl.col(c).fill_null(0) for c in _null_means_0] + [pl.col("activeMonth").fill_null(False)],)
        .sort(["participantId", "calendarMonth", "userMonth"])
        .with_columns(month_role=apply_rules()))

    os.makedirs("data/output", exist_ok=True)
    user_months.write_parquet("data/output/user_month_traj.parquet")
    logger.info("Wrote trajectory file with all users.")

    # Sample 20,000 users
    all_user_ids = user_months.select("participantId").unique().sort("participantId")
    hash = md5("".join(all_user_ids["participantId"]).encode("utf-8")).hexdigest()
    logger.info(f"Hash of all user ids: {hash}") # For reproducibility checks
    sampled_user_ids = all_user_ids.sample(20_000, seed=465309)
    sampled_user_months = user_months.join(sampled_user_ids, on="participantId", how="inner")
    sampled_user_months.write_parquet("data/output/sampled_user_month_traj.parquet")
    logger.info("Wrote sampled trajectory file. Sampled 20_000 users.")
