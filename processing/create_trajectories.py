from hashlib import md5

import polars as pl
from loguru import logger

logger.add("logs/create_trajectories.log", rotation="10 MB", level="DEBUG", serialize=True)

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

_top_5_topics = ["sports", "diaries_&_daily_life", "business_&_entrepreneurs", "science_&_technology", "news_&_social_concern"]


def _cast_ids(df: pl.DataFrame) -> pl.DataFrame:
    id_columns = [
        column
        for column in df.columns
        if (
            column.lower() == "post_id"
            or column.lower() == "tweet_author_id"
            or column.lower() == "noteid"
            or column.lower() == "note_id"
            or column.lower() == "tweetid"
            or column.lower() == "tweet_id"
            or column.lower().endswith("authorid")
            or column.lower().endswith("author_id")
            or column.lower().endswith("userid")
            or column.lower().endswith("user_id")
            or column.lower().endswith("participantid")
            or column.lower().endswith("participant_id")
        )
    ]
    if not id_columns:
        return df
    return df.with_columns([pl.col(column).cast(pl.String) for column in id_columns])



# Calculate calendar-based user month (months since first action) and calendar month
def _enrich_with_user_and_calendar_month(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        _actionDt=pl.from_epoch(pl.col("createdAtMillis"), time_unit="ms"),
        _firstActionDt=pl.from_epoch(pl.col("participantFirstActionMillis"), time_unit="ms"),
    ).with_columns(
        userMonth=(
            (pl.col("_actionDt").dt.year() - pl.col("_firstActionDt").dt.year()) * 12
            + pl.col("_actionDt").dt.month() - pl.col("_firstActionDt").dt.month()
        ).cast(pl.Int32),
        calendarMonth=pl.col("_actionDt").dt.strftime("%Y-%m"),
    ).drop("_actionDt", "_firstActionDt")


def _enrich_with_scores(
    notes: pl.DataFrame, ratings: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    scores = _cast_ids(pl.read_parquet("data/2026-02-03-scored_notes.parquet"))
    I_AND_F_COLUMNS = {
        "CoreModel (v1.1)": ("coreNoteIntercept", "coreNoteFactor1"),
        "ExpansionModel (v1.1)": ("expansionNoteIntercept", "expansionNoteFactor1"),
        "ExpansionPlusModel (v1.1)": ("expansionPlusNoteIntercept", "expansionPlusNoteFactor1"),
        "GroupModel01 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel02 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel03 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel04 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel05 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel06 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel07 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel08 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel09 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel10 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel11 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel12 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel13 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "GroupModel14 (v1.1)": ("groupNoteIntercept", "groupNoteFactor1"),
        "MultiGroupModel01 (v1.0)": ("multiGroupNoteIntercept", "multiGroupNoteFactor1"),
        "TopicModel01 (v1.0)": ("topicNoteIntercept", "topicNoteFactor1"),
        "TopicModel02 (v1.0)": ("topicNoteIntercept", "topicNoteFactor1"),
        "TopicModel03 (v1.0)": ("topicNoteIntercept", "topicNoteFactor1"),
        "ScoringDriftGuard (v1.0)": (None, None),
        "NmrDueToMinStableCrhTime (v1.0)": (None, None),
        "InsufficientExplanation (v1.0)": (None, None),
    }

    scores = (
        scores
        .with_columns(scoreCreatedAtDt=pl.from_epoch(pl.col("createdAtMillis"), time_unit="ms"))
        # Infer the pre drift guard model
        .with_columns(
            preDriftModel=pl.when(pl.col("decidedBy").str.contains("ScoringDriftGuard"))
                            .then(pl.col("metaScorerActiveRules").str.split(",").list[-2])
                            .otherwise(pl.col("decidedBy"))
        )
        # Retrieve the intercept from the inferred model
        .with_columns(
            noteFinalIntercept=pl.coalesce([
                pl.when(pl.col("preDriftModel").str.starts_with(prefix))
                  .then(pl.col(intercept_col))
                for prefix, (intercept_col, _) in I_AND_F_COLUMNS.items()
                if intercept_col is not None
            ]),
            # Retrieve the factor from the inferred model
            noteFinalFactor=pl.coalesce([
                pl.when(pl.col("preDriftModel").str.starts_with(prefix))
                  .then(pl.col(factor_col))
                for prefix, (_, factor_col) in I_AND_F_COLUMNS.items()
                if factor_col is not None
            ]),
        )
        .rename({"finalRatingStatus": "noteFinalRatingStatus"})
        .select("noteId", "noteFinalRatingStatus", "numRatings", "decidedBy", "noteFinalIntercept", "noteFinalFactor")
    )

    notes = notes.join(scores, on="noteId", how="left", coalesce=True, validate="1:1")
    ratings = ratings.join(scores, on="noteId", how="left", coalesce=True, validate="m:1")
    logger.info("Enriched notes and ratings with scores and factors")
    return notes, ratings


def _enrich_with_crh(
    notes: pl.DataFrame, ratings: pl.DataFrame, requests: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    statuses = _cast_ids(pl.read_csv("data/2026-02-27-note_status_records.csv"))  # Processed statuses, taken from a scm-prep run on 2/27.
    # Calculate whether a note ever achieved CRH status
    note_ever_crh = (
        statuses
        .with_columns(status_time=pl.col("status_time").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f%z"))
        .filter(pl.col("status_time") <= pl.datetime(2026, 2, 3, time_zone="UTC"))
        .with_columns(Crh = pl.col("status") == "CURRENTLY_RATED_HELPFUL")
        .group_by("note_id")
        .agg(noteEverCrh = pl.col("Crh").any())
    )
    post_ever_crh = (
        notes
        .select("noteId", "tweetId")
        .join(note_ever_crh, left_on="noteId", right_on="note_id", how="left", validate="1:1")
        .group_by("tweetId")
        .agg(postEverCrh=pl.col("noteEverCrh").fill_null(False).any())
    )
    logger.info(f"Calculated CRH statuses for {len(note_ever_crh):,} notes")
    logger.info(f"Calculated CRH statuses for {len(post_ever_crh):,} posts")

    notes    = notes   .join(note_ever_crh, left_on="noteId", right_on="note_id", how="left", coalesce=True, validate="1:1")
    ratings  = ratings .join(note_ever_crh, left_on="noteId", right_on="note_id", how="left", coalesce=True, validate="m:1")
    requests = requests.join(post_ever_crh, on="tweetId", how="left", validate="m:1")
    logger.info("Enriched notes and ratings with CRH statuses")
    return notes, ratings, requests


def _enrich_with_topics(
    notes: pl.DataFrame, ratings: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Join anything outside to top 5 topics into "other"
    topics = _cast_ids(pl.read_parquet("data/from-soham-notes_full.parquet"))
    topics = topics.with_columns(condensed_topic=pl.when(pl.col("topic").is_in(_top_5_topics)).then(pl.col("topic")).otherwise(pl.lit("other")))
    topics = topics.select("noteId", "topic", "condensed_topic")

    notes   = notes  .join(topics, on="noteId", how="left", validate="1:1")  # TODO: Get more recent data from soham
    ratings = ratings.join(topics, on="noteId", how="left", validate="m:1")
    logger.info("Enriched notes and ratings with topics")
    # TODO: Topics for note requests?
    return notes, ratings


def _enrich_with_first_action(
    notes: pl.DataFrame, ratings: pl.DataFrame, requests: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    first_note_written      = notes     .group_by("noteAuthorParticipantId").agg(createdAtMillis=pl.col("createdAtMillis").min())
    first_note_rated        = ratings   .group_by("raterParticipantId")     .agg(createdAtMillis=pl.col("createdAtMillis").min())
    first_note_requested    = requests  .group_by("requesterParticipantId") .agg(createdAtMillis=pl.col("createdAtMillis").min())
    first_action = (
        pl.concat([
            first_note_written  .rename({"noteAuthorParticipantId": "participantId"}),
            first_note_rated    .rename({"raterParticipantId":      "participantId"}),
            first_note_requested.rename({"requesterParticipantId":  "participantId"}),
        ])
        .group_by("participantId")
        .agg(participantFirstActionMillis=pl.col("createdAtMillis").min())
    )
    logger.info(f"First action calculated for {len(first_action):,} users")


    notes    = notes   .join(first_action.rename({"participantId": "noteAuthorParticipantId"}),on="noteAuthorParticipantId",   how="left", validate="m:1")
    ratings  = ratings .join(first_action.rename({"participantId": "raterParticipantId"}),     on="raterParticipantId",        how="left", validate="m:1")
    requests = requests.join(first_action.rename({"participantId": "requesterParticipantId"}), on="requesterParticipantId",    how="left", validate="m:1")
    logger.info("Enriched notes and ratings with user join dates")
    return notes, ratings, requests, first_action


def _enrich_with_post_lang_and_author(
    notes: pl.DataFrame,
) -> pl.DataFrame:
    _our_post_data = _cast_ids(
        pl.scan_parquet("/data/cn_archive/derivatives/20260227_raw_posts.parquet")
        .select("post_id", "author_id", "lang")
        .filter(pl.col("author_id").is_not_null() | pl.col("lang").is_not_null())
        .collect()
        # Select first non null lang/author_id
        .group_by("post_id")
        .agg(
            tweet_author_id = pl.col("author_id").filter(pl.col("author_id").is_not_null()).first(),
            tweet_lang = pl.col("lang").filter(pl.col("lang").is_not_null()).first(),
            n_non_null_author_id = pl.col("author_id").filter(pl.col("author_id").is_not_null()).n_unique(),
            n_non_null_lang = pl.col("lang").filter(pl.col("lang").is_not_null()).n_unique())
    )

    _renault_post_data = _cast_ids(
        pl.read_csv(
            "data/renault_partisanship_labels.csv", 
            schema_overrides={"tweet_author_id": pl.String, "tweet_id": pl.String})
        .group_by("tweet_id")
        .agg(
            tweet_author_id = pl.col("tweet_author_id").filter(pl.col("tweet_author_id").is_not_null()).first(),
            n_non_null_author_id = pl.col("tweet_author_id").filter(pl.col("tweet_author_id").is_not_null()).n_unique())
        .with_columns(tweet_lang=pl.lit("en")) # Renault data only contains English tweets (see M&M)
    )

    # Make sure no contradictory values within a dataset
    assert _our_post_data.filter(pl.col("n_non_null_author_id") > 1).is_empty()
    assert _our_post_data.filter(pl.col("n_non_null_lang") > 1).is_empty()
    assert _renault_post_data.filter(pl.col("n_non_null_author_id") > 1).is_empty()

    # Join renault/our data
    _post_data = (
        _our_post_data.select("post_id", "tweet_author_id", "tweet_lang")
        .join(_renault_post_data.select("tweet_id", "tweet_author_id", "tweet_lang"), 
            left_on="post_id", right_on="tweet_id",
            how="full", validate="1:1", suffix="_renault")
    )

    # Make sure no contradictory values between datasets
    assert _post_data.filter(
        pl.col("tweet_author_id").is_not_null() & pl.col("tweet_author_id_renault").is_not_null() 
        & (pl.col("tweet_author_id") != pl.col("tweet_author_id_renault"))).is_empty()
    assert _post_data.filter(
        pl.col("tweet_lang").is_not_null() & pl.col("tweet_lang_renault").is_not_null() 
        & (pl.col("tweet_lang") != pl.col("tweet_lang_renault"))).is_empty()

    # Coalesce 
    _post_data = (
        _post_data.with_columns(
            post_id = pl.coalesce(["post_id", "tweet_id"]),
            tweet_author_id = pl.coalesce(["tweet_author_id", "tweet_author_id_renault"]),
            tweet_lang = pl.coalesce(["tweet_lang", "tweet_lang_renault"]))
        .rename({"post_id": "tweetId"})
        .select("tweetId", "tweet_author_id", "tweet_lang")
    )
    
    # Join notes to post data
    notes = notes.join(_post_data, on="tweetId", how="left", coalesce=True, validate="m:1")

    logger.info("Enriched notes with post language and author id")
    return notes


def _enrich_with_renault_author_party(
    notes: pl.DataFrame,
) -> pl.DataFrame:
    _renault = _cast_ids(
        pl.read_csv(
            "data/renault_partisanship_labels.csv", 
            schema_overrides={"note_id": pl.String, "tweet_author_id": pl.String, "tweet_id": pl.String}))

    _renault_authors = _renault.select(["tweet_author_id", "party"]).unique().rename({"party": "author_party"})
    _renault_posts = _renault.select(["tweet_id", "party"]).unique().rename({"party": "post_party"})
    _renault_notes = _renault.select(["note_id", "party"]).unique().rename({"party": "note_party"})

    notes = (
        notes
        .join(
            _renault_authors, on="tweet_author_id", 
            how="left", coalesce=True, validate="m:1")
        .join(
            _renault_posts, left_on="tweetId", right_on="tweet_id", 
            how="left", coalesce=True, validate="m:1")
        .join(
            _renault_notes, left_on="noteId", right_on="note_id", 
            how="left", coalesce=True, validate="m:1")
    )

    # Make sure there aren't contradictory labels for the same note/author/post
    assert notes.filter(
        pl.col("author_party").is_not_null() & pl.col("post_party").is_not_null() 
        & (pl.col("author_party") != pl.col("post_party"))).is_empty()
    assert notes.filter(
        pl.col("author_party").is_not_null() & pl.col("note_party").is_not_null() 
        & (pl.col("author_party") != pl.col("note_party"))).is_empty()
    assert notes.filter(
        pl.col("post_party").is_not_null() & pl.col("note_party").is_not_null() 
        & (pl.col("post_party") != pl.col("note_party"))).is_empty()

    # Coalesce
    notes = notes.with_columns(
        tweet_author_party=pl.coalesce([pl.col("author_party"), pl.col("post_party"), pl.col("note_party")])
    ).drop("author_party", "post_party", "note_party")
    
    logger.info("Enriched notes with Renault author party labels")
    return notes



def _enrich_ratings_with_note_data(
    ratings: pl.DataFrame, notes: pl.DataFrame,
) -> pl.DataFrame:
    ratings = ratings.join(
        notes.select("noteId", "noteEverCrh", "noteFinalFactor", "noteFinalIntercept", "topic", "tweet_author_party", "tweet_lang"),
        on="noteId",
        how="left",
        validate="m:1"
    )
    logger.info("Enriched ratings with note-level data")
    return ratings


def _enrich_requests_with_outcomes(
    requests: pl.DataFrame, notes: pl.DataFrame,
) -> pl.DataFrame:
    request_outcomes = (
        notes
        .select("tweetId", "noteEverCrh")
        .group_by("tweetId")
        .agg(requestResultedInNote=pl.lit(True), requestResultedInCrh=pl.col("noteEverCrh").any())
    )
    requests = requests.join(request_outcomes, on="tweetId", how="left", coalesce=True, validate="m:1")
    requests = requests.with_columns(
        requestResultedInNote=pl.col("requestResultedInNote").fill_null(False),
        requestResultedInCrh =pl.col("requestResultedInCrh") .fill_null(False)
    )
    logger.info("Enriched requests with outcomes")
    return requests


if __name__ == "__main__":
    # Load data
    users = pl.read_parquet("data/2026-02-03/userEnrollment.parquet")
    notes = _cast_ids(pl.read_parquet("data/2026-02-03/notes.parquet"))
    ratings = _cast_ids(pl.read_parquet("data/2026-02-03/noteRatings.parquet"))
    requests = _cast_ids(pl.read_parquet("data/2026-01-09/noteRequests.parquet")).rename({"userId": "requesterParticipantId"}) # Using the user-level requests, not post-level requests!
    logger.info("Data loaded successfully")
    
    ratings  = ratings .with_columns(ratingDate= pl.from_epoch(pl.col("createdAtMillis"), time_unit="ms").dt.date())
    requests = requests.with_columns(requestDate=pl.from_epoch(pl.col("createdAtMillis"), time_unit="ms").dt.date())

    # Enrich
    notes, ratings = _enrich_with_scores(notes, ratings)
    notes, ratings, requests = _enrich_with_crh(notes, ratings, requests)
    notes, ratings = _enrich_with_topics(notes, ratings)
    notes, ratings, requests, first_action = _enrich_with_first_action(notes, ratings, requests)

    notes    = _enrich_with_user_and_calendar_month(notes)
    ratings  = _enrich_with_user_and_calendar_month(ratings)
    requests = _enrich_with_user_and_calendar_month(requests)
    logger.info("Calculated user months and calendar months")

    notes = _enrich_with_post_lang_and_author(notes)
    notes = _enrich_with_renault_author_party(notes)
    ratings = _enrich_ratings_with_note_data(ratings, notes)
    requests = _enrich_requests_with_outcomes(requests, notes)

    # Aggregate all users' notes per month
    user_notes = notes.group_by(["noteAuthorParticipantId", "userMonth"]).agg(
        calendarMonth=pl.col("calendarMonth").first(),
        notesCreated=pl.len(),
        hitRate=pl.col("noteEverCrh").mean(),
        hits=pl.col("noteEverCrh").sum(),
        avgNoteFactor=pl.col("noteFinalFactor").mean(),
        avgNoteIntercept=pl.col("noteFinalIntercept").mean(),
        topicsTargeted=pl.col("topic").filter(pl.col("topic").is_not_null()).n_unique(),
        avgRatingsEarned=pl.col("numRatings").mean(),

        antiDemNotes    =(_posted_by_dem & _note_claims_misinfo).sum(),
        proDemNotes     =(_posted_by_dem & _note_claims_not_misinfo).sum(),
        antiRepNotes    =(_posted_by_rep & _note_claims_misinfo).sum(),
        proRepNNotes     =(_posted_by_rep & _note_claims_not_misinfo).sum(),

        *[
            pl.col("condensed_topic")
            .filter(pl.col("condensed_topic") == topic)
            .count()
            .alias(f"{topic}Count")
            for topic in _top_5_topics + ["other"]
        ]
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
        antiDemNNRatings    =(_posted_by_dem & _note_claims_misinfo     & _rated_helpful).sum(),
        proDemNNRatings     =(_posted_by_dem & _note_claims_misinfo     & _rated_not_helpful).sum(),
        proDemNNNRatings    =(_posted_by_dem & _note_claims_not_misinfo & _rated_helpful).sum(),
        antiDemNNNRatings   =(_posted_by_dem & _note_claims_not_misinfo & _rated_not_helpful).sum(),
        antiRepNNRatings    =(_posted_by_rep & _note_claims_misinfo     & _rated_helpful).sum(),
        proRepNNRatings     =(_posted_by_rep & _note_claims_misinfo     & _rated_not_helpful).sum(),
        proRepNNNRatings    =(_posted_by_rep & _note_claims_not_misinfo & _rated_helpful).sum(),
        antiRepNNNRatings   =(_posted_by_rep & _note_claims_not_misinfo & _rated_not_helpful).sum(),
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
        proDemRatings=pl.col("proDemNNRatings") + pl.col("proDemNNNRatings"),
        antiDemRatings=pl.col("antiDemNNRatings") + pl.col("antiDemNNNRatings"),
        proRepRatings=pl.col("proRepNNRatings") + pl.col("proRepNNNRatings"),
        antiRepRatings=pl.col("antiRepNNRatings") + pl.col("antiRepNNNRatings"),
    ).sort("raterParticipantId", "userMonth")
    logger.info(f"Aggregated user ratings: {len(user_ratings):,} rows")

    user_requests = requests.group_by(["requesterParticipantId", "userMonth"]).agg(
        calendarMonth=pl.col("calendarMonth").first(),
        requestsMade=pl.len(),
        numRequestsResultingInCrh   = pl.col("requestResultedInCrh") .sum(),
        numRequestsResultingInNote  = pl.col("requestResultedInNote").sum(),
        pctRequestResultedInNote    = pl.col("requestResultedInNote").mean(),
        pctRequestResultedInCrh     = pl.col("requestResultedInCrh") .mean(),
    ).sort("requesterParticipantId", "userMonth")
    logger.info(f"Aggregated user requests: {len(user_requests):,} rows")

    # TODO: Number of ratings sessions + Average number of posts rated per session

    # Write
    user_notes.write_parquet("data/user_note_traj.parquet")
    user_ratings.write_parquet("data/user_rating_traj.parquet")
    user_requests.write_parquet("data/user_request_traj.parquet")
    logger.info("Wrote full trajectory files")

    # Sample 20,000 users
    all_user_ids = first_action.select("participantId").unique().sort("participantId")
    hash = md5("".join(all_user_ids["participantId"]).encode("utf-8")).hexdigest()
    logger.info(f"Hash of all user ids: {hash}") # For reproducibility checks
    sampled_user_ids = all_user_ids.sample(20_000, seed=465309)
    sampled_user_notes = user_notes.join(sampled_user_ids, left_on="noteAuthorParticipantId", right_on="participantId", how="inner")
    sampled_user_notes.write_parquet("data/sample_user_note_traj.parquet")
    sampled_user_ratings = user_ratings.join(sampled_user_ids, left_on="raterParticipantId", right_on="participantId", how="inner")
    sampled_user_ratings.write_parquet("data/sample_user_rating_traj.parquet")
    sampled_user_requests = user_requests.join(sampled_user_ids, left_on="requesterParticipantId", right_on="participantId", how="inner")
    sampled_user_requests.write_parquet("data/sample_user_request_traj.parquet")
    logger.info(
        "Wrote sampled trajectory files. Sampled 20_000 users. "
        f"{len(sampled_user_notes):,} user-months with notes from {len(sampled_user_notes['noteAuthorParticipantId'].unique()):,} unique note authors, and "
        f"{len(sampled_user_ratings):,} user-months with ratings from {len(sampled_user_ratings['raterParticipantId'].unique()):,} unique raters."
        f"{len(sampled_user_requests):,} user-months with requests from {len(sampled_user_requests['requesterParticipantId'].unique()):,} unique requesters."
    )
