import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import polars as pl
    import seaborn as sns

    return Path, pl, plt, sns


@app.cell
def _(Path):
    DATA_DIR = Path(".").absolute().parent.parent / "data"
    return (DATA_DIR,)


@app.cell
def _():
    return


@app.cell
def _(DATA_DIR, pl):
    def _enrich_with_scores(
        notes: pl.DataFrame, ratings: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        scores = pl.read_parquet(DATA_DIR / "2026-02-03-scored_notes.parquet")
        scores = scores.with_columns(noteId=pl.col("noteId").cast(pl.String))
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
        return notes, ratings


    def _enrich_with_tweet_author_ids(
        notes: pl.DataFrame,
    ) -> pl.DataFrame:
        tweet_authors = (
            pl.scan_parquet("/data/cn_archive/derivatives/20260227_raw_posts.parquet")
            .select("post_id", "author_id")
            .filter(pl.col("author_id").is_not_null())
            .unique()
            .collect()
        )
    
        notes = notes.join(
            tweet_authors.rename({"post_id": "tweetId", "author_id": "tweet_author_id"}),
            on="tweetId",
            how="left",
            coalesce=True,
            validate="m:1"
        )
        return notes


    def _enrich_with_post_lang(
        notes: pl.DataFrame,
    ) -> pl.DataFrame:
        langs = (
            pl.scan_parquet("/data/cn_archive/derivatives/20260227_raw_posts.parquet")
            .select("post_id", "lang")
            .filter(pl.col("lang").is_not_null())
            .unique()
            .collect()
        )
        notes = notes.join(
            langs.rename({"post_id": "tweetId", "lang": "tweet_lang"}),
            on="tweetId",
            how="left",
            coalesce=True,
            validate="m:1"
        )
        return notes

    renault = pl.read_csv(DATA_DIR / "renault_partisanship_labels.csv", schema_overrides={"note_id": pl.String, "tweet_author_id": pl.String, "tweet_id": pl.String})
    _raw_notes=pl.read_parquet(DATA_DIR / "2026-02-03/notes.parquet")
    _raw_ratings=pl.read_parquet(DATA_DIR / "2026-02-03/noteRatings.parquet")

    notes = _raw_notes.with_columns(
        tweetId=pl.col("tweetId").cast(pl.String),
        noteId=pl.col("noteId").cast(pl.String),
    )
    ratings = _raw_ratings.with_columns(
        noteId=pl.col("noteId").cast(pl.String),
    )
    notes, _ =_enrich_with_scores(notes=notes, ratings=ratings)
    notes = _enrich_with_tweet_author_ids(notes=notes)
    notes = _enrich_with_post_lang(notes=notes)
    return notes, renault


@app.cell
def _(notes, pl, renault):
    party_and_factor = (
        notes
        .with_columns(createdAtDt=pl.from_epoch(pl.col("createdAtMillis"), time_unit="ms"))
        .with_columns(noteCreatedMonth=pl.col("createdAtDt").dt.strftime("%Y-%m"))
        .select("noteId", "noteCreatedMonth", "noteFinalFactor", "classification", "summary")
        .with_columns(noteId = pl.col("noteId").cast(pl.String))
        .join(renault.select("note_id","party"), coalesce=True, how="full", left_on="noteId", right_on="note_id")
    )
    return (party_and_factor,)


@app.cell
def _(pl):
    _user_ideology_barbera = (
        pl.read_csv("../../data/mosleh/07-user_ideology-barbera.csv",
                   schema_overrides={"id_str": pl.String, "ideo": pl.Float64}, null_values=["NA"])
        .drop("")
        .rename({"id_str": "tweet_author_id", "ideo": "ideologyBarbera"})
    )
    _user_ideology_barbera_unhelpful = (
        pl.read_csv("../../data/mosleh/07b-user_ideology-barbera-unhelpful.csv",
                   schema_overrides={"id_str": pl.String, "ideo": pl.Float64}, null_values=["NA"])
        .drop("")
        .rename({"id_str": "tweet_author_id", "ideo": "ideologyBarberaUnhelpful"})
    )
    _barbera_mosleh_gpt = (
        pl.read_excel("../../data/mosleh/barbera_mosleh_gpt.xlsx")
        .drop("__UNNAMED__0")
    )
    _barbera_mosleh_perplexity = (
        pl.read_excel("../../data/mosleh/barbera_mosleh_perplexity.xlsx")
        .drop("__UNNAMED__0")
    )

    ideologies = (
        _user_ideology_barbera
        .join(_user_ideology_barbera_unhelpful, on="tweet_author_id", 
              how="full", validate="1:1", coalesce=True)
        .join(_barbera_mosleh_gpt, on="tweet_author_id", 
              how="full", validate="1:1", coalesce=True)
        .join(_barbera_mosleh_perplexity, on="tweet_author_id", 
              how="full", validate="1:1", coalesce=True)
        .filter(
            pl.any_horizontal(
                pl.selectors.exclude("tweet_author_id").is_not_null()
            )
        )
    )
    return (ideologies,)


@app.cell
def _(notes):
    notes
    return


@app.cell
def _(ideologies, notes, pl, renault):
    (
        notes
        .filter(pl.col("tweet_lang") == "en")
        .join(ideologies.with_columns(ideoAvailable=pl.lit(True)), 
              on="tweet_author_id", how="left", coalesce=True, validate="m:1")
        .join(renault.select("note_id","party").with_columns(renaultAvailable=pl.lit(True)),
              left_on="noteId", right_on="note_id", how="left", coalesce=True, validate="1:1")
        .with_columns(createdAtDt = pl.from_epoch(pl.col("createdAtMillis"), time_unit="ms"))
        .with_columns(createdAtMonth = pl.col("createdAtDt").dt.strftime("%Y-%m"))
        .with_columns(ideoAvailable=pl.col("ideoAvailable").fill_null(False))
        .with_columns(renaultAvailable=pl.col("renaultAvailable").fill_null(False))
        # .filter(~pl.col("ideoAvailable"))
        # .select(["tweet_author_id"])
        # .filter(pl.col("tweet_author_id").is_not_null())
        # .unique()

        .group_by("createdAtMonth")
        .agg(n=pl.len(), 
             nWithIdeo=pl.col("ideoAvailable").sum(), pctWithIdeo=pl.col("ideoAvailable").mean(),
                nWithRenault=pl.col("renaultAvailable").sum(), pctWithRenault=pl.col("renaultAvailable").mean()
            )
        .sort("createdAtMonth")


        # .group_by(pl.lit(1))
        # .agg(n=pl.len(), 
        #      nWithIdeo=pl.col("ideoAvailable").sum(), pctWithIdeo=pl.col("ideoAvailable").mean(),
        #         nWithRenault=pl.col("renaultAvailable").sum(), pctWithRenault=pl.col("renaultAvailable").mean()
        #     )
    )
    return


@app.cell
def _(notes, pl):
    (
        notes
        .select(["tweet_author_id"])
        .filter(pl.col("tweet_author_id").is_not_null())
        .unique()
    )
    return


@app.cell
def _(party_and_factor, sns):
    sns.violinplot(
        data=party_and_factor.to_pandas(),
        x="party",
        hue="classification",
        y="noteFinalFactor",
    )
    return


@app.cell
def _(party_and_factor, pl, sns):
    sns.boxplot(
        data=party_and_factor
            .sort("noteCreatedMonth")
            .filter(pl.col("party") != "unknown")
            .to_pandas(),
        x="noteCreatedMonth",
        hue="party",
        y="noteFinalFactor",
    )
    return


@app.cell
def _(party_and_factor, pl, plt, sns):
    _ax = sns.pointplot(
        data=party_and_factor
            .sort("noteCreatedMonth")
            .filter(pl.col("party") != "unknown")
            .to_pandas(),
        estimator="median",
        errorbar=("ci", 95),
        x="noteCreatedMonth",
        hue="party",
        y="noteFinalFactor",
    )


    _ax.tick_params(axis="x", rotation=45)
    # keep every third month label
    _ticks = _ax.get_xticks()
    _labels = _ax.get_xticklabels()

    # _ax.set_xticks(_ticks[::3])
    # _ax.set_xticklabels([l.get_text() for l in _labels[::3]])
    plt.setp(_ax.get_xticklabels(), ha="right")

    # move legend outside
    _ax
    return


@app.cell
def _(pl):
    writing_traj = pl.read_parquet("../../data/Archive/sample_user_note_traj.parquet")
    rating_traj = pl.read_parquet("../../data/Archive/sample_user_rating_traj.parquet")
    rating_traj = rating_traj.with_columns(
        pl.col([
            "proRepRatings",
            "antiDemRatings",
            "antiRepRatings",
            "proDemRatings"
        ]).cast(pl.Int64)
    ).with_columns(
        pl.col("avgHelpfulFactor").fill_null(0),
        pl.col("avgNotHelpfulFactor").fill_null(0),
        lean=pl.col("proRepRatings") + pl.col("antiDemRatings")
            - pl.col("antiRepRatings") - pl.col("proDemRatings"),
        politicalRatings=pl.col("proRepRatings") + 
            pl.col("antiDemRatings") + 
            pl.col("antiRepRatings") +
            pl.col("proDemRatings"),

    ).with_columns(
        avgFactorDiff = pl.col("avgHelpfulFactor") - pl.col("avgNotHelpfulFactor")
    )
    requesting_traj = pl.read_parquet("../../data/Archive/sample_user_request_traj.parquet")
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
            "notesCreated", "notesRated", "requestsMade", "lean", "politicalRatings", "avgFactorDiff", "avgNotHelpfulFactor", "avgHelpfulFactor"
        )
        .sort(["noteAuthorParticipantId", "userMonth"])
    )



    full_months = (
        traj.group_by("noteAuthorParticipantId").agg(
            minUserMonth=pl.col("userMonth").min(),
            maxUserMonth=pl.col("userMonth").max(),
            minCalendarMonth=pl.col("calendarMonth").min(),
            maxCalendarMonth=pl.col("calendarMonth").max(),
        )
        .with_columns(
            start = pl.col("minCalendarMonth").str.strptime(pl.Date, "%Y-%m"),
            end   = pl.col("maxCalendarMonth").str.strptime(pl.Date, "%Y-%m"),
        )
        .with_columns(
            calendarMonth = pl.date_ranges("start", "end", interval="1mo", closed="both")
        )
        .explode("calendarMonth")
        .with_columns(
            calendarMonth = pl.col("calendarMonth").dt.strftime("%Y-%m")
        )
        .select("noteAuthorParticipantId", "calendarMonth", "start")
        .with_columns(
            userMonth = (
                (
                    pl.col("calendarMonth").str.strptime(pl.Date, "%Y-%m").dt.year() * 12
                    + pl.col("calendarMonth").str.strptime(pl.Date, "%Y-%m").dt.month()
                )
                -
                (
                    pl.col("start").dt.year() * 12
                    + pl.col("start").dt.month()
                )
            )
        )
        .drop("start")
    )

    traj = (
        full_months
        .join(
            traj,
            on=["noteAuthorParticipantId", "calendarMonth", "userMonth"],
            how="left",
        )
        .with_columns(
            pl.selectors.numeric().fill_null(0)
        )
        .sort(["noteAuthorParticipantId", "userMonth"])
    )
    return (traj,)


@app.cell
def _(pl, plt, sns, traj):
    _traj = (
        traj
        .filter(pl.col("notesRated") > 0)
        .filter(pl.col("politicalRatings") > 0)
        .filter(pl.col("calendarMonth") < "2024-08")
        .filter(pl.col("calendarMonth") > "2022-12")
        .filter(pl.col("notesRated") > 0)
        .with_columns(
            leanBin = pl.col("lean").clip(
                lower_bound=-2, upper_bound=2
            )
        )
        .sort("calendarMonth")
    )
    _ax = sns.histplot(
        data=_traj.to_pandas(),
        x="calendarMonth",
        hue="leanBin",
        multiple="fill",   # stack to 100%
        stat="proportion",
        discrete=True,
        # hue_order=[1, 0, -1]
    )


    _ax.tick_params(axis="x", rotation=45)
    # keep every third month label
    _ticks = _ax.get_xticks()
    _labels = _ax.get_xticklabels()

    # _ax.set_xticks(_ticks[::3])
    # _ax.set_xticklabels([l.get_text() for l in _labels[::3]])
    plt.setp(_ax.get_xticklabels(), ha="right")

    # move legend outside
    sns.move_legend(_ax, "upper left", bbox_to_anchor=(1, 1))
    _ax
    return


@app.cell
def _(pl, sns, traj):
    _traj = (
        traj
        .filter(pl.col("notesRated") > 0)
        .filter(pl.col("politicalRatings") > 0)
        .filter(pl.col("calendarMonth") < "2024-08")
        .filter(pl.col("calendarMonth") > "2022-12")
        .filter(pl.col("notesRated") > 0)
        .with_columns(
            factorDiffBin = pl.col("avgFactorDiff").cut(
                breaks=[i/2 for i in range(-10, 13)]
            )
        )
        .with_columns(
            leanBin = pl.col("lean").clip(
                lower_bound=-1, upper_bound=1
            )
        )
        .sort("calendarMonth")
    )

    sns.violinplot(
        data=_traj.to_pandas(),
        x="leanBin",
        y="avgFactorDiff",
    )
    return


@app.cell
def _(pl, plt, sns, traj):
    _traj = (
        traj
        .filter(pl.col("notesRated") > 0)
        # .filter(pl.col("politicalRatings") > 0)
        .filter(pl.col("calendarMonth") > "2021-10")
        # .filter(pl.col("calendarMonth") > "2022-12")
        .with_columns(
            factorDiffBin = pl.col("avgFactorDiff").cut(
                breaks=[i/2 for i in range(-10, 13)]
            )
        )
        .sort("calendarMonth")
    )

    hue_order = [
      "(-2, -1.5]",
      "(-1.5, -1]",
      "(-1, -0.5]",
      "(-0.5, 0]",
      "(0, 0.5]",
      "(0.5, 1]",
      "(1, 1.5]"
    ]

    ax = sns.histplot(
        data=_traj.to_pandas(),
        x="calendarMonth",
        hue="factorDiffBin",
        hue_order=hue_order,
        multiple="fill",
        stat="proportion",
        discrete=True,
    )

    ax.tick_params(axis="x", rotation=45)
    # keep every third month label
    ticks = ax.get_xticks()
    labels = ax.get_xticklabels()

    ax.set_xticks(ticks[::3])
    ax.set_xticklabels([l.get_text() for l in labels[::3]])
    plt.setp(ax.get_xticklabels(), ha="right")

    # move legend outside
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax
    return (hue_order,)


@app.cell
def _(hue_order, pl, sns, traj):
    _traj = (
        traj
        .filter(pl.col("notesRated") > 0)
        # .filter(pl.col("politicalRatings") > 0)
        # .filter(pl.col("calendarMonth") < "2024-08")
        # .filter(pl.col("calendarMonth") > "2022-12")
        .with_columns(
            factorDiffBin = pl.col("avgFactorDiff").cut(
                breaks=[i/2 for i in range(-10, 13)]
            ),
            userMonth=pl.col("userMonth").clip(0, 36)
        )
        .sort("calendarMonth")
    )

    sns.histplot(
        data=_traj.to_pandas(),
        x="userMonth",
        hue="factorDiffBin",
        hue_order=hue_order,
        multiple="fill",
        stat="proportion",
        discrete=True,
    )
    return


if __name__ == "__main__":
    app.run()
