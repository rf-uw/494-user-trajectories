import pandas as pd
pd.set_option("display.max_columns", None)

PATH = "dataset 2.24/ratings-20260117-20260217.parquet"

# thresholds — change here to adjust all steps
SESSION_GAP_MIN = 5
SWARM_MIN_RATINGS = 20
SWARM_WINDOW_HOURS = 1
SAMPLE_N = 300_000

# toggle each step on/off
RUN_CHECK = False   # step 1: explore the dataset
RUN_PROTO = False   # step 2: prototype on 300k sample
RUN_FULL  = True    # step 3: full run on all data


# --- step 1: explore ---
if RUN_CHECK:
    df_explore = pd.read_parquet(PATH)
    print("Shape:", df_explore.shape)
    print("\nColumns:", list(df_explore.columns))
    print("\nDtypes:\n", df_explore.dtypes)
    print("\nNull counts:\n", df_explore.isnull().sum())
    print("\nHead:")
    print(df_explore.head(3))
    print("\nfromNotification value counts:")
    print(df_explore["fromNotification"].value_counts(dropna=False))
    print("\nRating time range:", df_explore["ratingCreatedAt"].min(), "to", df_explore["ratingCreatedAt"].max())
    del df_explore


# --- step 2: prototype on 300k sample ---
if RUN_PROTO:
    use_cols = [
        "noteId",
        "ratedOnTweetId",
        "raterParticipantId",
        "ratingCreatedAt",
        "fromNotification",
    ]

    df_full = pd.read_parquet(PATH, columns=use_cols)
    print("Full data loaded:", df_full.shape)

    # compute note-level stats on FULL data first (for swarm)
    # swarm detection needs all ratings for a note, not just a sample
    note_stats = df_full.groupby("noteId")["ratingCreatedAt"].agg(
        rating_count="count",
        first_rating="min",
        last_rating="max",
    )
    note_stats["span"] = note_stats["last_rating"] - note_stats["first_rating"]
    note_stats["is_rater_swarm"] = (
        (note_stats["rating_count"] >= SWARM_MIN_RATINGS)
        & (note_stats["span"] <= pd.Timedelta(hours=SWARM_WINDOW_HOURS))
    )
    print("\nSwarm notes (full data):", note_stats["is_rater_swarm"].sum(),
          "out of", len(note_stats), "notes")

    # work on a sample first so it doesn't take forever
    df = df_full.sample(n=SAMPLE_N, random_state=42).copy()
    print("Sampled:", df.shape)
    del df_full

    # rating session
    df = df.sort_values(["raterParticipantId", "ratingCreatedAt"])
    gap_prev = df.groupby("raterParticipantId")["ratingCreatedAt"].diff()
    gap_next = df.groupby("raterParticipantId")["ratingCreatedAt"].diff(-1).abs()
    session_gap = pd.Timedelta(minutes=SESSION_GAP_MIN)
    df["is_rating_session"] = (gap_prev <= session_gap) | (gap_next <= session_gap)
    print("\n--- is_rating_session ---")
    print(df["is_rating_session"].value_counts())

    # same post interest
    cnt = df.groupby(["raterParticipantId", "ratedOnTweetId"])["noteId"].transform("size")
    df["is_same_post_interest"] = cnt > 1
    print("\n--- is_same_post_interest ---")
    print(df["is_same_post_interest"].value_counts())

    # notification
    df["is_notification"] = df["fromNotification"].fillna(False).astype(bool)
    print("\n--- is_notification ---")
    print(df["is_notification"].value_counts())

    # rater swarm (merge back)
    df = df.merge(note_stats[["is_rater_swarm"]], on="noteId", how="left")
    df["is_rater_swarm"] = df["is_rater_swarm"].fillna(False)
    print("\n--- is_rater_swarm ---")
    print(df["is_rater_swarm"].value_counts())

    # quick summary
    print("\n--- Flag summary (% True) ---")
    for col in ["is_rating_session", "is_same_post_interest", "is_notification", "is_rater_swarm"]:
        pct = df[col].mean() * 100
        print(f"  {col}: {pct:.2f}%")


# --- step 3: full run ---
if RUN_FULL:
    use_cols = [
        "noteId",
        "ratedOnTweetId",
        "raterParticipantId",
        "ratingCreatedAt",
        "fromNotification",
    ]

    df = pd.read_parquet(PATH, columns=use_cols)
    print("Loaded:", df.shape)

    # rater swarm (note-level, compute first)
    # need all ratings per note before any filtering
    note_stats = df.groupby("noteId")["ratingCreatedAt"].agg(
        rating_count="count",
        first_rating="min",
        last_rating="max",
    )
    note_stats["span"] = note_stats["last_rating"] - note_stats["first_rating"]
    note_stats["is_rater_swarm"] = (
        (note_stats["rating_count"] >= SWARM_MIN_RATINGS)
        & (note_stats["span"] <= pd.Timedelta(hours=SWARM_WINDOW_HOURS))
    )
    print("Swarm notes:", note_stats["is_rater_swarm"].sum(), "/", len(note_stats))

    # rating session
    # if a user rates notes within 5 minutes of each other,
    # treat that as part of the same session
    df = df.sort_values(["raterParticipantId", "ratingCreatedAt"])
    session_gap = pd.Timedelta(minutes=SESSION_GAP_MIN)
    gap_prev = df.groupby("raterParticipantId")["ratingCreatedAt"].diff()
    gap_next = df.groupby("raterParticipantId")["ratingCreatedAt"].diff(-1).abs()
    df["is_rating_session"] = (gap_prev <= session_gap) | (gap_next <= session_gap)
    print("\n--- is_rating_session ---")
    print(df["is_rating_session"].value_counts())

    # same post interest
    # check if a user rated multiple notes on the same tweet
    cnt = df.groupby(["raterParticipantId", "ratedOnTweetId"])["noteId"].transform("size")
    df["is_same_post_interest"] = cnt > 1
    print("\n--- is_same_post_interest ---")
    print(df["is_same_post_interest"].value_counts())

    # notification
    # straightforward from the fromNotification column
    # no ratingSourceBucketed in this dataset, so using fromNotification instead
    df["is_notification"] = df["fromNotification"].fillna(False).astype(bool)
    print("\n--- is_notification ---")
    print(df["is_notification"].value_counts())

    # rater swarm (merge back)
    df = df.merge(note_stats[["is_rater_swarm"]], on="noteId", how="left")
    df["is_rater_swarm"] = df["is_rater_swarm"].fillna(False)
    print("\n--- is_rater_swarm ---")
    print(df["is_rater_swarm"].value_counts())

    # final summary
    print("\n--- Flag summary (% True) ---")
    for col in ["is_rating_session", "is_same_post_interest", "is_notification", "is_rater_swarm"]:
        pct = df[col].mean() * 100
        print(f"  {col}: {pct:.2f}%")

    # save final output
    out_path = "students/frecesca-wang/issue47/week6_features_full.parquet"
    df.to_parquet(out_path, index=False)
    print("\nSaved:", out_path)
    print("Final shape:", df.shape)