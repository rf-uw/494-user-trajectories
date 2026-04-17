import pandas as pd
import numpy as np

# =========================
# 0) Load dataset
# =========================
INPUT_PATH = "data/ratings-20260117-20260217.parquet"
OUTPUT_PATH = "ratings_with_features.csv"

df = pd.read_parquet(INPUT_PATH)

# Make sure timestamp is datetime
df["ratingCreatedAt"] = pd.to_datetime(df["ratingCreatedAt"])

# Sort for session logic
df = df.sort_values(["raterParticipantId", "ratingCreatedAt"]).reset_index(drop=True)

print("Loaded:", df.shape, "rows/cols")

# =========================
# 1) Rating session (<= 5 minutes gap)
# =========================
df["prev_time"] = df.groupby("raterParticipantId")["ratingCreatedAt"].shift(1)
df["next_time"] = df.groupby("raterParticipantId")["ratingCreatedAt"].shift(-1)

df["diff_prev_sec"] = (df["ratingCreatedAt"] - df["prev_time"]).dt.total_seconds()
df["diff_next_sec"] = (df["next_time"] - df["ratingCreatedAt"]).dt.total_seconds()

df["rating_session"] = (
    (df["diff_prev_sec"] <= 300) |
    (df["diff_next_sec"] <= 300)
).fillna(False).astype(int)

# =========================
# 2) Post interest (multiple ratings on same tweet by same user)
#    ratedOnTweetId is the post/tweet id
# =========================
post_counts = (
    df.groupby(["raterParticipantId", "ratedOnTweetId"])
      .size()
      .reset_index(name="post_ratings_by_user")
)

df = df.merge(post_counts, on=["raterParticipantId", "ratedOnTweetId"], how="left")

df["post_interest"] = (df["post_ratings_by_user"] > 1).astype(int)

# =========================
# 3) From notification (already boolean column in your data)
# =========================
df["from_notification"] = df["fromNotification"].fillna(False).astype(int)

# =========================
# 4) Rater swarm (majority of ratings for a note occur within a single hour)
# =========================
df["hour_block"] = df["ratingCreatedAt"].dt.floor("h")

hour_counts = (
    df.groupby(["noteId", "hour_block"])
      .size()
      .reset_index(name="hour_count")
)

total_counts = (
    df.groupby("noteId")
      .size()
      .reset_index(name="total_count")
)

swarm_df = hour_counts.merge(total_counts, on="noteId", how="left")
swarm_df["prop_in_hour"] = swarm_df["hour_count"] / swarm_df["total_count"]

swarm_summary = (
    swarm_df.groupby("noteId")["prop_in_hour"]
            .max()
            .reset_index()
)

swarm_summary["rater_swarm"] = (swarm_summary["prop_in_hour"] > 0.5).astype(int)

df = df.merge(swarm_summary[["noteId", "rater_swarm"]], on="noteId", how="left")

# Safety: fill any missing swarm values as 0
df["rater_swarm"] = df["rater_swarm"].fillna(0).astype(int)

# =========================
# 5) Quick summary stats
# =========================
print("\n=== Feature Rates ===")
print("rating_session %     :", round(df["rating_session"].mean() * 100, 2))
print("post_interest %      :", round(df["post_interest"].mean() * 100, 2))
print("from_notification %  :", round(df["from_notification"].mean() * 100, 2))
print("rater_swarm %        :", round(df["rater_swarm"].mean() * 100, 2))

# =========================
# 6) Save output
# =========================
df.to_csv(OUTPUT_PATH, index=False)
print("\nSaved:", OUTPUT_PATH)
