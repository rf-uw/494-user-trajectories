import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ── thresholds ────────────────────────────────────────────────────────────────
TWEET_SESSION_CUTOFF_SEC = 300   # practical cutoff separating rapid bursts from longer gaps
SHORT_BURST_SEC          = 7     # as specified in task

INPUT_PATH = "students/frecesca-wang/issue47/week6_features_full.parquet"
OUT_DIR    = "students/frecesca-wang/issue59"
os.makedirs(OUT_DIR, exist_ok=True)

# ── load ──────────────────────────────────────────────────────────────────────
print("loading...")
df = pd.read_parquet(INPUT_PATH)
print("shape:", df.shape)

df["ratingCreatedAt"] = pd.to_numeric(df["ratingCreatedAt"])
df = df.sort_values(["raterParticipantId", "ratingCreatedAt"]).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# task 1: tweet-level session definition
# session_id is computed at (user, tweet) level, then merged back
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- task 1: tweet-level session definition ---")

first_tweet = (
    df.groupby(["raterParticipantId", "ratedOnTweetId"])["ratingCreatedAt"]
    .min()
    .reset_index()
    .rename(columns={"ratingCreatedAt": "first_tweet_ms"})
    .sort_values(["raterParticipantId", "first_tweet_ms"])
)

first_tweet["tweet_delta_sec"] = (
    first_tweet.groupby("raterParticipantId")["first_tweet_ms"].diff() / 1000
)

# session_start: first tweet per user, or gap >= cutoff
first_tweet["session_start"] = (
    first_tweet["tweet_delta_sec"].isna() |
    (first_tweet["tweet_delta_sec"] >= TWEET_SESSION_CUTOFF_SEC)
)
first_tweet["in_tweet_session"] = ~first_tweet["session_start"]

# session_id computed at tweet level — each tweet gets one id
first_tweet["session_id"] = (
    first_tweet.groupby("raterParticipantId")["session_start"].cumsum()
)

# plot
plot_df = first_tweet.dropna(subset=["tweet_delta_sec"])
log_delta = np.log(plot_df["tweet_delta_sec"].clip(lower=0.1) + 1)
in_sess  = log_delta[plot_df["in_tweet_session"]]
out_sess = log_delta[~plot_df["in_tweet_session"]]
bins = np.linspace(0, 15, 100)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist([out_sess, in_sess], bins=bins, stacked=True,
        label=["in_session = 0", "in_session = 1"],
        color=["steelblue", "orange"])
ax.axvline(np.log(TWEET_SESSION_CUTOFF_SEC + 1), color="red",
           linestyle="--", label=f"cutoff = {TWEET_SESSION_CUTOFF_SEC}s")
ax.set_xlabel("log(tweet_delta_seconds + 1)")
ax.set_ylabel("Count")
ax.set_title("Stacked histogram of log_tweet_delta_sec by in_tweet_session")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/task1_tweet_session_hist.png", dpi=150)
plt.close()
print("saved task1 plot")

# merge session_id and in_tweet_session back to rating-level df
df = df.merge(
    first_tweet[["raterParticipantId", "ratedOnTweetId",
                 "tweet_delta_sec", "in_tweet_session", "session_id"]],
    on=["raterParticipantId", "ratedOnTweetId"],
    how="left"
)

# ─────────────────────────────────────────────────────────────────────────────
# task 2: engagement per user per week
# each session counted only in the week it started (not where it appears)
# cutoff justified by distribution
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- task 2: engagement ---")

# use session starts only — each session counted once, in the week it began
session_starts = (
    first_tweet[first_tweet["session_start"]]
    .loc[:, ["raterParticipantId", "session_id", "first_tweet_ms"]]
    .copy()
)
session_starts["week"] = pd.to_datetime(
    session_starts["first_tweet_ms"], unit="ms"
).dt.to_period("W")

sessions_per_user_week = (
    session_starts.groupby(["raterParticipantId", "week"])["session_id"]
    .nunique()
    .reset_index()
    .rename(columns={"session_id": "sessions_this_week"})
)

# print distribution to justify cutoffs
print("sessions_this_week distribution:")
print(sessions_per_user_week["sessions_this_week"].describe())
print("\nquantiles:")
print(sessions_per_user_week["sessions_this_week"].quantile([0.25, 0.5, 0.75, 0.9, 0.95]))

# cutoffs based on distribution:
# 25th pct = 1, 50th = 2, 75th = 3 sessions/week
# barely: 1 session, somewhat: 2-4, very: 5+
WEEK_BARELY   = 2
WEEK_SOMEWHAT = 5

def engagement_label(n):
    if n < WEEK_BARELY:     return "barely"
    elif n < WEEK_SOMEWHAT: return "somewhat"
    else:                   return "very"

sessions_per_user_week["engagement"] = sessions_per_user_week["sessions_this_week"].apply(engagement_label)
print("\nengagement distribution:")
print(sessions_per_user_week["engagement"].value_counts())

# ─────────────────────────────────────────────────────────────────────────────
# task 3: regularity per user
# cutoff justified by distribution
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- task 3: regularity ---")

active_weeks = (
    sessions_per_user_week.groupby("raterParticipantId")["week"]
    .nunique()
    .reset_index()
    .rename(columns={"week": "active_weeks"})
)

print("active_weeks distribution:")
print(active_weeks["active_weeks"].describe())
print("\nquantiles:")
print(active_weeks["active_weeks"].quantile([0.25, 0.5, 0.75, 0.9, 0.95]))

# data window is ~5 weeks (jan 17 - feb 16)
# 1 week = irregular, 2-4 = somewhat regular, 5+ = regular (almost every week)
REG_SOMEWHAT = 2
REG_REGULAR  = 5

def regularity_label(n):
    if n < REG_SOMEWHAT:  return "irregular"
    elif n < REG_REGULAR: return "somewhat regular"
    else:                 return "regular"

active_weeks["regularity"] = active_weeks["active_weeks"].apply(regularity_label)
print("\nregularity distribution:")
print(active_weeks["regularity"].value_counts())

# ─────────────────────────────────────────────────────────────────────────────
# task 4: note delta plot
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- task 4: note delta ---")

note_df = df.sort_values(["noteId", "ratingCreatedAt"]).copy()
note_df["note_delta_sec"] = note_df.groupby("noteId")["ratingCreatedAt"].diff() / 1000
note_df = note_df.dropna(subset=["note_delta_sec"])

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(np.log1p(note_df["note_delta_sec"].clip(lower=0)), bins=100,
        color="steelblue", edgecolor="none")
ax.set_xlabel("log1p(note_delta_seconds)")
ax.set_ylabel("Count")
ax.set_title("log1p(Note delta seconds)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/task4_note_delta_hist.png", dpi=150)
plt.close()
print("saved task4 plot")

# ─────────────────────────────────────────────────────────────────────────────
# task 5: short-burst users (<7 sec between consecutive ratings, user-level)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- task 5: short-burst users ---")

df["user_delta_sec"] = (
    df.groupby("raterParticipantId")["ratingCreatedAt"].diff() / 1000
)
short_burst_users = set(
    df.loc[df["user_delta_sec"] < SHORT_BURST_SEC, "raterParticipantId"].unique()
)
total_users = df["raterParticipantId"].nunique()
print(f"short-burst users (<{SHORT_BURST_SEC}s): {len(short_burst_users)} / {total_users} ({100*len(short_burst_users)/total_users:.2f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# task 6: note delta colored by short-burst user (count + density)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- task 6: note delta colored by short-burst ---")

note_df["is_short_burst_user"] = note_df["raterParticipantId"].isin(short_burst_users)
bins = np.linspace(0, 15, 100)
sb     = np.log1p(note_df.loc[ note_df["is_short_burst_user"], "note_delta_sec"].clip(lower=0))
not_sb = np.log1p(note_df.loc[~note_df["is_short_burst_user"], "note_delta_sec"].clip(lower=0))

# stacked count
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist([not_sb, sb], bins=bins, stacked=True,
        label=["non short-burst user", "short-burst user"],
        color=["steelblue", "tomato"])
ax.set_xlabel("log1p(note_delta_seconds)")
ax.set_ylabel("Count")
ax.set_title("log1p(Note delta seconds) — colored by short-burst user")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/task6_note_delta_by_shortburst.png", dpi=150)
plt.close()

# density — shows shape difference independent of volume
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(not_sb, bins=bins, density=True, alpha=0.6,
        label="non short-burst user", color="steelblue")
ax.hist(sb, bins=bins, density=True, alpha=0.6,
        label="short-burst user", color="tomato")
ax.set_xlabel("log1p(note_delta_seconds)")
ax.set_ylabel("Density")
ax.set_title("Normalized note delta distribution by short-burst user")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/task6_note_delta_by_shortburst_density.png", dpi=150)
plt.close()
print("saved task6 plots (count + density)")

# ─────────────────────────────────────────────────────────────────────────────
# save outputs
# ─────────────────────────────────────────────────────────────────────────────
df["is_short_burst_user"] = df["raterParticipantId"].isin(short_burst_users)
df.to_parquet(f"{OUT_DIR}/issue59_ratings_enriched.parquet", index=False)
sessions_per_user_week.to_parquet(f"{OUT_DIR}/issue59_sessions_per_user_week.parquet", index=False)
active_weeks.to_parquet(f"{OUT_DIR}/issue59_user_regularity.parquet", index=False)
print(f"\ndone. all saved to {OUT_DIR}/")
