import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

DATA_DIR = "data - 3.17/Archive"
OUT_DIR  = "students/frecesca-wang/issue59"
os.makedirs(OUT_DIR, exist_ok=True)

# ── load ──────────────────────────────────────────────────────────────────────
print("loading...")
rating  = pd.read_parquet(f"{DATA_DIR}/sample_user_rating_traj.parquet")
note    = pd.read_parquet(f"{DATA_DIR}/sample_user_note_traj.parquet")
request = pd.read_parquet(f"{DATA_DIR}/sample_user_request_traj.parquet")

# standardize user id column name
rating  = rating .rename(columns={"raterParticipantId":       "userId"})
note    = note   .rename(columns={"noteAuthorParticipantId":  "userId"})
request = request.rename(columns={"requesterParticipantId":   "userId"})

# tag each with contribution type and activity count
rating ["type"]  = "rating"
note   ["type"]  = "writing"
request["type"]  = "requesting"

rating ["activity"] = rating ["notesRated"]
note   ["activity"] = note   ["notesCreated"]
request["activity"] = request["requestsMade"]

# combine into one long dataframe
cols = ["userId", "userMonth", "calendarMonth", "type", "activity"]
df = pd.concat([rating[cols], note[cols], request[cols]], ignore_index=True)
print("combined shape:", df.shape)

# ─────────────────────────────────────────────────────────────────────────────
# Q1: most common contribution type in first active month (userMonth == 0)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Q1: first month contribution types ---")

month0 = df[df["userMonth"] == 0]
type_counts = month0.groupby("type")["userId"].nunique()
print("unique users by type in userMonth 0:")
print(type_counts)
print("\n% of first-month users:")
print((type_counts / type_counts.sum() * 100).round(1))

# ─────────────────────────────────────────────────────────────────────────────
# Q2: stacked bar — dominant contribution type by user month
# for each user-month, find user's dominant type (most activity)
# then plot % of users in each userMonth dominated by each type
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Q2: dominant type per user per month ---")

# for each user-userMonth, find dominant type
dominant = (
    df.sort_values("activity", ascending=False)
    .groupby(["userId", "userMonth"])
    .first()
    .reset_index()
    [["userId", "userMonth", "type"]]
    .rename(columns={"type": "dominant_type"})
)

# count users per userMonth per dominant type
dom_counts = (
    dominant.groupby(["userMonth", "dominant_type"])["userId"]
    .nunique()
    .reset_index()
    .rename(columns={"userId": "n_users"})
)

# pivot and normalize to %
dom_pivot = dom_counts.pivot(index="userMonth", columns="dominant_type", values="n_users").fillna(0)
dom_pct   = dom_pivot.div(dom_pivot.sum(axis=1), axis=0) * 100

# only plot userMonth 0-24 to keep it readable
dom_pct_plot = dom_pct[dom_pct.index <= 24]

colors = {"rating": "#E87070", "writing": "#5C9E6E", "requesting": "#5B9BD5"}
fig, ax = plt.subplots(figsize=(12, 6))
bottom = np.zeros(len(dom_pct_plot))
for col in ["rating", "writing", "requesting"]:
    if col in dom_pct_plot.columns:
        ax.bar(dom_pct_plot.index, dom_pct_plot[col],
               bottom=bottom, label=col.capitalize(),
               color=colors.get(col, "gray"))
        bottom += dom_pct_plot[col].values

ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_xlabel("User Month")
ax.set_ylabel("% of Users")
ax.set_title("Dominant Contribution Types Over Users' Lifetimes")
ax.legend(title="Type")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/q2_dominant_type_by_user_month.png", dpi=150)
plt.close()
print("saved q2 plot")

# ─────────────────────────────────────────────────────────────────────────────
# Q3: individual user trajectories
# pick 3 users who have activity across multiple months and types
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Q3: individual user trajectories ---")

# find users active in 5+ months across multiple types
multi_type_users = (
    dominant.groupby("userId")
    .agg(n_months=("userMonth", "nunique"), n_types=("dominant_type", "nunique"))
    .reset_index()
    .query("n_months >= 5 and n_types >= 2")
    .sort_values("n_months", ascending=False)
)
sample_users = multi_type_users["userId"].iloc[:3].tolist()
print("sample users:", [u[:8] for u in sample_users])

fig, ax = plt.subplots(figsize=(12, 6))
type_markers = {"rating": "o", "writing": "s", "requesting": "^"}

for i, uid in enumerate(sample_users):
    user_data = df[df["userId"] == uid].sort_values("userMonth")
    for t, grp in user_data.groupby("type"):
        ax.plot(grp["userMonth"], grp["activity"],
                marker=type_markers.get(t, "o"),
                label=f"User {i+1} — {t}" if i == 0 else f"_ User {i+1} — {t}",
                alpha=0.8)

ax.set_xlabel("User Month")
ax.set_ylabel("Number of Contributions")
ax.set_title("Individual User Trajectories")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/q3_individual_trajectories.png", dpi=150)
plt.close()
print("saved q3 plot")

# ─────────────────────────────────────────────────────────────────────────────
# Q4: average user age by contribution type per calendar month
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Q4: avg user age by type per calendar month ---")

avg_age = (
    df.groupby(["calendarMonth", "type"])["userMonth"]
    .mean()
    .reset_index()
    .rename(columns={"userMonth": "avg_user_month"})
)

fig, ax = plt.subplots(figsize=(14, 6))
for t, grp in avg_age.groupby("type"):
    grp = grp.sort_values("calendarMonth")
    ax.plot(grp["calendarMonth"], grp["avg_user_month"],
            label=t.capitalize(), marker="o", markersize=3,
            color=colors.get(t, "gray"))

# show every 6th month on x axis to avoid crowding
months = sorted(avg_age["calendarMonth"].unique())
ax.set_xticks(months[::6])
ax.set_xticklabels(months[::6], rotation=45, ha="right")
ax.set_xlabel("Calendar Month")
ax.set_ylabel("Average User Month")
ax.set_title("Average User Age by Contribution Type Over Time")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/q4_avg_user_age_by_type.png", dpi=150)
plt.close()
print("saved q4 plot")

# ─────────────────────────────────────────────────────────────────────────────
# Q5: regularity classification per user, then contributions by regularity
# per calendar month
# regularity based on unique active months (same logic as issue59 weekly)
# irregular: 1 month, somewhat regular: 2-4, regular: 5+
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Q5: regularity by calendar month ---")

# count unique calendar months per user per type
user_active_months = (
    df.groupby(["userId", "type"])["calendarMonth"]
    .nunique()
    .reset_index()
    .rename(columns={"calendarMonth": "active_months"})
)

def regularity_label(n):
    if n < 2:   return "irregular"
    elif n < 5: return "somewhat regular"
    else:       return "regular"

user_active_months["regularity"] = user_active_months["active_months"].apply(regularity_label)
print("regularity distribution:")
print(user_active_months["regularity"].value_counts())

# merge regularity back to main df
df = df.merge(user_active_months[["userId", "type", "regularity"]],
              on=["userId", "type"], how="left")

# sum activity per calendar month, type, regularity
reg_monthly = (
    df.groupby(["calendarMonth", "type", "regularity"])["activity"]
    .sum()
    .reset_index()
)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
types = ["rating", "writing", "requesting"]
reg_order = ["irregular", "somewhat regular", "regular"]
reg_colors = {"irregular": "#E87070", "somewhat regular": "#F5C05A", "regular": "#5C9E6E"}

for ax, t in zip(axes, types):
    sub = reg_monthly[reg_monthly["type"] == t].sort_values("calendarMonth")
    pivot = sub.pivot(index="calendarMonth", columns="regularity", values="activity").fillna(0)
    pivot = pivot.reindex(columns=reg_order, fill_value=0)
    bottom = np.zeros(len(pivot))
    for reg in reg_order:
        if reg in pivot.columns:
            ax.bar(range(len(pivot)), pivot[reg],
                   bottom=bottom, label=reg,
                   color=reg_colors[reg])
            bottom += pivot[reg].values
    # show every 6th month
    tick_idx = list(range(0, len(pivot), 6))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([pivot.index[i] for i in tick_idx], rotation=45, ha="right", fontsize=7)
    ax.set_title(f"{t.capitalize()} by Regularity")
    ax.set_xlabel("Calendar Month")
    ax.set_ylabel("Total Activity")
    ax.legend(fontsize=7)

plt.suptitle("Contributions by Regularity per Calendar Month", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/q5_contributions_by_regularity.png", dpi=150)
plt.close()
print("saved q5 plot")

print("\ndone.")
