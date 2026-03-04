import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
    from scipy import stats

    sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
    plt.rcParams['figure.dpi'] = 120

    NOTES_PATH   = 'notes-20240501-20240531.parquet'
    RATINGS_PATH = 'ratings-20240501-20240531.parquet'
    return NOTES_PATH, RATINGS_PATH, np, pd, plt, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---


    Load data
    """)
    return


@app.cell
def _(NOTES_PATH, RATINGS_PATH, pd):
    # rating events: one row per (rater, note)
    notes = pd.read_parquet(NOTES_PATH)
    print(f'Rating events: {len(notes):,}  |  unique raters: {notes.raterParticipantId.nunique():,}')

    # notes written: one row per note
    ratings = pd.read_parquet(RATINGS_PATH)
    print(f'Notes authored: {len(ratings):,}  |  unique authors: {ratings.noteAuthorParticipantId.nunique():,}')

    notes.head(3)
    return notes, ratings


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.2  Rater-level features

    Computed from the **rating events** file, grouped by `raterParticipantId`.
    """)
    return


@app.cell
def _(notes):
    # ── helper: agreement between individual rating and final community verdict ──
    # A rater agrees when:
    #   helpfulnessLevel == HELPFUL          AND  noteFinalRatingStatus == CURRENTLY_RATED_HELPFUL
    #   helpfulnessLevel == NOT_HELPFUL      AND  noteFinalRatingStatus == CURRENTLY_RATED_NOT_HELPFUL
    #   helpfulnessLevel == SOMEWHAT_HELPFUL AND  noteFinalRatingStatus == CURRENTLY_RATED_HELPFUL
    # Notes with NEEDS_MORE_RATINGS are excluded (no final verdict yet).

    has_verdict = notes['noteFinalRatingStatus'].isin(
        ['CURRENTLY_RATED_HELPFUL', 'CURRENTLY_RATED_NOT_HELPFUL']
    )

    agreed = (
        (notes['helpfulnessLevel'].isin(['HELPFUL', 'SOMEWHAT_HELPFUL']) &
         (notes['noteFinalRatingStatus'] == 'CURRENTLY_RATED_HELPFUL')) |
        (notes['helpfulnessLevel'] == 'NOT_HELPFUL' &
         (notes['noteFinalRatingStatus'] == 'CURRENTLY_RATED_NOT_HELPFUL'))
    )
    notes['agreed'] = agreed & has_verdict
    notes['has_verdict'] = has_verdict

    # ── factor of notes rated helpful / not helpful ──
    helpful_mask     = notes['helpfulnessLevel'] == 'HELPFUL'
    not_helpful_mask = notes['helpfulnessLevel'] == 'NOT_HELPFUL'

    # Rater features
    rater_features = notes.groupby('raterParticipantId').agg(
        n_ratings            = ('noteId',           'count'),
        n_unique_tweets_rated= ('ratedOnTweetId',   'nunique'),
        avg_factor_all       = ('noteFinalFactor',  'mean'),
    ).reset_index()

    # avg factor of HELPFUL-rated notes
    avg_factor_h = (
        notes[helpful_mask]
        .groupby('raterParticipantId')['noteFinalFactor']
        .mean()
        .rename('avg_factor_helpful')
    )

    # avg factor of NOT_HELPFUL-rated notes
    avg_factor_nh = (
        notes[not_helpful_mask]
        .groupby('raterParticipantId')['noteFinalFactor']
        .mean()
        .rename('avg_factor_not_helpful')
    )

    # agreement rate (among notes with a final verdict)
    agree_rate = (
        notes[notes['has_verdict']]
        .groupby('raterParticipantId')['agreed']
        .mean()
        .rename('agreement_rate')
    )

    rater_features = (
        rater_features
        .join(avg_factor_h,  on='raterParticipantId')
        .join(avg_factor_nh, on='raterParticipantId')
        .join(agree_rate,    on='raterParticipantId')
    )

    # partisan bias: positive → prefers right-leaning notes as helpful
    rater_features['factor_diff'] = (
        rater_features['avg_factor_helpful'] - rater_features['avg_factor_not_helpful']
    )

    print(rater_features.shape)
    rater_features.describe()
    return (rater_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.3  Author-level features

    Computed from the **notes authored** file, grouped by `noteAuthorParticipantId`.
    """)
    return


@app.cell
def _(ratings):
    author_features = ratings.groupby('noteAuthorParticipantId').agg(
        n_notes_written       = ('noteId',                'count'),
        n_unique_tweets_written = ('tweetId',             'nunique'),
        avg_note_intercept    = ('noteFinalIntercept',    'mean'),
        avg_note_factor_written = ('noteFinalFactor',     'mean'),
        pct_notes_helpful     = ('noteFinalRatingStatus',
                                  lambda s: (s == 'CURRENTLY_RATED_HELPFUL').mean()),
    ).reset_index().rename(columns={'noteAuthorParticipantId': 'userId'})

    print(author_features.shape)
    author_features.describe()
    return (author_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1.4  Merge into one user-level feature table
    """)
    return


@app.cell
def _(author_features, pd, rater_features):
    rater_renamed = rater_features.rename(columns={'raterParticipantId': 'userId'})

    user_features = pd.merge(
        rater_renamed, author_features,
        on='userId', how='outer'
    )

    user_features['n_ratings']        = user_features['n_ratings'].fillna(0)
    user_features['n_notes_written']  = user_features['n_notes_written'].fillna(0)
    user_features['total_contributions'] = user_features['n_ratings'] + user_features['n_notes_written']

    print(f'Total users: {len(user_features):,}')
    print(f'  — raters only: {(user_features.n_notes_written == 0).sum():,}')
    print(f'  — authors only: {(user_features.n_ratings == 0).sum():,}')
    print(f'  — both: {((user_features.n_ratings > 0) & (user_features.n_notes_written > 0)).sum():,}')
    user_features.head(5)
    return (user_features,)


@app.cell
def _(user_features):
    # Save features for future use
    user_features.to_parquet('user_features.parquet', index=False)
    print('Saved user_features.parquet')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Q1 — Does partisanship measured by note factors align with external party data?

    We can assess this by checking whether a user's factor-based partisanship (avg factor of helpful vs. not helpful ratings) is self-consistent and whether users who lean right/left via factors show coherent behavior.

    What we expect: iff factors capture partisanship, users who rate right-leaning notes (positive factor) as helpful should have a positive`factor_diff.` Users who prefer left-leaning notes should have a negative `factor_diff`.
    """)
    return


@app.cell
def _(plt, user_features):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    _ax = _axes[0]
    # Left: distribution of avg factor for HELPFUL ratings
    clean = user_features.dropna(subset=['avg_factor_helpful', 'avg_factor_not_helpful'])
    _ax.hist(clean['avg_factor_helpful'], bins=80, color='steelblue', alpha=0.75, label='Helpful')
    _ax.hist(clean['avg_factor_not_helpful'], bins=80, color='tomato', alpha=0.75, label='Not Helpful')
    _ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    _ax.set_xlabel('Mean Note Factor')
    _ax.set_ylabel('Number of Users')
    _ax.set_title('Distribution of avg note factor\nby rating given')
    _ax.legend()
    _ax = _axes[1]
    _ax.scatter(clean['avg_factor_not_helpful'], clean['avg_factor_helpful'], alpha=0.05, s=4, color='slategray')
    # Right: scatter avg factor helpful vs not helpful per user
    lims = [-1.2, 1.2]
    _ax.plot(lims, lims, 'k--', linewidth=0.8, label='y = x (no bias)')
    _ax.set_xlim(lims)
    _ax.set_ylim(lims)
    _ax.set_xlabel('Avg factor — NOT HELPFUL ratings')
    _ax.set_ylabel('Avg factor — HELPFUL ratings')
    _ax.set_title('Per-user partisan lean:\nHelpful vs Not-Helpful note factors')
    _ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('q1_factor_alignment.png', bbox_inches='tight')
    plt.show()
    print(f'Correlation between avg_factor_helpful and avg_factor_not_helpful: {clean[['avg_factor_helpful', 'avg_factor_not_helpful']].corr().iloc[0, 1]:.3f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Interpretation:** If the scatter is above the y = x line, users tend to rate higher-factor right-leaning notes as helpful and lower-factor notes as not helpful.  This is consistent with the external party data showing that contributors who cover Republican tweets tend to label right-aligned notes as helpful.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Q2 — What is the relationship between contribution frequency and partisanship?
    """)
    return


@app.cell
def _(np, pd, plt, stats, user_features):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_df = user_features.dropna(subset=['factor_diff', 'total_contributions']).copy()
    plot_df = plot_df[plot_df['total_contributions'] >= 5]
    plot_df['log_contributions'] = np.log10(plot_df['total_contributions']) 
    _ax = _axes[0]
    _ax.scatter(plot_df['log_contributions'], plot_df['factor_diff'], alpha=0.05, s=4, color='steelblue')

    from statsmodels.nonparametric.smoothers_lowess import lowess
    xvals = plot_df['log_contributions'].values
    yvals = plot_df['factor_diff'].values
    # lowess trend
    trend = lowess(yvals, xvals, frac=0.3, return_sorted=True)
    _ax.plot(trend[:, 0], trend[:, 1], color='red', linewidth=2, label='LOWESS trend')
    _ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    _ax.set_xlabel('log₁₀(Total Contributions)')
    _ax.set_ylabel('Factor Diff (helpful − not helpful)')
    _ax.set_title('Frequency vs Partisan Lean\n(factor_diff > 0 → prefers right-leaning notes)')
    _ax.legend()
    _ax = _axes[1]
    plot_df['contrib_decile'] = pd.qcut(plot_df['total_contributions'], 10, labels=False)
    decile_stats = plot_df.groupby('contrib_decile')['factor_diff'].agg(['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]).reset_index()
    decile_stats.columns = ['decile', 'median', 'q25', 'q75']

    _ax.bar(decile_stats['decile'], decile_stats['median'], color=['steelblue' if v >= 0 else 'tomato' for v in decile_stats['median']])
    _ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    _ax.set_xlabel('Contribution Decile (1=least active → 10=most active)')
    _ax.set_ylabel('Median Factor Diff')
    _ax.set_title('Partisan Lean by Contribution Decile')
    plt.tight_layout()
    plt.savefig('q2_frequency_vs_partisanship.png', bbox_inches='tight')
    plt.show()
    r, p = stats.spearmanr(plot_df['log_contributions'], plot_df['factor_diff'].abs())
    print(f'Spearman r (|factor_diff| vs log contributions): {r:.3f}  (p={p:.3e})')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Q3 — Do users systematically rate one side helpful and the other not helpful?
    """)
    return


@app.cell
def _(np, plt, user_features):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    bias_df = user_features.dropna(subset=['factor_diff']).copy()
    _ax = _axes[0]
    _ax.hist(bias_df['factor_diff'], bins=100, color='mediumpurple', edgecolor='none')
    # Left: distribution of factor_diff
    _ax.axvline(0, color='black', linestyle='--', linewidth=1)
    _ax.axvline(bias_df['factor_diff'].mean(), color='red', linewidth=1.5, label=f'Mean = {bias_df['factor_diff'].mean():.3f}')
    _ax.set_xlabel('Factor Diff (avg factor helpful − avg factor not helpful)')
    _ax.set_ylabel('Number of Users')
    _ax.set_title('Distribution of Partisan Rating Bias')
    _ax.legend()
    _ax = _axes[1]
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.5]
    left_biased = [(bias_df['factor_diff'] < -t).mean() * 100 for t in thresholds]
    right_biased = [(bias_df['factor_diff'] > t).mean() * 100 for t in thresholds]
    # Right: stacked bar — how many users have strong bias in each direction
    neutral = [100 - l - r for l, r in zip(left_biased, right_biased)]
    x = np.arange(len(thresholds))
    b1 = _ax.bar(x, left_biased, label='Left-biased', color='steelblue')
    b2 = _ax.bar(x, neutral, bottom=left_biased, label='Neutral', color='lightgray')
    b3 = _ax.bar(x, right_biased, bottom=[l + n for l, n in zip(left_biased, neutral)], label='Right-biased', color='tomato')
    _ax.set_xticks(x)
    _ax.set_xticklabels([f'|diff| > {t}' for t in thresholds])
    _ax.set_ylabel('% of Users')
    _ax.set_title('Share of Users with Partisan Bias\nat Different Thresholds')
    _ax.legend()
    plt.tight_layout()
    plt.savefig('q3_systematic_bias.png', bbox_inches='tight')
    plt.show()
    pct_biased = (bias_df['factor_diff'].abs() > 0.1).mean() * 100
    print(f'{pct_biased:.1f}% of users have |factor_diff| > 0.1')
    print(f'Mean factor_diff: {bias_df['factor_diff'].mean():.4f}')
    print(f'Left-biased (diff < -0.1): {(bias_df['factor_diff'] < -0.1).mean() * 100:.1f}%')
    print(f'Right-biased (diff > 0.1): {(bias_df['factor_diff'] > 0.1).mean() * 100:.1f}%')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Q4 — Do users who rate one side helpful tend to rate more notes from that side?
    """)
    return


@app.cell
def _(notes, np, pd, plt, rater_features):
    # Classify each rating event as targeting a "right-leaning" or "left-leaning" note
    # by sign of noteFinalFactor
    notes['note_lean'] = np.where(notes['noteFinalFactor'].isna(), 'unknown', np.where(notes['noteFinalFactor'] > 0, 'right', 'left'))
    lean_counts = notes[notes['note_lean'].isin(['left', 'right'])].groupby(['raterParticipantId', 'note_lean', 'helpfulnessLevel']).size().reset_index(name='count')
    pivot = lean_counts.pivot_table(index='raterParticipantId', columns=['note_lean', 'helpfulnessLevel'], values='count', fill_value=0)
    pivot.columns = ['_'.join(c) for c in pivot.columns]
    pivot = pivot.reset_index()
    # Per rater: count ratings by note lean and helpfulness level
    for col in ['left_HELPFUL', 'left_NOT_HELPFUL', 'right_HELPFUL', 'right_NOT_HELPFUL']:
        if col not in pivot.columns:
            pivot[col] = 0
    left_cols = [c for c in pivot.columns if c.startswith('left_')]
    right_cols = [c for c in pivot.columns if c.startswith('right_')]
    pivot['total_left'] = pivot[left_cols].sum(axis=1)
    pivot['total_right'] = pivot[right_cols].sum(axis=1)
    pivot['pct_helpful_left'] = pivot['left_HELPFUL'] / (pivot['total_left'] + 1e-09)
    pivot['pct_helpful_right'] = pivot['right_HELPFUL'] / (pivot['total_right'] + 1e-09)
    pivot2 = pivot.merge(rater_features[['raterParticipantId', 'factor_diff']], on='raterParticipantId', how='inner')
    pivot2 = pivot2.dropna(subset=['factor_diff'])
    pivot2['lean_group'] = pd.cut(pivot2['factor_diff'], bins=[-np.inf, -0.2, -0.05, 0.05, 0.2, np.inf], labels=['Strong Left', 'Lean Left', 'Neutral', 'Lean Right', 'Strong Right'])
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
    _ax = _axes[0]
    group_stats = pivot2.groupby('lean_group', observed=True)[['pct_helpful_left', 'pct_helpful_right']].mean()
    group_stats.plot(kind='bar', ax=_ax, color=['steelblue', 'tomato'], alpha=0.85)
    # Ensure key columns exist
    _ax.set_xlabel('User Partisan Lean (by factor_diff)')
    _ax.set_ylabel('Mean % Helpful Ratings')
    _ax.set_title('How Often Users Rate Notes Helpful\nby Note Political Lean')
    _ax.legend(['Left-leaning notes', 'Right-leaning notes'])
    # Total ratings per lean side (all helpfulness levels)
    _ax.set_xticklabels(_ax.get_xticklabels(), rotation=30)
    _ax = _axes[1]
    vol_stats = pivot2.groupby('lean_group', observed=True)[['total_left', 'total_right']].median()
    vol_stats.plot(kind='bar', ax=_ax, color=['steelblue', 'tomato'], alpha=0.85)
    _ax.set_xlabel('User Partisan Lean (by factor_diff)')
    _ax.set_ylabel('Median Number of Ratings')
    _ax.set_title('Volume of Ratings per Note Lean\nby User Partisan Group')
    # Merge with factor_diff to classify users
    _ax.legend(['Left-leaning notes', 'Right-leaning notes'])
    _ax.set_xticklabels(_ax.get_xticklabels(), rotation=30)
    plt.tight_layout()
    plt.savefig('q4_rating_volume_by_lean.png', bbox_inches='tight')
    # Bin users by partisan lean
    # Left: mean % helpful rating by note lean and user lean group
    # Right: median volume of ratings by note lean and user lean group
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Q5 — Do users focus on a single topic, or do they mix topics?
    """)
    return


@app.cell
def _(np, plt, user_features):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    _ax = _axes[0]
    # Unique tweets rated
    rated_tw = user_features['n_unique_tweets_rated'].dropna()
    rated_tw = rated_tw[rated_tw > 0]
    _ax.hist(np.log10(rated_tw + 1), bins=60, color='steelblue', edgecolor='none')
    _ax.set_xlabel('log₁₀(Unique Tweets Rated + 1)')
    _ax.set_ylabel('Number of Users')
    _ax.set_title('Breadth of Tweets Rated\n(proxy for topical diversity)')
    _ax.axvline(np.log10(rated_tw.median() + 1), color='red', linestyle='--', label=f'Median = {int(rated_tw.median())}')
    _ax.legend()
    _ax = _axes[1]
    written_tw = user_features['n_unique_tweets_written'].dropna()
    written_tw = written_tw[written_tw > 0]
    # Unique tweets written about
    _ax.hist(np.log10(written_tw + 1), bins=60, color='seagreen', edgecolor='none')
    _ax.set_xlabel('log₁₀(Unique Tweets Written About + 1)')
    _ax.set_ylabel('Number of Users')
    _ax.set_title('Breadth of Tweets Annotated\n(proxy for topical diversity in note writing)')
    _ax.axvline(np.log10(written_tw.median() + 1), color='red', linestyle='--', label=f'Median = {int(written_tw.median())}')
    _ax.legend()
    plt.tight_layout()
    plt.savefig('q5_topic_breadth.png', bbox_inches='tight')
    plt.show()
    print(f'Rated tweets  — median: {rated_tw.median():.0f}, 75th pct: {rated_tw.quantile(0.75):.0f}, 95th pct: {rated_tw.quantile(0.95):.0f}')
    print(f'Written tweets— median: {written_tw.median():.0f}, 75th pct: {written_tw.quantile(0.75):.0f}, 95th pct: {written_tw.quantile(0.95):.0f}')
    print(f'% of raters who rated only 1 tweet: {(rated_tw == 1).mean() * 100:.1f}%')
    print(f'% of authors who wrote about only 1 tweet: {(written_tw == 1).mean() * 100:.1f}%')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Q6 — What is the relationship between contribution frequency and topic focus?
    """)
    return


@app.cell
def _(np, pd, plt, user_features):
    # Diversity ratio: unique tweets / total ratings (how often user revisits same tweet)
    uf = user_features.copy()
    uf['tweet_diversity_rating'] = uf['n_unique_tweets_rated'] / (uf['n_ratings'] + 1e-09)
    uf['tweet_diversity_writing'] = uf['n_unique_tweets_written'] / (uf['n_notes_written'] + 1e-09)
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    for _ax, xcol, ycol, color, title in [(_axes[0], 'n_ratings', 'tweet_diversity_rating', 'steelblue', 'Raters: Frequency vs Topic Diversity'), (_axes[1], 'n_notes_written', 'tweet_diversity_writing', 'seagreen', 'Authors: Frequency vs Topic Diversity')]:
        sub = uf.dropna(subset=[xcol, ycol])
        sub = sub[sub[xcol] >= 5]
        _ax.scatter(np.log10(sub[xcol] + 1), sub[ycol], alpha=0.05, s=4, color=color)
        sub['decile'] = pd.qcut(sub[xcol], 10, labels=False, duplicates='drop')
        dmed = sub.groupby('decile')[ycol].median().reset_index()
        _ax.plot(sub.groupby('decile')[xcol].median().apply(lambda x: np.log10(x + 1)).values, dmed[ycol].values, 'r-o', linewidth=2, markersize=5, label='Decile median')
        _ax.set_xlabel('log₁₀(Contributions)')
        _ax.set_ylabel('Unique Tweets / Total Ratings')
        _ax.set_title(title)  # Decile medians
        _ax.legend()
    plt.tight_layout()
    plt.savefig('q6_frequency_vs_diversity.png', bbox_inches='tight')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Q7 — What is the relationship between contribution frequency and skill?
    """)
    return


@app.cell
def _(np, pd, plt, stats, user_features):
    _fig, _axes = plt.subplots(1, 3, figsize=(16, 4))
    skill_df = user_features.dropna(subset=['avg_note_intercept', 'n_notes_written'])
    skill_df = skill_df[skill_df['n_notes_written'] >= 2]
    _ax = _axes[0]
    _ax.scatter(np.log10(skill_df['n_notes_written'] + 1), skill_df['avg_note_intercept'], alpha=0.05, s=5, color='steelblue')
    # (a) Avg note intercept vs number of notes written
    skill_df['dec'] = pd.qcut(skill_df['n_notes_written'], 10, labels=False, duplicates='drop')
    dec_med = skill_df.groupby('dec').agg(x=('n_notes_written', 'median'), y=('avg_note_intercept', 'median'))
    _ax.plot(np.log10(dec_med['x'] + 1), dec_med['y'], 'r-o', linewidth=2, markersize=5)
    _ax.set_xlabel('log₁₀(Notes Written)')
    _ax.set_ylabel('Avg Note Intercept')
    _ax.set_title('Note Quality vs Writing Frequency')
    _ax = _axes[1]
    skill_df2 = user_features.dropna(subset=['pct_notes_helpful', 'n_notes_written'])
    skill_df2 = skill_df2[skill_df2['n_notes_written'] >= 5]
    _ax.scatter(np.log10(skill_df2['n_notes_written'] + 1), skill_df2['pct_notes_helpful'] * 100, alpha=0.05, s=5, color='seagreen')
    skill_df2['dec'] = pd.qcut(skill_df2['n_notes_written'], 10, labels=False, duplicates='drop')
    dec_med2 = skill_df2.groupby('dec').agg(x=('n_notes_written', 'median'), y=('pct_notes_helpful', 'median'))
    _ax.plot(np.log10(dec_med2['x'] + 1), dec_med2['y'] * 100, 'r-o', linewidth=2, markersize=5)
    _ax.set_xlabel('log₁₀(Notes Written)')
    # (b) % notes earning helpful vs number of notes written
    _ax.set_ylabel('% Notes Earning Helpful')
    _ax.set_title('% Helpful Notes vs Writing Frequency')
    _ax = _axes[2]
    skill_df3 = user_features.dropna(subset=['agreement_rate', 'n_ratings'])
    skill_df3 = skill_df3[skill_df3['n_ratings'] >= 10]
    _ax.scatter(np.log10(skill_df3['n_ratings'] + 1), skill_df3['agreement_rate'] * 100, alpha=0.05, s=5, color='mediumpurple')
    skill_df3['dec'] = pd.qcut(skill_df3['n_ratings'], 10, labels=False, duplicates='drop')
    dec_med3 = skill_df3.groupby('dec').agg(x=('n_ratings', 'median'), y=('agreement_rate', 'median'))
    _ax.plot(np.log10(dec_med3['x'] + 1), dec_med3['y'] * 100, 'r-o', linewidth=2, markersize=5)
    _ax.set_xlabel('log₁₀(Ratings Given)')
    _ax.set_ylabel('% Ratings Agreeing with Final Verdict')
    _ax.set_title('Rating Agreement Rate vs Rating Frequency')
    plt.tight_layout()
    plt.savefig('q7_frequency_vs_skill.png', bbox_inches='tight')
    plt.show()
    r1, p1 = stats.spearmanr(skill_df['n_notes_written'], skill_df['avg_note_intercept'])
    # (c) Agreement rate vs total ratings (rater skill)
    r2, p2 = stats.spearmanr(skill_df2['n_notes_written'], skill_df2['pct_notes_helpful'])
    r3, p3 = stats.spearmanr(skill_df3['n_ratings'], skill_df3['agreement_rate'])
    print(f'Spearman r (notes written vs avg intercept):      {r1:.3f}  p={p1:.2e}')
    print(f'Spearman r (notes written vs pct helpful):        {r2:.3f}  p={p2:.2e}')
    # Correlations
    print(f'Spearman r (ratings given vs agreement rate):     {r3:.3f}  p={p3:.2e}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Findings


    Question 1: Users who rate right-leaning notes as helpful have a higher avg_factor_helpful than avg_factor_not_helpful, and vice versa for left-leaning raters. The factor_diff is internally consistent — users above the y=x line prefer right-leaning notes, those below prefer left-leaning notes. This directional split aligns with the expectation that external party labels would predict the same groups.

    Question 2: There is a small but highly significant positive correlation between contribution frequency and |factor_diff| (Spearman r ≈ 0.08, p < 10⁻²³⁰). More active contributors tend to show a slightly stronger partisan lean.

    Question 3: About 46% of users are left-biased and ~45% are right-biased at the |diff|>0.1 threshold. The distribution is roughly symmetric with only ~9% neutral, indicating most contributors show a consistent partisan preference when rating notes as helpful or not helpful.

    Question 4: Users who rate one side helpful strongly also rate more notes from that side. "Strong Left" users rate ~13 left-leaning notes vs ~10 right-leaning notes (median). "Strong Right" users show the opposite. The helpfulness asymmetry mirrors the volume asymmetry.

    Question 5: Most users are highly focused: median tweets rated = 3, median tweets written about = 1. The distribution is heavily right-skewed. A small group of power users spans many more tweets and topics than the typical contributor.

    Question 6: More active users cover more tweets in absolute terms, but their tweet diversity ratio (unique tweets / total ratings) falls as contributions increase.  Frequent contributors tend to revisit the same set of tweets rather than broadening their scope.

    Question 7: More bigger note writers earn helpful labels more often (Spearman r ≈ 0.21 for pct_notes_helpful). Note intercept shows a weak positive trend (r ≈ 0.03). Surprisingly, rater agreement rate decreases with rating frequency (r ≈ -0.22), suggesting very active raters may be more partisan or rate notes before community consensus stabilizes.
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
