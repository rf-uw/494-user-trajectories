import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full", auto_download=["html", "ipynb"])


@app.cell
def _():
    import polars as pl
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go
    import marimo as mo
    import plotly.express as px
    from datetime import date
    import colorsys

    return Path, colorsys, go, mo, pl, plt, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    THIS USES THE FULL DATA, WHICH IS ONLY AVAILABLE ON OUR SERVER. IF YOU RUN LOCALLY, REPLACE "user_...\_traj.parquet" WITH "sample_user_...\_traj.parquet"
    """)
    return


@app.cell
def _(Path, pl):
    # Load data
    archive_dir = Path("../../data")

    note_path = archive_dir / "user_note_traj.parquet"
    rating_path = archive_dir / "user_rating_traj.parquet"
    request_path = archive_dir / "user_request_traj.parquet"

    notes = (
        pl.read_parquet(note_path)
        .select(["noteAuthorParticipantId", "userMonth", "calendarMonth", "notesCreated"])
        .rename(
            {
                "noteAuthorParticipantId": "participantId",
                "notesCreated": "notesWritten",
            }
        )
    )

    ratings = (
        pl.read_parquet(rating_path)
        .select(["raterParticipantId", "userMonth", "calendarMonth", "notesRated"])
        .rename({"raterParticipantId": "participantId"})
    )

    requests = (
        pl.read_parquet(request_path)
        .select(["requesterParticipantId", "userMonth", "calendarMonth", "requestsMade"])
        .rename(
            {
                "requesterParticipantId": "participantId",
                "requestsMade": "notesRequested",
            }
        )
    )
    return notes, ratings, requests


@app.cell
def _(apply_rules, month_activity_rules, notes, pl, ratings, requests):
    user_months = (
        notes.join(
            ratings, how="full", coalesce=True,
            on=["participantId", "userMonth", "calendarMonth"],)
        .join(
            requests, how="full", coalesce=True,
            on=["participantId", "userMonth", "calendarMonth"],)
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
        .drop("userFirstCalendarMonth", "yearsSinceJoining", "monthsSinceJoining")
    )


    user_months = (
        user_months
        .join(
            empty_user_months, how="full", coalesce=True,
            on=["participantId", "userMonth", "calendarMonth", "calendarDate"])
        .with_columns(
                pl.col("notesWritten").fill_null(0),
                pl.col("notesRated").fill_null(0),
                pl.col("notesRequested").fill_null(0),
                pl.col("activeMonth").fill_null(False),)
        .sort(["participantId", "calendarMonth", "userMonth"])
        .with_columns(month_role=apply_rules(month_activity_rules)))
    return (user_months,)


@app.cell
def _(pl):
    total_activity_rules = [
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

    month_activity_rules = [
        ("double_digit_writer", pl.col("notesWritten") >= 10),
        ("single_digit_writer", pl.col("notesWritten") >= 2),
        ("single_note_writer",  pl.col("notesWritten") == 1),

        ("double_digit_rater",  pl.col("notesRated") >= 10),
        ("single_digit_rater",  pl.col("notesRated") >= 2),
        ("single_note_rater",   pl.col("notesRated") == 1),

        ("double_digit_requestor", pl.col("notesRequested") >= 10),
        ("single_digit_requestor", pl.col("notesRequested") >= 2),
        ("single_post_requestor",  pl.col("notesRequested") == 1),

        ("not_active", pl.lit(True)),
    ]

    # Build the classification expression from rules
    def apply_rules(rules):
        # Apply rules in reverse order to ensure first match takes precedence
        expr = pl.lit(None, dtype=pl.String)
        for label, condition in reversed(rules):
            expr = pl.when(condition).then(pl.lit(label)).otherwise(expr)
    
        # Extract ordered labels from rules
        activity_levels = [label for label, _ in total_activity_rules]
    
        # Make the column an ordered categorical with the specified levels
        expr = expr.cast(pl.Enum(activity_levels))
    
        return expr

    return apply_rules, month_activity_rules, total_activity_rules


@app.cell
def _(
    apply_rules,
    month_activity_rules,
    pl,
    total_activity_rules,
    user_months,
):
    users = (
        user_months
        .group_by("participantId")
        .agg(
            notesWritten = pl.col("notesWritten").sum(),
            notesRated = pl.col("notesRated").sum(),
            notesRequested = pl.col("notesRequested").sum(),
            userFirstCalendarMonth = pl.col("calendarMonth").min(),
            userLastActiveCalendarMonth = pl.col("userLastActiveCalendarMonth").max(),
            nActiveMonths = pl.col("activeMonth").sum(),
            age = pl.col("userMonth").max(),
            activeWindow = pl.col("userMonth").filter(pl.col("activeMonth")).max() - pl.col("userMonth").min() + 1,
            firstMonthRole = pl.col("month_role").filter(pl.col("activeMonth")).first(),
            lastMonthRole = pl.col("month_role").filter(pl.col("activeMonth")).last(),
            *[(pl.col("month_role") == role).sum().alias(f"nMonths{role}") for role, _ in month_activity_rules[:-1]],
        )
        .with_columns(
            total_role=apply_rules(total_activity_rules),
            *[(pl.col(f"nMonths{role}") / pl.col("activeWindow")).alias(f"pctActiveMonths{role}") for role, _ in month_activity_rules[:-1]]
        )
    )
    return (users,)


@app.cell
def _():
    role_colors = {
        # Writers — red
        "single_note_writer": "rgba(252,146,114,0.85)",
        "single_digit_writer": "rgba(222,45,38,0.85)",
        "double_digit_writer": "rgba(165,15,21,0.85)",
        # Raters — blue
        "single_note_rater": "rgba(158,202,225,0.85)",
        "single_digit_rater": "rgba(49,130,189,0.85)",
        "double_digit_rater": "rgba(8,48,107,0.85)",
        # Requestors — green
        "single_post_requestor": "rgba(161,217,155,0.85)",
        "single_digit_requestor": "rgba(49,163,84,0.85)",
        "double_digit_requestor": "rgba(0,68,27,0.85)",
        # Inactive — gray
        "not_active": "rgba(150,150,150,0.85)",
    }
    return (role_colors,)


@app.cell
def _(pl, users):
    users.group_by("firstMonthRole", "userFirstCalendarMonth").agg(median_lifetime=pl.col("activeWindow").median()).sort("userFirstCalendarMonth", "firstMonthRole")
    return


@app.cell
def _(pl, users):
    cohort_df = (
        users.group_by("firstMonthRole", "userFirstCalendarMonth")
        .agg(
            n=pl.len(),
            median_lifetime=pl.col("activeWindow").median(),
            median_active_months=pl.col("nActiveMonths").median(),
            median_notes_written=pl.col("notesWritten").median(),
            median_notes_rated=pl.col("notesRated").median(),
            median_notes_requested=pl.col("notesRequested").median(),
        )
        .sort("userFirstCalendarMonth", "firstMonthRole")
    )

    roles = cohort_df["firstMonthRole"].unique().sort().to_list()
    return (cohort_df,)


@app.cell
def _(cohort_df, mo, px):

    _fig = px.line(
        cohort_df,
        x="userFirstCalendarMonth",
        y="median_lifetime",
        color="firstMonthRole",
        markers=True,
        title="Median Lifetime by Cohort Month and Role",
        labels={
            "userFirstCalendarMonth": "First Calendar Month",
            "median_lifetime": "Median Lifetime",
        },
        height=550 ,
    )

    _fig.update_yaxes(matches=None)
    _fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))


    mo.ui.plotly(_fig)
    return


@app.cell
def _(cohort_df, mo, px, role_colors):
    _fig = px.line(
        cohort_df,
        x="userFirstCalendarMonth",
        y="median_active_months",
        color="firstMonthRole",
        markers=True,
        color_discrete_map=role_colors,
        title="Median Number of Active Months by Cohort Month and Role",
        labels={
            "userFirstCalendarMonth": "First Calendar Month",
            "median_active_months": "Median N Active Months",
        },
        height=550 ,
    )

    _fig.update_yaxes(matches=None)
    _fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))


    mo.ui.plotly(_fig)
    return


@app.cell
def _(cohort_df, mo, px, role_colors):
    _fig = px.line(
        cohort_df,
        x="userFirstCalendarMonth",
        y="n",
        color="firstMonthRole",
        markers=True,
        color_discrete_map=role_colors,
        title="Number of Users Joining in a Month by First Role",
        labels={
            "userFirstCalendarMonth": "First Calendar Month",
            "n": "Num Users",
        },
        height=550,
        log_y=True,
    )

    _fig.update_yaxes(matches=None)
    _fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))


    mo.ui.plotly(_fig)
    return


@app.cell
def _(colorsys, mo, pl, px, user_months):
    def plot_notes_by_join_month(user_months, metric: str, pct: bool = False):
        label_map = {
            "notesWritten": "Notes Written",
            "notesRated": "Notes Rated",
            "notesRequested": "Notes Requested",
        }
        label = label_map[metric]

        join_month_df = (
            user_months
            .with_columns(
                (pl.col("calendarDate") - pl.duration(days=pl.col("userMonth") * 30))
                .dt.strftime("%Y-%m")
                .alias("joinMonth")
            )
            .group_by(["calendarMonth", "calendarDate", "joinMonth"])
            .agg(pl.col(metric).sum())
            .with_columns(
                (pl.col(metric) / pl.col(metric).sum().over("calendarMonth") * 100)
                .alias("pctMetric")
            )
            .sort(["calendarDate", "joinMonth"])
            .filter(pl.col("calendarMonth") >= "2023-01")
        )

        y_col = "pctMetric" if pct else metric
        y_label = f"% of {label}" if pct else label
        title = f"{'% of' if pct else 'Raw'} {label} by User Join Month"

        join_months = sorted(join_month_df["joinMonth"].unique().to_list())
        join_years  = sorted(set(m[:4] for m in join_months))
        year_hues   = {year: i / len(join_years) for i, year in enumerate(join_years)}

        color_map = {}
        for year in join_years:
            months_in_year = sorted(m for m in join_months if m.startswith(year))
            n = len(months_in_year)
            hue = year_hues[year]
            for i, month in enumerate(months_in_year):
                lightness = 0.25 + 0.45 * (i / max(n - 1, 1))
                r, g, b = colorsys.hls_to_rgb(hue, lightness, 0.75)
                color_map[month] = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

        fig = px.area(
            join_month_df,
            x="calendarMonth",
            y=y_col,
            color="joinMonth",
            color_discrete_map=color_map,
            title=title,
            labels={"calendarMonth": "Calendar Month", y_col: y_label, "joinMonth": "Join Month"},
            height=500,
            width=600,
        )
        fig.update_traces(line=dict(width=0.4))

        boundary_years = join_years[:-1]
        for year in boundary_years:
            months_up_to = [m for m in join_months if m[:4] <= year]
            boundary_df = (
                join_month_df
                .filter(pl.col("joinMonth").is_in(months_up_to))
                .group_by(["calendarMonth", "calendarDate"])
                .agg(pl.col(y_col).sum().alias("cumVal"))
                .sort("calendarDate")
            )
            fig.add_scatter(
                x=boundary_df["calendarMonth"].to_list(),
                y=boundary_df["cumVal"].to_list(),
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
            )

        if pct:
            fig.update_yaxes(range=[0, 100])

        return mo.ui.plotly(fig)


    mo.vstack([
        plot_notes_by_join_month(user_months, "notesWritten", pct=False),
        plot_notes_by_join_month(user_months, "notesRated",   pct=False),
        plot_notes_by_join_month(user_months, "notesRequested", pct=False),
    ])
    return


@app.cell
def _(colorsys, mo, pl, px, user_months):
    def plot_active_users_by_join_month(user_months, pct: bool = False, user_type: str = "total"):
        type_map = {
            "total":     (pl.col("activeMonth"),          "Active Users"),
            "writers":   (pl.col("notesWritten") > 0,     "Active Writers"),
            "raters":    (pl.col("notesRated") > 0,       "Active Raters"),
            "requesters":(pl.col("notesRequested") > 0,   "Active Requesters"),
        }
        filter_expr, type_label = type_map[user_type]

        join_month_df = (
            user_months
            .with_columns(
                (pl.col("calendarDate") - pl.duration(days=pl.col("userMonth") * 30))
                .dt.strftime("%Y-%m")
                .alias("joinMonth")
            )
            .filter(filter_expr)
            .group_by(["calendarMonth", "calendarDate", "joinMonth"])
            .agg(pl.len().alias("activeUsers"))
            .with_columns(
                (pl.col("activeUsers") / pl.col("activeUsers").sum().over("calendarMonth") * 100)
                .alias("pctMetric")
            )
            .sort(["calendarDate", "joinMonth"])
            .filter(pl.col("calendarMonth") >= "2023-01")
        )

        y_col = "pctMetric" if pct else "activeUsers"
        y_label = f"% of {type_label}" if pct else type_label
        title = f"{'% of' if pct else ''} {type_label} by Join Month".strip()

        join_months = sorted(join_month_df["joinMonth"].unique().to_list())
        join_years  = sorted(set(m[:4] for m in join_months))
        year_hues   = {year: i / len(join_years) for i, year in enumerate(join_years)}

        color_map = {}
        for year in join_years:
            months_in_year = sorted(m for m in join_months if m.startswith(year))
            n = len(months_in_year)
            hue = year_hues[year]
            for i, month in enumerate(months_in_year):
                lightness = 0.25 + 0.45 * (i / max(n - 1, 1))
                r, g, b = colorsys.hls_to_rgb(hue, lightness, 0.75)
                color_map[month] = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

        fig = px.area(
            join_month_df,
            x="calendarMonth",
            y=y_col,
            color="joinMonth",
            color_discrete_map=color_map,
            title=title,
            labels={"calendarMonth": "Calendar Month", y_col: y_label, "joinMonth": "Join Month"},
            height=500,
            width=600,
        )
        fig.update_traces(line=dict(width=0.4))

        boundary_years = join_years[:-1]
        for year in boundary_years:
            months_up_to = [m for m in join_months if m[:4] <= year]
            boundary_df = (
                join_month_df
                .filter(pl.col("joinMonth").is_in(months_up_to))
                .group_by(["calendarMonth", "calendarDate"])
                .agg(pl.col(y_col).sum().alias("cumVal"))
                .sort("calendarDate")
            )
            fig.add_scatter(
                x=boundary_df["calendarMonth"].to_list(),
                y=boundary_df["cumVal"].to_list(),
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
            )

        if pct:
            fig.update_yaxes(range=[0, 100])

        return mo.ui.plotly(fig)


    mo.vstack([
        plot_active_users_by_join_month(user_months, user_type="total"),
        plot_active_users_by_join_month(user_months, user_type="writers"),
        plot_active_users_by_join_month(user_months, user_type="raters"),
        plot_active_users_by_join_month(user_months, user_type="requesters"),
    ])
    return


@app.cell
def _():
    return


@app.cell
def _(month_activity_rules, pl, users):
    users.group_by("total_role").agg(
        n = pl.len(),
        median_notes_written = pl.col("notesWritten").median(),
        median_notes_rated = pl.col("notesRated").median(),
        median_notes_requested = pl.col("notesRequested").median(),
        pct_having_rated = (pl.col("notesRated") > 0).mean(),
        pct_having_requested = (pl.col("notesRequested") > 0).mean(),
        medianActiveWindow = pl.col("activeWindow").median(),
        *[(pl.col(f"nMonths{role}")).mean().alias(f"avg_nMonths{role}") for role, _ in month_activity_rules[:-1]],
        *[(pl.col(f"pctActiveMonths{role}")).mean().alias(f"avg_pctActiveMonths{role}") for role, _ in month_activity_rules[:-1]]
    ).sort("total_role")
    return


app._unparsable_cell(
    r"""
    # Build month-to-next-month transitions within each user trajectory
    transitions = (
        user_months
        .with_columns(
            [
                prev_role=pl.col("month_role")
                .shift(-1)
                .over("participantId")
                .alias("next_state"),

                pl.col("userMonth").shift(-1).over("participantId").alias("next_userMonth"),
            ]
        )
        .filter(
            pl.col("next_state").is_not_null()
            & (pl.col("next_userMonth") - pl.col("userMonth") == 1)
        )
        .select(["activity_class", "next_state"])
    )

    # Count transitions
    transition_counts = transitions.group_by(["activity_class", "next_state"]).len().rename({"len": "count"})

    # Ensure all state pairs exist
    state_grid = pl.DataFrame({"activity_class": states}).join(
        pl.DataFrame({"next_state": states}),
        how="cross",
    )

    transition_full = (
        state_grid.join(transition_counts, on=["activity_class", "next_state"], how="left")
        .with_columns(pl.col("count").fill_null(0).cast(pl.Int64))
    )

    # Row-normalized probabilities
    transition_matrix_long = (
        transition_full.join(
            transition_full.group_by("activity_class").agg(
                pl.col("count").sum().alias("row_total")
            ),
            on="activity_class",
            how="left",
        )
        .with_columns(
            pl.when(pl.col("row_total") > 0)
            .then(pl.col("count") / pl.col("row_total"))
            .otherwise(0.0)
            .alias("probability")
        )
    )

    transition_matrix = transition_matrix_long.select(
        ["activity_class", "next_state", "probability"]
    ).pivot(
        index="activity_class",
        on="next_state",
        values="probability",
        aggregate_function="sum",
    )

    # Reorder rows to canonical state order
    state_order = pl.DataFrame(
        {
            "activity_class": states,
            "state_order": list(range(len(states))),
        }
    )

    transition_matrix_ordered = (
        transition_matrix.select(["activity_class"] + states)
        .join(state_order, on="activity_class", how="left")
        .sort("state_order")
        .drop("state_order")
    )

    # Plot heatmap
    _heat_values = transition_matrix_ordered.select(states).to_numpy()

    plt.figure(figsize=(12, 8))

    _img = plt.imshow(
        _heat_values,
        cmap="YlOrRd",
        aspect="auto",
        vmin=0,
        vmax=max(1e-9, float(np.max(_heat_values))),
    )

    plt.colorbar(_img, label="Transition Probability")

    plt.xticks(
        ticks=np.arange(len(states)),
        labels=states,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        ticks=np.arange(len(states)),
        labels=transition_matrix_ordered["activity_class"].to_list(),
    )

    # Annotate cells
    for _i in range(_heat_values.shape[0]):
        for _j in range(_heat_values.shape[1]):
            _value = _heat_values[_i, _j]
            _text_color = "white" if _value > 0.5 else "black"

            plt.text(
                _j,
                _i,
                f"{_value:.3f}",
                ha="center",
                va="center",
                color=_text_color,
                fontsize=8,
            )

    plt.title("Empirical Transition Matrix (Row-Normalized)")
    plt.xlabel("Next State")
    plt.ylabel("Current State")

    plt.tight_layout()
    plt.show()

    transition_matrix_ordered
    """,
    name="_"
)


@app.cell
def _(classified_panel_df, plt):
    sequence_lengths = classified_panel_df.group_by("participantId").len().rename(
        {"len": "sequence_length"}
    )
    length_distribution = (
        sequence_lengths.group_by("sequence_length")
        .len()
        .rename({"len": "num_users"})
        .sort("sequence_length")
    )

    x = length_distribution["sequence_length"].to_numpy()
    y = length_distribution["num_users"].to_numpy()

    plt.figure(figsize=(10, 5))
    plt.bar(x, y, color="#4C78A8", width=0.9)
    plt.title("Sequence Length Distribution (Raw)")
    plt.xlabel("Sequence Length (months)")
    plt.ylabel("Number of Users")
    plt.tight_layout()
    plt.show()

    sl = sequence_lengths["sequence_length"]
    print(f"Total users: {sequence_lengths.height}")
    print(f"Total user-month rows: {int(sl.sum())}")
    print(f"Min sequence length: {int(sl.min())}")
    print(f"Q1 sequence length: {float(sl.quantile(0.25)):.2f}")
    print(f"Median sequence length: {float(sl.median()):.2f}")
    print(f"Mean sequence length: {float(sl.mean()):.2f}")
    print(f"Q3 sequence length: {float(sl.quantile(0.75)):.2f}")
    print(f"Max sequence length: {int(sl.max())}")
    print(f"Std sequence length: {float(sl.std()):.2f}")

    length_distribution
    return


@app.cell
def _(classified_panel_df, pl):
    transitions_by_month = (
        classified_panel_df.sort(["participantId", "userMonth"])
        .with_columns(
            pl.col("activity_class").shift(-1).over("participantId").alias("next_state"),
            pl.col("userMonth").shift(-1).over("participantId").alias("next_userMonth"),
        )
        .filter(
            pl.col("next_state").is_not_null()
            & (pl.col("next_userMonth") - pl.col("userMonth") == 1)
        )
        .select(
            "participantId",
            "userMonth",
            pl.col("activity_class").alias("from_state"),
            "next_state",
        )
    )


    return (transitions_by_month,)


@app.cell
def _(go, pl, state_colors, states, transitions_by_month):
    def build_sankey(month: int, from_filter: str):
        df = transitions_by_month.filter(pl.col("userMonth") == month)

        if from_filter != "all":
            df = df.filter(pl.col("from_state") == from_filter)

        if df.height == 0:
            return go.Figure()

        edges = (
            df.group_by("from_state", "next_state").len().rename({"len": "count"})
            .with_columns(
                (pl.col("count") / pl.col("count").sum().over("from_state")).alias("prob")
            )
            .sort("from_state", "next_state")
        )

        used_from = [s for s in states if s in edges["from_state"]]
        used_to = [s for s in states if s in edges["next_state"]]

        from_idx = {s: i for i, s in enumerate(used_from)}
        to_idx = {s: i + len(used_from) for i, s in enumerate(used_to)}

        node_labels = [f"from: {s}" for s in used_from] + [f"to: {s}" for s in used_to]
        node_colors = [state_colors[s] for s in used_from] + [state_colors[s] for s in used_to]

        n_from = len(used_from)
        n_to = len(used_to)

        node_cfg = dict(
            pad=10,
            thickness=12,
            line={"color": "black", "width": 0.3},
            label=node_labels,
            color=node_colors,
            x=[0.01] * n_from + [0.99] * n_to,
            y=[(i + 0.5) / n_from for i in range(n_from)]
            + [(i + 0.5) / n_to for i in range(n_to)],
        )

        link = {
            "source": [from_idx.get(s) for s in edges["from_state"]],
            "target": [to_idx.get(s) for s in edges["next_state"]],
            "value": edges["count"].to_list(),
            "color": [state_colors[s] for s in edges["from_state"]],
            "customdata": [
                f"month={month}<br>{f} -> {t}<br>count={c}<br>P(j|i)={p:.3f}"
                for f, t, c, p in zip(
                    edges["from_state"], edges["next_state"], edges["count"], edges["prob"]
                )
            ],
            "hovertemplate": "%{customdata}<extra></extra>",
        }

        fig = go.Figure(
            data=[
                go.Sankey(
                    arrangement="snap",
                    node=node_cfg,
                    link=link,
                )
            ]
        )

        fig.update_layout(
            title_text=f"Empirical Sankey — userMonth {month}",
            font_size=11,
            height=400,
        )

        return fig

    return (build_sankey,)


@app.cell
def _(build_sankey, from_dropdown, month_slider):
    build_sankey(month_slider.value, from_dropdown.value)
    return


@app.cell
def _(mo, states):
    month_slider = mo.ui.slider(0, 39, value=0, label="User Month")
    from_dropdown = mo.ui.dropdown(
        options=["all"] + states,
        value="all",
        label="From state",
    )

    mo.vstack([month_slider, from_dropdown])
    return from_dropdown, month_slider


if __name__ == "__main__":
    app.run()
