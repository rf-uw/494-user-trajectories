import marimo

__generated_with = "0.22.4"
app = marimo.App(auto_download=["html", "ipynb"])


@app.cell
def _():
    import polars as pl
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go
    import marimo as mo

    return Path, go, mo, np, pl, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    THIS USES THE FULL DATA, WHICH IS ONLY AVAILABLE ON OUR SERVER. IF YOU RUN LOCALLY, REPLACE "user_...\_traj.parquet" WITH "sample_user_...\_traj.parquet"
    """)
    return


@app.cell
def _(Path, pl):
    # Resolve data directory relative to this notebook location
    archive_dir = Path(r"../../data")

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

    joined_chronological = (
        notes.join(
            ratings,
            on=["participantId", "userMonth", "calendarMonth"],
            how="full",
            coalesce=True,
        )
        .join(
            requests,
            on=["participantId", "userMonth", "calendarMonth"],
            how="full",
            coalesce=True,
        )
        .with_columns(
            [
                pl.col("notesWritten").fill_null(0),
                pl.col("notesRated").fill_null(0),
                pl.col("notesRequested").fill_null(0),
            ]
        )
        .sort(["participantId", "calendarMonth", "userMonth"])
    )

    # Save the chronological joined table before further aggregation.
    joined_output_path = archive_dir / "user_month_actions_chronological.parquet"
    joined_chronological.write_parquet(joined_output_path)

    user_month_actions = (
        joined_chronological.group_by(["participantId", "userMonth", "calendarMonth"])
        .agg(
            [
                pl.sum("notesWritten").alias("notesWritten"),
                pl.sum("notesRated").alias("notesRated"),
                pl.sum("notesRequested").alias("notesRequested"),
            ]
        )
        .sort(["participantId", "calendarMonth", "userMonth"])
    )

    user_month_actions.head()
    return (user_month_actions,)


@app.cell
def _(pl, user_month_actions):
    main_df = user_month_actions

    # Parse calendar month for month-wise panel expansion.
    main_df_dates = main_df.with_columns(
        pl.col("calendarMonth").str.strptime(pl.Date, "%Y-%m").alias("calendarDate")
    )

    user_starts = main_df_dates.group_by("participantId").agg(
        [
            pl.col("calendarDate").min().alias("firstCalendarDate"),
            pl.col("userMonth").min().cast(pl.Int64).alias("firstUserMonth"),
        ]
    )

    global_max_calendar = main_df_dates.select(pl.col("calendarDate").max()).row(0)[0]
    global_min_calendar = user_starts.select(pl.col("firstCalendarDate").min()).row(0)[0]

    all_months = pl.DataFrame(
        {
            "calendarDate": pl.date_range(
                start=global_min_calendar,
                end=global_max_calendar,
                interval="1mo",
                eager=True,
            )
        }
    )

    # Build a complete panel from each user's first observed month to the global max month.
    panel_index = (
        user_starts.select("participantId")
        .join(all_months, how="cross")
        .join(user_starts, on="participantId", how="left")
        .filter(pl.col("calendarDate") >= pl.col("firstCalendarDate"))
        .with_columns(
            (
                (pl.col("calendarDate").dt.year() - pl.col("firstCalendarDate").dt.year()) * 12
                + (pl.col("calendarDate").dt.month() - pl.col("firstCalendarDate").dt.month())
            ).alias("monthOffset")
        )
        .with_columns(
            [
                (pl.col("firstUserMonth") + pl.col("monthOffset")).cast(pl.Int64).alias("userMonth"),
                pl.col("calendarDate").dt.strftime("%Y-%m").alias("calendarMonth"),
            ]
        )
        .select(["participantId", "userMonth", "calendarMonth"])
    )

    classified_panel_df = (
        panel_index.join(
            main_df.select(["participantId", "calendarMonth", "notesWritten", "notesRated", "notesRequested"]),
            on=["participantId", "calendarMonth"],
            how="left",
        )
        .with_columns(
            [
                pl.col("notesWritten").fill_null(0).cast(pl.Int64),
                pl.col("notesRated").fill_null(0).cast(pl.Int64),
                pl.col("notesRequested").fill_null(0).cast(pl.Int64),
            ]
        )
        .with_columns(
            [
                (pl.col("notesWritten") == 1).cast(pl.Int8).alias("single_note_writer"),
                (pl.col("notesWritten").is_between(2, 9, closed="both")).cast(pl.Int8).alias("single_digit_writer"),
                (pl.col("notesWritten") >= 10).cast(pl.Int8).alias("double_digit_writer"),
                ((pl.col("notesWritten") == 0) & (pl.col("notesRated") == 1)).cast(pl.Int8).alias("single_note_rater"),
                ((pl.col("notesWritten") == 0) & pl.col("notesRated").is_between(2, 9, closed="both")).cast(pl.Int8).alias("single_digit_rater"),
                ((pl.col("notesWritten") == 0) & (pl.col("notesRated") >= 10)).cast(pl.Int8).alias("double_digit_rater"),
                ((pl.col("notesWritten") == 0) & (pl.col("notesRated") == 0) & (pl.col("notesRequested") == 1)).cast(pl.Int8).alias("single_post_requestor"),
                ((pl.col("notesWritten") == 0) & (pl.col("notesRated") == 0) & pl.col("notesRequested").is_between(2, 9, closed="both")).cast(pl.Int8).alias("single_digit_requestor"),
                ((pl.col("notesWritten") == 0) & (pl.col("notesRated") == 0) & (pl.col("notesRequested") >= 10)).cast(pl.Int8).alias("double_digit_requestor"),
                ((pl.col("notesWritten") == 0) & (pl.col("notesRated") == 0) & (pl.col("notesRequested") == 0)).cast(pl.Int8).alias("not_active"),
            ]
        )
        .with_columns(
            pl.when(pl.col("single_note_writer") == 1)
            .then(pl.lit("single_note_writer"))
            .when(pl.col("single_digit_writer") == 1)
            .then(pl.lit("single_digit_writer"))
            .when(pl.col("double_digit_writer") == 1)
            .then(pl.lit("double_digit_writer"))
            .when(pl.col("single_note_rater") == 1)
            .then(pl.lit("single_note_rater"))
            .when(pl.col("single_digit_rater") == 1)
            .then(pl.lit("single_digit_rater"))
            .when(pl.col("double_digit_rater") == 1)
            .then(pl.lit("double_digit_rater"))
            .when(pl.col("single_post_requestor") == 1)
            .then(pl.lit("single_post_requestor"))
            .when(pl.col("single_digit_requestor") == 1)
            .then(pl.lit("single_digit_requestor"))
            .when(pl.col("double_digit_requestor") == 1)
            .then(pl.lit("double_digit_requestor"))
            .otherwise(pl.lit("not_active"))
            .alias("activity_class")
        )
        .sort(["participantId", "calendarMonth", "userMonth"])
    )

    classified_panel_df.head()
    return (classified_panel_df,)


@app.cell
def _(classified_panel_df, pl):
    binary_cols = [
        "single_note_writer",
        "single_digit_writer",
        "double_digit_writer",
        "single_note_rater",
        "single_digit_rater",
        "double_digit_rater",
        "single_post_requestor",
        "single_digit_requestor",
        "double_digit_requestor",
        "not_active",
    ]

    violations = (
        classified_panel_df.with_columns(pl.sum_horizontal([pl.col(c) for c in binary_cols]).alias("num_active_classes"))
        .filter(pl.col("num_active_classes") != 1)
    )

    rows_breaking_exclusivity = violations.height
    users_breaking_exclusivity = violations.select(pl.col("participantId").n_unique()).item()

    print(f"Rows breaking exclusivity: {rows_breaking_exclusivity}")
    print(f"Users with at least one violating month: {users_breaking_exclusivity}")
    return


@app.cell
def _(classified_panel_df, np, pl, plt):
    states = [
        "double_digit_writer", "single_digit_writer", "single_note_writer",
        "double_digit_rater", "single_digit_rater", "single_note_rater",
        "double_digit_requestor", "single_digit_requestor", "single_post_requestor",
        "not_active",
    ]

    # Build month-to-next-month transitions within each user trajectory
    transitions = (
        classified_panel_df
        .sort(["participantId", "userMonth"])
        .with_columns(
            [
                pl.col("activity_class")
                .shift(-1)
                .over("participantId")
                .alias("next_state"),

                pl.col("userMonth")
                .shift(-1)
                .over("participantId")
                .alias("next_userMonth"),
            ]
        )
        .filter(
            pl.col("next_state").is_not_null()
            & (pl.col("next_userMonth") - pl.col("userMonth") == 1)
        )
        .select(["activity_class", "next_state"])
    )

    # Count transitions
    transition_counts = (
        transitions
        .group_by(["activity_class", "next_state"])
        .len()
        .rename({"len": "count"})
    )

    # Ensure all state pairs exist
    state_grid = (
        pl.DataFrame({"activity_class": states})
        .join(pl.DataFrame({"next_state": states}), how="cross")
    )

    transition_full = (
        state_grid
        .join(transition_counts, on=["activity_class", "next_state"], how="left")
        .with_columns(pl.col("count").fill_null(0).cast(pl.Int64))
    )

    # Row-normalized probabilities
    transition_matrix_long = (
        transition_full
        .join(
            transition_full
            .group_by("activity_class")
            .agg(pl.col("count").sum().alias("row_total")),
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

    transition_matrix = (
        transition_matrix_long
        .select(["activity_class", "next_state", "probability"])
        .pivot(
            index="activity_class",
            on="next_state",
            values="probability",
            aggregate_function="sum",
        )
    )

    # Reorder rows to canonical state order
    state_order = pl.DataFrame({
        "activity_class": states,
        "state_order": list(range(len(states))),
    })

    transition_matrix_ordered = (
        transition_matrix
        .select(["activity_class"] + states)
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
    return (states,)


@app.cell
def _(classified_panel_df, plt):
    sequence_lengths = classified_panel_df.group_by('participantId').len().rename({'len': 'sequence_length'})
    length_distribution = sequence_lengths.group_by('sequence_length').len().rename({'len': 'num_users'}).sort('sequence_length')
    x = length_distribution['sequence_length'].to_numpy()
    # Sequence length per user = number of user-month rows in the classified panel.
    y = length_distribution['num_users'].to_numpy()
    plt.figure(figsize=(10, 5))
    plt.bar(x, y, color='#4C78A8', width=0.9)
    plt.title('Sequence Length Distribution (Raw)')
    plt.xlabel('Sequence Length (months)')
    plt.ylabel('Number of Users')
    plt.tight_layout()
    plt.show()
    sl = sequence_lengths['sequence_length']
    print(f'Total users: {sequence_lengths.height}')
    print(f'Total user-month rows: {int(sl.sum())}')
    print(f'Min sequence length: {int(sl.min())}')
    print(f'Q1 sequence length: {float(sl.quantile(0.25)):.2f}')
    print(f'Median sequence length: {float(sl.median()):.2f}')
    print(f'Mean sequence length: {float(sl.mean()):.2f}')
    print(f'Q3 sequence length: {float(sl.quantile(0.75)):.2f}')
    # Raw frequency bar chart only.
    print(f'Max sequence length: {int(sl.max())}')
    print(f'Std sequence length: {float(sl.std()):.2f}')
    # Extended metadata summary.
    length_distribution
    return


@app.cell
def _(classified_panel_df, pl):
    # Build transitions

    transitions_by_month = (
        classified_panel_df
        .sort("participantId", "userMonth")
        .with_columns(
            pl.col("activity_class").shift(-1).over("participantId").alias("next_state"),
            pl.col("userMonth").shift(-1).over("participantId").alias("next_userMonth"),
        )
        .filter(pl.col("next_state").is_not_null() & (pl.col("next_userMonth") - pl.col("userMonth") == 1))
        .select("participantId", "userMonth", pl.col("activity_class").alias("from_state"), "next_state")
    )

    # Node setup

    state_colors = {
        # Writers — red
        "single_note_writer":  "rgba(252,146,114,0.85)",
        "single_digit_writer": "rgba(222,45,38,0.85)",
        "double_digit_writer": "rgba(165,15,21,0.85)",
        # Raters — blue
        "single_note_rater":   "rgba(158,202,225,0.85)",
        "single_digit_rater":  "rgba(49,130,189,0.85)",
        "double_digit_rater":  "rgba(8,48,107,0.85)",
        # Requestors — green
        "single_post_requestor":  "rgba(161,217,155,0.85)",
        "single_digit_requestor": "rgba(49,163,84,0.85)",
        "double_digit_requestor": "rgba(0,68,27,0.85)",
        # Inactive — gray
        "not_active": "rgba(150,150,150,0.85)",
    }
    return state_colors, transitions_by_month


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
            line=dict(color="black", width=0.3),
            label=node_labels,
            color=node_colors,
            x=[0.01] * n_from + [0.99] * n_to,
            y=[(i + 0.5) / n_from for i in range(n_from)] +
              [(i + 0.5) / n_to for i in range(n_to)],
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

        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=node_cfg,
            link=link,
        )])

        fig.update_layout(
            title_text=f"Empirical Sankey — userMonth {month}",
            font_size=11,
            height=400
        )

        return fig

    return (build_sankey,)


@app.cell
def _(build_sankey, from_dropdown, month_slider):
    build_sankey(month_slider.value, from_dropdown.value)
    return


@app.cell
def _(mo, states):
    # --- marimo controls ---

    month_slider = mo.ui.slider(0, 39, value=0, label="User Month")
    from_dropdown = mo.ui.dropdown(
        options=["all"] + states,
        value="all",
        label="From state"
    )

    mo.vstack([month_slider, from_dropdown])
    return from_dropdown, month_slider


if __name__ == "__main__":
    app.run()
