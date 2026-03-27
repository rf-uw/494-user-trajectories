# src/filter_polars.py
import polars as pl


def filter_by_date(df, year, month):
    # Create end date (last day of the month)
    if month == 12:
        end_year = year + 1
        end_month = 1
    else:
        end_year = year
        end_month = month + 1

    # Polars filtering
    df_filtered = df.with_columns(
        pl.from_epoch(pl.col('createdAtMillis'), time_unit='ms').alias('date')
    ).filter(
        pl.col('date') < pl.datetime(end_year, end_month, 1)
    ).drop('date')

    return df_filtered
