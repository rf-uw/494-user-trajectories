import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 8]})
    df
    return (df,)


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
