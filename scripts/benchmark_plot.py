# /// script
# dependencies = [
#     "altair==6.1.0",
#     "marimo",
#     "polars==1.40.1",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import polars as pl

    return alt, pl


@app.cell
def _(pl):
    # Prepare the data
    data = {
        "benchmark": ["Human Eval", "Human Eval", "MBPP", "MBPP"],
        # "model": ["lfm2.5-instruct", "lfm2.5-instruct-coder", "lfm2.5-instruct", "lfm2.5-instruct-coder"],
        "model": ["base", "fine-tuned", "base", "fine-tuned"],
        "pass@1": [0.521, 0.509, 0.316, 0.471],
        "change": [None, -0.023, None, 0.491],
    }

    df = pl.DataFrame(data)
    df
    return (df,)


@app.cell
def _(alt, df):
    # Interactivity
    selection = alt.selection_point(
        fields=["model"], on="mouseover", bind="legend"
    )

    # Create the chart
    _chart = (
        alt
        .Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                field="benchmark",
                type="nominal",
                title="Benchmark",
                sort="descending",
            ),
            y=alt.Y(
                field="pass@1",
                type="quantitative",
                title="pass@1 accuracy",
            ),
            color=alt.Color(
                field="model",
                type="nominal",
                scale=alt.Scale(
                    domain=["base", "fine-tuned"],
                    # Gray for base, Blue for fine-tuned
                    range=["#d2d2d4", "#1f77b4"],
                ),
            ),
            # Separate the bars by model type
            xOffset=alt.XOffset(field="model", type="nominal"),
            opacity=alt.condition(selection, alt.value(1.0), alt.value(0.3)),
            tooltip=[
                alt.Tooltip(field="model"),
                alt.Tooltip(field="benchmark", title="Benchmark"),
                alt.Tooltip(
                    field="pass@1", format=",.3f", title="pass@1 accuracy"
                ),
                alt.Tooltip(field="model"),
                alt.Tooltip(field="change", format="+.1%"),
            ],
        )
        .add_params(selection)
    )

    # Add text labels for the "change" column
    _text = _chart.mark_text(
        # Shift text above the bar
        dy=-5,
        baseline="bottom",
        fontWeight="bold",
    ).encode(
        text=alt.Text(field="pass@1", format=".1%"),
        color=alt.value("white"),
        # Color by change if desired:
        # color=alt.condition(
        #     alt.datum.change >= 0, alt.value("#38ba5b"), alt.value("#ba3838")
        # ),
    )

    _final_chart = (
        (_chart + _text)
        .properties(
            title="LFM2.5-instruct (base) vs. LFM2.5-instruct-coder (fine-tuned)",
            height=300,
            width=450,
            background="black",
        )
        .configure_axis(
            # Make x-axis labels horizontal
            labelAngle=0,
            # Remove grid lines
            grid=False,
        )
        .configure_title(color="white")
        .configure_legend(labelColor="white", titleColor="white")
        .configure_view(stroke=None, fill="black")
    )
    _final_chart
    return


if __name__ == "__main__":
    app.run()
