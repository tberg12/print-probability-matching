import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Forty Sermons Character Damage Visualizations

    1. Individual character matches across gatherings
    2. Same as 1 but without showing each character
    3. Same as 1 but broken down by character class
    4. Matrix showing shared damages from gathering to gathering
    5. Network showing linked gatherings
    6. Table of matches
    """
    )
    return


@app.cell(hide_code=True)
def _(alt, full_sample):
    images_l = (
        alt.Chart(full_sample)
        .mark_image()
        .encode(y=alt.Y("letter_with_class:O"), x=alt.X("gathering"), url="image")
    ).properties(
        width=900,
        # height=180,
    )

    # colors_l = (
    #     alt.Chart(full_sample)
    #     .mark_rect()
    #     .encode(
    #         y=alt.Y("letter_with_class:O"),
    #         x=alt.X("gathering"),
    #         color=alt.Color("Font:N").scale(scheme="category10"),
    #         opacity=alt.value(0.3),
    #     )
    # )

    # (images_l + colors_l).configure_axis(
    #     grid=True, tickBand="extent"
    # )  # .facet(row="letter:N")

    images_l.configure_axis(
        grid=True, tickBand="extent"
    )  # .facet(row="letter:N")
    return


@app.cell(hide_code=True)
def _(alt, full_sample):
    alt.Chart(full_sample).mark_rect(height=5).encode(
        y=alt.Y("letter_with_class:O", title="damage"),
        x=alt.X("gathering"),
        color=alt.Color("Font:N", legend=None),
        opacity=alt.value(0.3),
        # row=alt.Row("letter:O"),
    ).properties(height=alt.Step(10))
    return


@app.cell(hide_code=True)
def _(alt, full_sample):
    images = (
        alt.Chart(full_sample)
        .mark_image()
        .encode(y=alt.Y("letter_class:O"), x=alt.X("gathering"), url="image")
    ).properties(
        width=900,
        # height=180,
    )

    colors = (
        alt.Chart(full_sample)
        .mark_rect()
        .encode(
            y=alt.Y("letter_class:O", title="damage"),
            x=alt.X("gathering"),
            color=alt.Color("Font:N", legend=None),
            opacity=alt.value(0.3),
        )
    )

    (images + colors).facet(row="letter:N")
    return


@app.cell(hide_code=True)
def _():
    # alt.Chart(edgelist).mark_rect().encode(
    #     x=alt.X("source").scale(domain=full_sample["gathering"].unique()),
    #     y=alt.Y("target").scale(domain=full_sample["gathering"].unique()),
    #     color=alt.Color("weight", legend=None),
    # ).configure_axis(tickBand="extent", grid=True).properties(
    #     height=900, width=900
    # )
    return


@app.cell(hide_code=True)
def _():
    # pos = nx.spring_layout(P, k=0.75, seed=42, iterations=100)
    # nx.draw(P, pos=pos, with_labels=True)
    # plt.show()
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import polars as pl
    import networkx as nx
    from networkx.algorithms import bipartite
    import matplotlib.pyplot as plt
    import cv2
    import base64, io
    from PIL import Image
    import numpy as np
    import altair as alt
    return alt, mo, pl


@app.cell(hide_code=True)
def _():
    return


@app.cell
def _(pl):
    sample = pl.read_excel("sample_review_updated_latex_bigletter_dedup.xlsx")
    # filter out questionable matches
    # uncomment to filter out body type
    # sample = sample.filter(pl.col("filename").str.contains("chunk"))
    sample
    return (sample,)


@app.cell(hide_code=True)
def _(pl):
    gathering_map = pl.read_excel("forty_sermons_map_w_printer.xlsx")
    gathering_map
    return (gathering_map,)


@app.cell(hide_code=True)
def _(gathering_map, pl, sample):
    signature_enum = pl.Enum(gathering_map["gathering"].to_list())
    full_sample = (
        sample
        .filter(pl.col("filename").str.contains("fortysermons"))
        .sort("page_number")
        .join_asof(
            gathering_map,
            left_on="page_number",
            right_on="first_page",
            strategy="backward",
        )
        .select(
            "root_image",
            "page_number",
            "filename",
            "image",
            "is_root",
            "gathering",
            "letter",
            "letter_class",
            "Font",
        )
        .with_columns(
            gathering=pl.col("gathering").cast(signature_enum),
            letter_with_class=pl.concat_str(
                [pl.col("letter"), pl.col("letter_class")]
            ),
        )
    )
    full_sample.sort("letter_with_class")  # .filter(pl.col("gathering") > "FF")
    return (full_sample,)


@app.cell(hide_code=True)
def _():
    # Create matrix visualization of gatherings

    # G = nx.from_pandas_edgelist(
    #     full_sample, source="gathering", target="root_image"
    # )
    # print(G)
    return


@app.cell
def _():
    # Get node sets for later:
    # top_nodes = full_sample["gathering"].unique().to_list()
    return


@app.cell
def _():
    # G.remove_edges_from(nx.selfloop_edges(G))
    # P = bipartite.weighted_projected_graph(G, top_nodes)
    # print(P)
    return


@app.cell
def _():
    # edgelist = (
    #     pl.DataFrame(nx.to_pandas_edgelist(P))
    #     .with_columns(
    #         source=pl.col("source").cast(signature_enum),
    #         target=pl.col("target").cast(signature_enum),
    #     )
    #     .sort("source", "target")
    # )
    # edgelist
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
