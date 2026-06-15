import dash
from dash import dcc, html, Input, Output
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import argparse
import datetime
import json
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
import time
import base64
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eig
from sklearn.cluster import SpectralClustering
import re



def extract_book(path):
    try:
        book = Path(path).name.split("_")[4].split("-")[0]
        if '.jpg' not in book:
            return book
        else:
            return 'None'
    except IndexError:
        return None

def extract_year(path):
    try:
        return int(extract_book(path)[-4:])
    except:
        return None

def extract_estc(path):
    try:
        return Path(path).name.split("_")[1]
    except IndexError:
        return None

def extract_printer(path):
    try:
        printer = Path(path).name.split("_")[0]
        if '.jpg' not in printer:
            return printer
        else:
            return 'None'
    except IndexError:
        return None

def extract_char(path):
    try:
        char = Path(path).parent.name if 'chunk' in path else Path(path).name.split("_")[7]
        if char.upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            return char
        else:
            return None
    except IndexError:
        return None

def extract_page(path):
    try:
        return int(Path(path).name.split("-")[1].split("_")[0])
    except:
        return None

def extract_chunk(path):
    try:
        return int(Path(path).name.split("_")[-1].split(".")[0])
    except:
        return None


def read_filelist_into_df(image_filelist):
    nodes_df = pd.DataFrame()
    nodes_df["path"] = pd.read_csv(image_filelist, header=None)[0].values
    nodes_df["printer"] = nodes_df["path"].apply(lambda x: extract_printer(x))
    nodes_df["book"] = nodes_df["path"].apply(lambda x: extract_book(x))
    nodes_df["year"] = nodes_df["path"].apply(lambda x: extract_year(x))
    nodes_df["estc"] = nodes_df["path"].apply(lambda x: extract_estc(x))
    nodes_df["char"] = nodes_df["path"].apply(lambda x: extract_char(x))
    nodes_df["page"] = nodes_df["path"].apply(lambda x: extract_page(x))
    nodes_df["chunk"] = nodes_df["path"].apply(lambda x: extract_chunk(x))
    return nodes_df


# Load JSON data
def load_graph_data(sim_mat, image_filelist, damage_csv, min_sim=0.0, char='C'):
    # NOTE: image_filelist and sim_mat_npy must be the same size and in same order
    # Load nodes data into df
    nodes_df = read_filelist_into_df(image_filelist)
    print(f"Nodes df shape: {nodes_df.shape}")

    damage_df = pd.read_csv(damage_csv)
    print(f"Damage df shape: {damage_df.shape}")
    damage_df = damage_df.set_index("path")["damage"]

    # create global numbering that's consistent with the sim_mat
    nodes_df["node_id"] = range(sim_mat.shape[0])
    # create separate graphs grouped by character
    # for char in nodes_df["char"].unique():
    # TODO:
    nodes_df["char"] = nodes_df["char"].str.upper()
    char_nodes_df = nodes_df[nodes_df["char"] == char]
    char_sim_mat = sim_mat[char_nodes_df.index][:, char_nodes_df.index]
    print(f"Character: {char}")

    # Filter out self-similarity
    np.fill_diagonal(char_sim_mat, 0)
    # Find nodes (rows) that have at least one edge > threshold
    # valid_nodes = np.any(sim_mat >= min_sim, axis=1)
    # Filter the similarity matrix to keep only valid nodes
    # sim_mat = sim_mat[valid_nodes][:, valid_nodes]
    # print(f"Filtered similarity matrix shape: {sim_mat.shape}")
    # Filter out edges with similarity less than min_sim
    print(f"Filtering out edges with similarity less than {min_sim}")
    # sim_mat[char_sim_mat < min_sim] = 0
    char_sim_mat[char_sim_mat < min_sim] = 0
    # Filter out rows/edges entirely if all values are 0
    # sim_mat = sim_mat[~np.all(sim_mat == 0, axis=1)]
    # sim_mat = sim_mat[:, ~np.all(sim_mat == 0, axis=0)]
    # convert to sparse csr matrix
    # sim_mat = csr_matrix(sim_mat.astype(np.float32))
    # find mean of nonzero entries
    if np.count_nonzero(char_sim_mat) == 0:
        print(f"Error: Character {char} now has no edges whatsoever!")
        exit(1)
    print(f"Mean of nonzero entries: {np.mean(char_sim_mat[char_sim_mat > 0])}")

    # Option 1: Build a Laplacian matrix based on a k-neighbors graph (n_neighbors=5)
    n_neighbors = 10
    knn_graph = kneighbors_graph(char_sim_mat, n_neighbors=n_neighbors, mode='connectivity', include_self=False)
    L = laplacian(knn_graph, normed=True)
    print(f"Constructed normalized Laplacian matrix using {n_neighbors} neighbors.")

    # Option 2: Perform spectral clustering using the eigengap heuristic on the Laplacian
    # Compute eigenvalues of the Laplacian (convert to dense array)
    L_dense = L.toarray()
    eigenvalues = np.real(eig(L_dense, left=False, right=False))
    # Sort the eigenvalues in ascending order
    sorted_eigenvalues = np.sort(eigenvalues)
    # Compute differences between consecutive eigenvalues (eigengaps)
    eigengaps = np.diff(sorted_eigenvalues)
    # Heuristic: choose number of clusters as index with maximum gap + 1, with at least 2 clusters
    # optimal_clusters = max(2, np.argmax(eigengaps[:10]) + 1)
    # print(f"Optimal number of clusters determined by eigengap heuristic: {optimal_clusters}")

    # import ipdb; ipdb.set_trace()

    clusters = 150

    # spectral = SpectralClustering(n_clusters=optimal_clusters, affinity='precomputed', random_state=42)
    spectral = SpectralClustering(
        n_clusters=clusters, 
        affinity='precomputed', 
        random_state=42,
        assign_labels='kmeans',
    )
    cluster_labels = spectral.fit_predict(char_sim_mat)
    print("Spectral clustering complete. Cluster labels assigned to nodes.")

    breakpoint()

    # Build a new similarity matrix that retains only intra-cluster edges.
    for i in range(len(cluster_labels)):
        for j in range(len(cluster_labels)):
            if cluster_labels[i] != cluster_labels[j]:
                char_sim_mat[i, j] = 0

    print("Filtered similarity matrix to include only intra-cluster edges based on spectral clustering.")

    # Build graph G from the filtered intra-cluster similarity matrix.
    G = nx.from_numpy_array(char_sim_mat)

    # Annotate each node with its corresponding spectral cluster label.
    for i, node_id in enumerate(char_nodes_df["node_id"]):
        G.nodes[i]["cluster"] = int(cluster_labels[i])

    print(f"Num nodes: {char_sim_mat.shape[0]}")
    print(f"Num edges: {np.count_nonzero(char_sim_mat) / 2}")
    # print average number of edges per node
    print(f"Avg num edges per node: {np.count_nonzero(sim_mat) / char_sim_mat.shape[0]}")
    # print number of rows with at least two nonzero entry
    print(f"Num nodes with at least two edges: {np.sum(np.sum(char_sim_mat > 0, axis=1) >= 2)}")
    # at least 3
    print(f"Num nodes with at least three edges: {np.sum(np.sum(char_sim_mat > 0, axis=1) >= 3)}")

    
    print(f"{char}: Num nodes: {char_sim_mat.shape[0]}")
    print(f"{char}: Num edges: {np.count_nonzero(char_sim_mat) / 2}")
    print(f"{char}: Avg num edges per node: {np.count_nonzero(char_sim_mat) / char_sim_mat.shape[0]}")
    print(f"{char}: Num nodes with at least two edges: {np.sum(np.sum(char_sim_mat > 0, axis=1) >= 2)}")
    print(f"{char}: Num nodes with at least three edges: {np.sum(np.sum(char_sim_mat > 0, axis=1) >= 3)}")
    print(f"{char}: Mean of nonzero entries: {np.mean(char_sim_mat[char_sim_mat > 0])}")

    start_time = time.time()
    G = nx.from_numpy_array(char_sim_mat)
    # add fields to nodes
    for i, node_id in enumerate(char_nodes_df["node_id"]):
        G.nodes[i]["path"] = char_nodes_df.loc[node_id]["path"]
        G.nodes[i]["printer"] = char_nodes_df.loc[node_id]["printer"]
        G.nodes[i]["year"] = char_nodes_df.loc[node_id]["year"]
        G.nodes[i]["book"] = char_nodes_df.loc[node_id]["book"]
        G.nodes[i]["char"] = char_nodes_df.loc[node_id]["char"]
        # G.nodes[i]["damage"] = float(damage_df[char_nodes_df.loc[node_id]["path"].replace('.jpg', '.tif')])
        G.nodes[i]["damage"] = float(damage_df[char_nodes_df.loc[node_id]["path"]])
    print(f"{char}: created graph in {time.time() - start_time}s.")
    # find cliques in graph
    # cliques = list(nx.find_cliques(G))
    # print(f"{char}: Num cliques: {len(cliques)}")
    # map cliques back to original node ids
    # cliques = [
    #     [char_nodes_df.iloc[node_id]["node_id"] for node_id in clique] 
    #     for clique in cliques
    # ]
    # TODO: map char ids to global ids in df
    # import ipdb; ipdb.set_trace()

    # remove isolates from graph
    G.remove_nodes_from(list(nx.isolates(G)))

    edges = nx.to_pandas_edgelist(G)
    edges_df = edges.rename(columns={"source": "source", "target": "target", "weight": "weight"})
    return nodes_df, edges_df, G


# Function to encode images as Base64 for display
def encode_image(image_path):
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        # return None
        raise FileNotFoundError(f"Image not found: {image_path}")
    with open(image_path, 'rb') as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/tif;base64,{encoded}"


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Load graph data from a sim mat.")
parser.add_argument("sim_mat", type=str, help="Path to the numpy file containing sim_mat.")
parser.add_argument("image_filelist", type=str, help="Path to the file containing the list of images.")
parser.add_argument("--damage_csv", type=str, default="/graft2/code/nvog/git/matching/damage_classifier_output/damage_scores.csv", help="Path to the file containing the damage csv.")
args = parser.parse_args()

#
#
#
#
#
#   Load global variables like graph
#
#
#
#
#
# Load data and create graph
save = True
if save:
    # Load similarity matrix
    start_time = time.time()
    sim_mat = np.load(args.sim_mat)
    print(f"Similarity matrix shape: {sim_mat.shape} (load time: {time.time() - start_time})")
    # for c in 'abcdefghijklmnopqrstuvwxyz':
    match = re.search(r'-([A-Z])\.npy$', args.sim_mat)
    if match:
        c = match.group(1)
        print(c)
    else:
        print("No uppercase character found. Setting c to None.")
        c = None
        # exit(1)
    print(f"Loading data for character {c}")
    nodes_df, edges_df, graph = load_graph_data(sim_mat, args.image_filelist, args.damage_csv, char=c, min_sim=0.0)
    # Save data to disk
    print("Saving data to disk...")
    nodes_df.to_pickle(f"nodes_df_{Path(args.sim_mat).name.replace('.npy', '')}.pkl")
    edges_df.to_pickle(f"edges_df_{Path(args.sim_mat).name.replace('.npy', '')}.pkl")
    pickle.dump(graph, open(f"graph_{Path(args.sim_mat).name.replace('.npy', '')}.pkl", 'wb'))
    print("Data saved to disk.")
else:
    print("Reading data from disk...")
    nodes_df = pickle.load(open(f"nodes_df_{Path(args.sim_mat).name.replace('.npy', '')}.pkl", 'rb'))
    edges_df = pickle.load(open(f"edges_df_{Path(args.sim_mat).name.replace('.npy', '')}.pkl", 'rb'))
    graph = pickle.load(open(f"graph_{Path(args.sim_mat).name.replace('.npy', '')}.pkl", 'rb'))
    print("Data read from disk.")
    print(f"Graph has {len(graph.nodes)} nodes and {len(graph.edges)} edges.")

# Get unique printer names for dropdown options
printer_names = sorted(nodes_df["printer"].unique())
book_names = sorted(b if b is not None else 'None' for b in nodes_df["book"].unique())

# import ipdb; ipdb.set_trace()

def remove_large_components(G, max_size=5):
    """
    Removes connected components with more than `max_size` nodes from the graph.

    Args:
        G (nx.Graph): The input graph (undirected).
        max_size (int): The maximum size of a connected component to keep.

    Returns:
        nx.Graph: The graph with large connected components removed.
    """
    # Find all connected components
    # start_time = time.time()
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    # print(f"Number of connected components: {len(components)}")
    # print(f"Size of largest component: {len(components[0])}")
    # print(f"Took {time.time() - start_time:0.2f}s to find connected components.")

    # plot and save histogram of connected component sizes
    plt.hist([len(comp) for comp in components], bins=50)
    plt.xlabel("Component size")
    plt.ylabel("Frequency")
    plt.title("Histogram of connected component sizes")
    plt.savefig("component_sizes.png")
    plt.close()


    # Identify components larger than the max size
    # large_components = [comp for comp in components if len(comp) > max_size]

    # Remove nodes in large components
    # for comp in large_components:
    #     G.remove_nodes_from(comp)

    return G


def print_degree_info(G):
    degrees = dict(G.degree())
    print(f"Median degree: {np.median(list(degrees.values()))}")
    print(f"Mean degree: {np.mean(list(degrees.values()))}")
    print(f"Std degree: {np.std(list(degrees.values()))}")
    print(f"Max degree: {np.max(list(degrees.values()))}")
    print(f"Min degree: {np.min(list(degrees.values()))}")


def generate_graph_figure(original_graph, damage_threshold=0.4, weight_threshold=0.7, degree_filter=10, printer_filter=None, book_filter=None, year_range=None):
    print(f"Generating graph with weight threshold={weight_threshold}, degree filter={degree_filter}, printer filter={printer_filter}, book filter={book_filter}, year range={year_range}")
    
    start_time = time.time()
    # Start with all nodes
    filtered_nodes = set(original_graph.nodes)
    
    # Filter by printer
    if printer_filter:
        filtered_nodes = {n for n in filtered_nodes if original_graph.nodes[n]["printer"] in printer_filter}
    print(f"Number of nodes after filtering by printer {printer_filter}: {len(filtered_nodes)}")
    
    if not filtered_nodes:
        return go.Figure(), f"No nodes to display after applying printer filter {printer_filter}. Try adjusting the filters."

    # Filter by book
    if book_filter:
        filtered_nodes = {n for n in filtered_nodes if original_graph.nodes[n]["book"] in book_filter}
    print(f"Number of nodes after filtering by book {book_filter}: {len(filtered_nodes)}")

    if not filtered_nodes:
        return go.Figure(), f"No nodes to display after applying book filter {book_filter}. Try adjusting the filters."

    # Filter by year range
    if year_range:
        start_year, end_year = year_range
        filtered_nodes = {n for n in filtered_nodes if start_year <= original_graph.nodes[n]["year"] <= end_year}
    print(f"Number of nodes after filtering by year range {year_range}: {len(filtered_nodes)}")

    if not filtered_nodes:
        return go.Figure(), f"No nodes to display after applying year range filter {year_range}. Try adjusting the filters."

    # Create a subgraph instead of copying
    graph = original_graph.subgraph(filtered_nodes).copy()
    
    # Filter edges by weight
    edges_to_remove = [(u, v) for u, v, d in graph.edges(data=True) if d["weight"] < weight_threshold]
    graph.remove_edges_from(edges_to_remove)
    print(f"Number of edges after filtering by weight: {len(graph.edges)}")

    if not graph.edges:
        return go.Figure(), f"Found {len(graph.nodes)} nodes, but no edges to display after applying weight threshold of {weight_threshold}. Try adjusting the filters."


    # TODO: perform kNN sparsification
    # graph = nx.k_nearest_neighbors(graph, n_neighbors=3, weight="weight")

    # Filter nodes by damage threshold
    # nodes_to_remove = {n for n in filtered_nodes if original_graph.nodes[n]["damage"] < damage_threshold}
    # graph.remove_nodes_from(nodes_to_remove)
    # print(f"Removed {len(nodes_to_remove)} nodes with damage < {damage_threshold}.")
    print(f"Number of nodes after removing low-damage nodes: {len(graph.nodes)}")
    print(f"Number of edges after removing low-damage nodes: {len(graph.edges)}")

    # Remove high-degree nodes
    degrees = dict(graph.degree())
    nodes_to_remove = {n for n, d in degrees.items() if d > degree_filter}
    graph.remove_nodes_from(nodes_to_remove)
    print(f"Removed {len(nodes_to_remove)} nodes with degree > {degree_filter}.")
    print(f"Number of nodes after removing high-degree nodes: {len(graph.nodes)}")
    print(f"Number of edges after removing high-degree nodes: {len(graph.edges)}")

    if not graph.nodes:
        return go.Figure(), f"No nodes found after filtering to remove nodes with degree > {degree_filter}. Try adjusting the filters."

    if not graph.edges:
        return go.Figure(), f"Found {len(graph.nodes)} nodes, but no edges to display after removing high-degree nodes. Try adjusting the filters."

    # Remove isolated nodes
    graph.remove_nodes_from(list(nx.isolates(graph)))
    if not graph.nodes:
        return go.Figure(), "No nodes found after filtering to remove isolates. Try adjusting the filters."

    # Compute layout
    start_time = time.time()
    pos = nx.spring_layout(graph, k=2, seed=42)
    print(f"Positioned nodes in {time.time() - start_time:0.2f}s.")

    # Create node trace
    node_images = [
        encode_image(graph.nodes[n]["path"]) for n in graph.nodes
    ]
    node_trace = go.Scatter(
        x=[pos[n][0] for n in graph.nodes],
        y=[pos[n][1] for n in graph.nodes],
        mode="markers",
        marker=dict(size=50, opacity=0.0),
        hoverinfo="text",
        text=[
            # f"{graph.nodes[n]['printer']} ({int(graph.nodes[n]['year'])}) °{graph.degree(n)}" 
            # f"{graph.nodes[n]['path']}" 
            f"{graph.nodes[n]['printer']} {graph.nodes[n]['book']} (ds={graph.nodes[n]['damage']:0.2f}, °{graph.degree(n)})"
            for n in graph.nodes]
    )

    start_time = time.time()
    # Add images as annotations
    layout_images = []
    # import ipdb; ipdb.set_trace()
    for n, img_src in zip(graph.nodes, node_images):
        # print(f"Adding image annotation for node {n}: {img_src}")
        if img_src:
            x, y = pos[n]
            layout_images.append(
                dict(
                    source=img_src,
                    x=x,
                    y=y,
                    xref="x",
                    yref="y",
                    sizex=0.1,
                    sizey=0.1,
                    xanchor="center",
                    yanchor="middle",
                    layer="above",
                )
            )
    print(f"Added {len(layout_images)} images as annotations in {time.time() - start_time:0.2f}s.")


    # Create edge traces
    edge_traces = [
        go.Scatter(
            x=[pos[u][0], pos[v][0], None],
            y=[pos[u][1], pos[v][1], None],
            text=[f"sim={d['weight']:.2f}", f"sim={d['weight']:.2f}", None],
            hoverinfo="text",
            line=dict(width=4, color="black"),
            mode="lines"
        )
        for u, v, d in graph.edges(data=True)
    ]

    fig = go.Figure()
    for trace in edge_traces:
        fig.add_trace(trace)
    fig.add_trace(node_trace)

    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        images=layout_images
    )

    return fig, f"Found {len(graph.nodes)} nodes, {len(graph.edges)} edges."


# Layout of the Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Matching Graph"),

    # html.Div([
    #     html.Label("Filter by Character:"),
    #     dcc.Dropdown(
    #         id='char-dropdown',
    #         options=[{"label": char, "value": char} for char in chars],
    #         placeholder="Select a character",
    #         clearable=True,
    #         value='c'
    #     )
    # ], style={'margin-top': '20px'}),

    html.Div([
        html.Label("Filter by Damage Score:"),
        dcc.Slider(
            id='damage-slider',
            min=0.0,
            max=1.0,
            step=0.005,
            value=0.3,
            marks={i/10: f'{i/10}' for i in range(11)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        html.Div(id='damage-slider-output', style={'margin-top': '10px'})
    ]),

    
    html.Div([
        html.Label("Filter Edges by Matcher Score:"),
        dcc.Slider(
            id='weight-slider',
            min=-1.0,
            max=1.0,
            step=0.005,
            value=0.9,
            marks={i/10: f'{i/10}' for i in range(11)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        html.Div(id='weight-slider-output', style={'margin-top': '10px'})
    ]),

    html.Div([
        html.Label("Filter out by Max Degree:"),
        dcc.Slider(
            id='degree-slider',
            min=0,
            max=max(dict(graph.degree()).values()),
            step=1,
            value=max(dict(graph.degree()).values()),
            marks={i: str(i) for i in range(min(dict(graph.degree()).values()), max(dict(graph.degree()).values()), (max(dict(graph.degree()).values()) - min(dict(graph.degree()).values()))//10)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        html.Div(id='degree-slider-output', style={'margin-top': '10px'})
    ]),
    
    html.Div([
        html.Label("Filter by Printer:"),
        dcc.Dropdown(
            id='printer-dropdown',
            options=[{"label": printer, "value": printer} for printer in printer_names],
            placeholder="Select a printer",
            clearable=True,
            value=['everingham', 'tbraddyll', 'anon'],
            multi=True
        )
    ], style={'margin-top': '20px'}),

    html.Div([
        html.Label("Filter by Book:"),
        dcc.Dropdown(
            id='book-dropdown',
            options=[{"label": book, "value": book} for book in book_names],
            placeholder="Select a book",
            clearable=True,
            value=[],
            multi=True
        )
    ], style={'margin-top': '20px'}),
    
    html.Div([
        html.Label("Filter by Year Range:"),
        dcc.RangeSlider(
            id='year-slider',
            # min=nodes_df["year"].min(),
            min=int(nodes_df["year"].min()),
            max=min(int(nodes_df["year"].max()), 1710) + 1,
            step=5,
            # value=[nodes_df["year"].min(), nodes_df["year"].max()],
            # value=[1680, 1695],
            # value=[1675, 1695],
            value=[int(nodes_df["year"].min()), min(int(nodes_df["year"].max()), 1710)],
            marks={year: str(year) for year in range(int(nodes_df["year"].min()), min(int(nodes_df["year"].max()), 1710) + 1, 5)},
        )
    ], style={'margin-top': '20px'}),
    html.Div(
            id="notification",
            style={"color": "red", "fontWeight": "bold", "marginBottom": "10px"},
    ),
    dcc.Graph(
        id='graph',
        config={'scrollZoom': True},
        style={"height": "calc(100vh - 200px)"}
    )
])

# Callbacks for interactivity
@app.callback(
    [Output("graph", "figure"),
     Output("notification", "children")],
    [Input('damage-slider', 'value'),
     Input('weight-slider', 'value'),
     Input('degree-slider', 'value'),
     Input('printer-dropdown', 'value'),
     Input('book-dropdown', 'value'),
     Input('year-slider', 'value')]
)
def update_graph(damage_threshold, weight_threshold, degree_filter, printer_filter, book_filter, year_range):
    global graph
    start_time = time.time()
    g = generate_graph_figure(graph, damage_threshold, weight_threshold, degree_filter, printer_filter, book_filter, year_range)
    print(f"Generated graph in {time.time() - start_time:0.2f}s.")
    return g

if __name__ == '__main__':
    app.run_server(debug=False)
