import polars as pl
import altair as alt
import re
from utils import load_and_prepare_sample, SPLIT_RANGES

# all_sample = load_and_prepare_sample()
all_sample = load_and_prepare_sample('LockeSpinozaMatches_processed.csv')
print(all_sample.select(pl.col("book_name").unique()).to_series().to_list())

gathering_map = pl.read_excel("forty_sermons_map_w_printer.xlsx")

# Helper function to create background shading for sections
def create_section_background(gathering_map, split_ranges, book_key="fortysermons1685"):
    """Create background rectangles for section splits"""
    if book_key not in split_ranges:
        return None
    
    # Create a list of section ranges mapped to gatherings
    section_data = []
    for section, page_ranges in split_ranges[book_key].items():
        for start_page, end_page in page_ranges:
            # Find gatherings that fall within this page range
            gatherings_in_range = gathering_map.filter(
                (pl.col("first_page") >= start_page) & (pl.col("first_page") < end_page)
            )["gathering"].unique().to_list()
            
            if gatherings_in_range:
                section_data.append({
                    "section": section,
                    "gatherings": gatherings_in_range,
                    "start_page": start_page,
                    "end_page": end_page
                })
    
    # Create a dataframe with one row per gathering per section
    rows = []
    for item in section_data:
        for gathering in item["gatherings"]:
            rows.append({
                "gathering": gathering,
                "section": f"Part {item['section']}",
            })
    
    if not rows:
        return None
    
    return pl.DataFrame(rows)

# Find letters that appear in non-fortysermons books
non_fortysermons_letters = set(
    all_sample
    .filter(~pl.col("filename").str.contains("sermons"))
    ["letter_with_class"]
    .unique()
    .to_list()
)

print(f"Found {len(non_fortysermons_letters)} letter types in non-fortysermons books")

# Print book names found in the dataset
all_book_names = set(all_sample["book_name"].unique().to_list())
fortysermons_book_names = set(
    all_sample
    .filter(pl.col("filename").str.contains("sermons"))
    ["book_name"]
    .unique()
    .to_list()
)
non_fortysermons_book_names = all_book_names - fortysermons_book_names

print(f"All book names found: {sorted(list(all_book_names))}")
print(f"Fortysermons book names: {sorted(list(fortysermons_book_names))}")
print(f"Non-fortysermons book names: {sorted(list(non_fortysermons_book_names))}")

# Custom sort function for gathering signatures
def gathering_sort_key(gathering_str):
    """
    Custom sort key for gathering signatures.
    Order: pi, a, then single uppercase (B-Z), then double (AA-ZZ), then triple (AAA-III),
           then single+25 (A25-Z25), then double+25 (AA25-KK25)
    
    Examples:
    - Lowercase: pi, a
    - Single uppercase: B, C, D, ..., Z
    - Double uppercase: AA, BB, CC, ..., ZZ
    - Triple uppercase: AAA, BBB, ..., III
    - Single + 25: A25, B25, ..., Z25
    - Double + 25: AA25, BB25, ..., KK25
    """
    gathering_str = str(gathering_str)
    
    # Pattern to match pure lowercase (like "a", "pi")
    if re.match(r'^[a-z]+$', gathering_str):
        # Pure lowercase - category 0
        return (0, gathering_str.lower(), 0)
    
    # Pattern to match uppercase letters with optional numbers
    pattern = r'^([A-Z]+)(\d*)$'
    match = re.match(pattern, gathering_str)
    
    if not match:
        # Fallback for unexpected formats
        return (999, gathering_str, 0)
    
    uppercase_part, num_part = match.groups()
    num_value = int(num_part) if num_part else 0
    
    # Determine category based on structure
    if len(uppercase_part) == 1 and not num_part:
        # Single uppercase letter without number (like "B", "C")
        category = 1
    elif len(uppercase_part) == 2 and not num_part:
        # Double uppercase letters (like "AA", "BB", "DD")
        category = 2
    elif len(uppercase_part) == 3 and not num_part:
        # Triple uppercase letters (like "AAA", "BBB")
        category = 3
    elif len(uppercase_part) == 1 and num_part:
        # Single uppercase letter with number (like "A25", "Z25")
        category = 4
    elif len(uppercase_part) == 2 and num_part:
        # Double uppercase letter with number (like "AA25", "KK25")
        category = 5
    else:
        # Other formats
        category = 6
    
    # Return tuple for sorting: (category, uppercase letters, number)
    return (category, uppercase_part.upper(), num_value)

# Get all gatherings and sort them using custom function
all_gatherings = gathering_map["gathering"].to_list()
all_gatherings_sorted = sorted(all_gatherings, key=gathering_sort_key)

print(f"Gathering order (first 20): {all_gatherings_sorted[:20]}")
print(f"Gathering order (last 20): {all_gatherings_sorted[-20:]}")

# load gathering map and create enum from sorted list
signature_enum = pl.Enum(all_gatherings_sorted)
# order by page number and join asof to get gathering info for each page
full_sample = (
    all_sample
    .filter(pl.col("filename").str.contains("sermons"))
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
        "letter_with_class",
        "book_name"
    )
    .with_columns(
        gathering=pl.col("gathering").cast(signature_enum),
        appears_in_other_books=pl.col("letter_with_class").is_in(list(non_fortysermons_letters))
    )
)
full_sample = full_sample.sort("letter_with_class")

# Create a mapping of letter_with_class to non-fortysermons books where they appear
letter_to_other_books = {}
for letter in full_sample["letter_with_class"].unique().to_list():
    other_books = set(
        all_sample
        .filter(
            (pl.col("letter_with_class") == letter) & 
            (~pl.col("filename").str.contains("sermons"))
        )
        ["book_name"]
        .unique()
        .to_list()
    )
    if other_books:
        # Join multiple books with " + " if letter appears in multiple non-fortysermons books
        letter_to_other_books[letter] = " + ".join(sorted(list(other_books)))
    else:
        letter_to_other_books[letter] = "fortysermons only"

# Add the other_books column to full_sample
full_sample = full_sample.with_columns(
    other_books=pl.col("letter_with_class").map_elements(
        lambda x: letter_to_other_books[x], return_dtype=pl.Utf8
    )
)

# Print statistics about cross-book letter occurrences
fortysermons_letters = set(full_sample["letter_with_class"].unique().to_list())
cross_book_letters = fortysermons_letters.intersection(non_fortysermons_letters)
fortysermons_only_letters = fortysermons_letters - non_fortysermons_letters

print(f"Letters in fortysermons: {len(fortysermons_letters)}")
print(f"Letters that also appear in other books: {len(cross_book_letters)}")
print(f"Letters only in fortysermons: {len(fortysermons_only_letters)}")
print(f"Cross-book letter examples: {sorted(list(cross_book_letters))[:10]}")
print(f"Fortysermons-only letter examples: {sorted(list(fortysermons_only_letters))[:10]}")
print()




# plot letters by gathering
images_l = (
    alt.Chart(full_sample)
    .mark_image()
    .encode(
        y=alt.Y("letter_with_class:O"), 
        x=alt.X("gathering"), 
        url="image",
        stroke=alt.condition(
            alt.datum.appears_in_other_books,
            alt.value("red"),
            alt.value("transparent")
        ),
        strokeWidth=alt.condition(
            alt.datum.appears_in_other_books,
            alt.value(2),
            alt.value(0)
        )
    )
).properties(
    width=900,
    # height=180,
)
images_l.configure_axis(
    grid=True, tickBand="extent"
)

# heatmap of letters by gathering with cross-book highlighting
heatmap = alt.Chart(full_sample).mark_rect(height=5).encode(
    y=alt.Y("letter_with_class:O", title="letter type"),
    x=alt.X("gathering"),
    color=alt.condition(
        alt.datum.appears_in_other_books,
        alt.Color("other_books:N", legend=alt.Legend(
            title="Also appears in books",
            labelLimit=300,
            symbolLimit=50,
            labelExpr="split(datum.label, ' + ')"
        )),
        alt.value("lightgray")
    ),
    opacity=alt.condition(
        alt.datum.appears_in_other_books,
        alt.value(0.8),
        alt.value(0.3)
    ),
    stroke=alt.condition(
        alt.datum.appears_in_other_books,
        alt.value("red"),
        alt.value("transparent")
    ),
    strokeWidth=alt.condition(
        alt.datum.appears_in_other_books,
        alt.value(1),
        alt.value(0)
    ),
    tooltip=["letter_with_class", "gathering", "other_books", "appears_in_other_books"]
).properties(
    height=alt.Step(10),
    width=800
).resolve_scale(
    color='independent'
)


# Iterative clustering of gatherings based on shared letters
print("="*60)
print("ITERATIVE CLUSTERING OF GATHERINGS")
print("="*60)

# Get unique gatherings and create a mapping of gathering -> letters
unique_gatherings = full_sample["gathering"].unique().sort().to_list()
gathering_to_letters = {}

for gathering in unique_gatherings:
    letters = set(full_sample.filter(pl.col("gathering") == gathering)["letter_with_class"].unique().to_list())
    gathering_to_letters[gathering] = letters

print(f"Total gatherings to cluster: {len(unique_gatherings)}")
print("Building clusters iteratively...\n")

# Track which gatherings have been assigned to clusters
assigned_gatherings = set()
clusters = []

def find_connected_gatherings(start_gathering, current_cluster):
    """Iteratively find all gatherings connected to the start gathering through shared letters"""
    to_process = [start_gathering]
    processed = set()
    
    while to_process:
        current_gathering = to_process.pop(0)
        
        if current_gathering in processed:
            continue
            
        processed.add(current_gathering)
        current_cluster.add(current_gathering)
        assigned_gatherings.add(current_gathering)
        
        # Get letters for current gathering
        current_letters = gathering_to_letters[current_gathering]
        
        # Find all gatherings that share any letter with current gathering
        for other_gathering in unique_gatherings:
            if other_gathering not in processed and other_gathering not in assigned_gatherings:
                other_letters = gathering_to_letters[other_gathering]
                shared_letters = current_letters.intersection(other_letters)
                
                if shared_letters:
                    print(f"  Adding {other_gathering} to cluster (shares {sorted(list(shared_letters))} with {current_gathering})")
                    to_process.append(other_gathering)

# Build clusters iteratively
cluster_id = 1
for gathering in unique_gatherings:
    if gathering not in assigned_gatherings:
        print(f"\nStarting Cluster {cluster_id} with gathering: {gathering}")
        
        current_cluster = set()
        find_connected_gatherings(gathering, current_cluster)
        
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
            print(f"  Cluster {cluster_id} completed with {len(current_cluster)} gatherings: {sorted([str(g) for g in current_cluster])}")
        else:
            print(f"  {gathering} is isolated (no shared letters with other gatherings)")
            
        cluster_id += 1

print("\n" + "="*60)
print("CLUSTERING RESULTS")
print("="*60)

if clusters:
    print(f"Found {len(clusters)} clusters with multiple gatherings:")
    
    for i, cluster in enumerate(clusters, 1):
        cluster_list = sorted([str(g) for g in cluster])
        print(f"\nCluster {i} ({len(cluster)} gatherings):")
        print(f"  Gatherings: {', '.join(cluster_list)}")
        
        # Find all letters that appear in multiple gatherings within this cluster
        all_shared_pairs = []
        cluster_gatherings = list(cluster)
        
        for j in range(len(cluster_gatherings)):
            for k in range(j + 1, len(cluster_gatherings)):
                g1, g2 = cluster_gatherings[j], cluster_gatherings[k]
                shared = gathering_to_letters[g1].intersection(gathering_to_letters[g2])
                if shared:
                    all_shared_pairs.append(f"{g1}↔{g2}: {sorted(list(shared))}")
        
        if all_shared_pairs:
            print("  Shared letter pairs (sample of first 5):")
            for pair in all_shared_pairs[:5]:
                print(f"    {pair}")
            if len(all_shared_pairs) > 5:
                print(f"    ... and {len(all_shared_pairs) - 5} more pairs")
else:
    print("No clusters with multiple gatherings found.")

# Count isolated gatherings
isolated = []
for gathering in unique_gatherings:
    if not any(gathering in cluster for cluster in clusters):
        isolated.append(gathering)

if isolated:
    print(f"\nIsolated gatherings ({len(isolated)}): {sorted([str(g) for g in isolated])}")

print("="*60)

# Clustering without 'chunk' files
print("\n" + "="*60)
print("ITERATIVE CLUSTERING OF GATHERINGS (EXCLUDING 'CHUNK' FILES)")
print("="*60)

# Filter out rows where filename contains 'chunk'
full_sample_no_chunks = full_sample.filter(~pl.col("filename").str.contains("chunk"))

print(f"Original sample size: {len(full_sample)}")
print(f"After filtering out 'chunk' files: {len(full_sample_no_chunks)}")

# Get unique gatherings and create a mapping of gathering -> letters (without chunks)
unique_gatherings_no_chunks = full_sample_no_chunks["gathering"].unique().sort().to_list()
gathering_to_letters_no_chunks = {}

for gathering in unique_gatherings_no_chunks:
    letters = set(full_sample_no_chunks.filter(pl.col("gathering") == gathering)["letter_with_class"].unique().to_list())
    gathering_to_letters_no_chunks[gathering] = letters

print(f"Total gatherings to cluster (no chunks): {len(unique_gatherings_no_chunks)}")
print("Building clusters iteratively (excluding chunk files)...\n")

# Track which gatherings have been assigned to clusters
assigned_gatherings_no_chunks = set()
clusters_no_chunks = []

def find_connected_gatherings_no_chunks(start_gathering, current_cluster):
    """Iteratively find all gatherings connected to the start gathering through shared letters (no chunks)"""
    to_process = [start_gathering]
    processed = set()
    
    while to_process:
        current_gathering = to_process.pop(0)
        
        if current_gathering in processed:
            continue
            
        processed.add(current_gathering)
        current_cluster.add(current_gathering)
        assigned_gatherings_no_chunks.add(current_gathering)
        
        # Get letters for current gathering
        current_letters = gathering_to_letters_no_chunks[current_gathering]
        
        # Find all gatherings that share any letter with current gathering
        for other_gathering in unique_gatherings_no_chunks:
            if other_gathering not in processed and other_gathering not in assigned_gatherings_no_chunks:
                other_letters = gathering_to_letters_no_chunks[other_gathering]
                shared_letters = current_letters.intersection(other_letters)
                
                if shared_letters:
                    print(f"  Adding {other_gathering} to cluster (shares {sorted(list(shared_letters))} with {current_gathering})")
                    to_process.append(other_gathering)

# Build clusters iteratively (no chunks)
cluster_id = 1
for gathering in unique_gatherings_no_chunks:
    if gathering not in assigned_gatherings_no_chunks:
        print(f"\nStarting Cluster {cluster_id} with gathering: {gathering}")
        
        current_cluster = set()
        find_connected_gatherings_no_chunks(gathering, current_cluster)
        
        if len(current_cluster) > 1:
            clusters_no_chunks.append(current_cluster)
            print(f"  Cluster {cluster_id} completed with {len(current_cluster)} gatherings: {sorted([str(g) for g in current_cluster])}")
        else:
            print(f"  {gathering} is isolated (no shared letters with other gatherings)")
            
        cluster_id += 1

print("\n" + "="*60)
print("CLUSTERING RESULTS (NO CHUNKS)")
print("="*60)

if clusters_no_chunks:
    print(f"Found {len(clusters_no_chunks)} clusters with multiple gatherings:")
    
    for i, cluster in enumerate(clusters_no_chunks, 1):
        cluster_list = sorted([str(g) for g in cluster])
        print(f"\nCluster {i} ({len(cluster)} gatherings):")
        print(f"  Gatherings: {', '.join(cluster_list)}")
        
        # Find all letters that appear in multiple gatherings within this cluster
        all_shared_pairs_no_chunks = []
        cluster_gatherings = list(cluster)
        
        for j in range(len(cluster_gatherings)):
            for k in range(j + 1, len(cluster_gatherings)):
                g1, g2 = cluster_gatherings[j], cluster_gatherings[k]
                shared = gathering_to_letters_no_chunks[g1].intersection(gathering_to_letters_no_chunks[g2])
                if shared:
                    all_shared_pairs_no_chunks.append(f"{g1}↔{g2}: {sorted(list(shared))}")
        
        if all_shared_pairs_no_chunks:
            print("  Shared letter pairs (sample of first 5):")
            for pair in all_shared_pairs_no_chunks[:5]:
                print(f"    {pair}")
            if len(all_shared_pairs_no_chunks) > 5:
                print(f"    ... and {len(all_shared_pairs_no_chunks) - 5} more pairs")
else:
    print("No clusters with multiple gatherings found.")

# Count isolated gatherings (no chunks)
isolated_no_chunks = []
for gathering in unique_gatherings_no_chunks:
    if not any(gathering in cluster for cluster in clusters_no_chunks):
        isolated_no_chunks.append(gathering)

if isolated_no_chunks:
    print(f"\nIsolated gatherings ({len(isolated_no_chunks)}): {sorted([str(g) for g in isolated_no_chunks])}")

# Compare results
print("\nComparison:")
print("  With chunks: 1 cluster with 60 gatherings, 2 isolated")
print(f"  Without chunks: {len(clusters_no_chunks)} clusters, {len(isolated_no_chunks)} isolated")

# Show what letters were removed by filtering chunks
chunk_letters = set(full_sample.filter(pl.col("filename").str.contains("chunk"))["letter_with_class"].unique().to_list())
no_chunk_letters = set(full_sample_no_chunks["letter_with_class"].unique().to_list())
removed_letters = chunk_letters - no_chunk_letters

if removed_letters:
    print(f"\nLetters only found in 'chunk' files (removed): {sorted(list(removed_letters))}")
else:
    print("\nAll chunk letters also appear in non-chunk files")

print("="*60)

# Create mapping from gathering to cluster ID
gathering_to_cluster = {}
for i, cluster in enumerate(clusters, 1):
    for gathering in cluster:
        gathering_to_cluster[gathering] = f"Cluster {i}"
for gathering in isolated:
    gathering_to_cluster[gathering] = "Isolated"

# Same for no-chunks clustering
gathering_to_cluster_no_chunks = {}
for i, cluster in enumerate(clusters_no_chunks, 1):
    for gathering in cluster:
        gathering_to_cluster_no_chunks[gathering] = f"Cluster {i}"
for gathering in isolated_no_chunks:
    gathering_to_cluster_no_chunks[gathering] = "Isolated"

# Add cluster information to full_sample
full_sample_with_clusters = full_sample.with_columns(
    cluster=pl.col("gathering").map_elements(
        lambda g: gathering_to_cluster.get(g, "Unknown"), 
        return_dtype=pl.Utf8
    )
)

# No-chunks version
full_sample_no_chunks_with_clusters = full_sample.filter(
    ~pl.col("filename").str.contains("chunk")
).with_columns(
    cluster=pl.col("gathering").map_elements(
        lambda g: gathering_to_cluster_no_chunks.get(g, "Unknown"), 
        return_dtype=pl.Utf8
    )
)

# Create the original heatmaps (preserved from original code)
# Original version - all cross-book letters
cross_book_heatmap = alt.Chart(full_sample.filter(pl.col("appears_in_other_books"))).mark_rect(height=8).encode(
    y=alt.Y("letter_with_class:O", title="Letters also in other books"),
    x=alt.X("gathering"),
    color=alt.Color("other_books:N", legend=alt.Legend(
        title="Also appears in books",
        labelLimit=300,
        symbolLimit=50,
        labelExpr="split(datum.label, ' + ')"
    )),
    opacity=alt.value(0.9),
    tooltip=["letter_with_class", "gathering", "other_books"]
).properties(
    title="Letters that appear in both fortysermons and other books",
    height=alt.Step(12),
    width=800
).resolve_scale(
    color='independent'
)
cross_book_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# Create section background data for shading
section_background = create_section_background(gathering_map, SPLIT_RANGES, "fortysermons1685")
print(f"Created section background data with {len(section_background) if section_background is not None else 0} rows")

# Get the original gathering order using custom sort to preserve in all plots
original_gathering_order = all_gatherings_sorted

# Version with section background shading
if section_background is not None:
    # Create background layer - must match main layer encoding exactly to preserve order
    background_layer = alt.Chart(section_background).mark_rect().encode(
        x=alt.X("gathering", sort=original_gathering_order),
        color=alt.Color("section:N", 
            scale=alt.Scale(scheme="pastel1"),
            legend=alt.Legend(title="Part")
        ),
        opacity=alt.value(0.3)
    )
    
    # Create the main heatmap layer
    heatmap_layer = alt.Chart(full_sample.filter(pl.col("appears_in_other_books"))).mark_rect(height=8).encode(
        y=alt.Y("letter_with_class:O", title="Letters also in other books"),
        x=alt.X("gathering", sort=original_gathering_order),
        color=alt.Color("other_books:N", legend=alt.Legend(
            title="Also appears in books",
            labelLimit=300,
            symbolLimit=50,
            labelExpr="split(datum.label, ' + ')"
        )),
        opacity=alt.value(0.9),
        tooltip=["letter_with_class", "gathering", "other_books"]
    )
    
    # Layer them together
    cross_book_heatmap_shaded = alt.layer(
        background_layer,
        heatmap_layer
    ).properties(
        title="Letters that appear in both fortysermons and other books (with section shading)",
        height=alt.Step(12),
        width=800
    ).resolve_scale(
        color='independent'
    ).configure_axis(
        grid=True, tickBand="extent"
    )
else:
    cross_book_heatmap_shaded = cross_book_heatmap

# Version with only chunk letters
chunk_cross_book_heatmap = alt.Chart(
    full_sample.filter(
        (pl.col("appears_in_other_books")) & 
        (pl.col("filename").str.contains("chunk"))
    )
).mark_rect(height=8).encode(
    y=alt.Y("letter_with_class:O", title="Letters from chunk files also in other books"),
    x=alt.X("gathering"),
    color=alt.Color("other_books:N", legend=alt.Legend(
        title="Also appears in books",
        labelLimit=300,
        symbolLimit=50,
        labelExpr="split(datum.label, ' + ')"
    )),
    opacity=alt.value(0.9),
    tooltip=["letter_with_class", "gathering", "other_books"]
).properties(
    title="Letters from chunk files that appear in both fortysermons and other books",
    height=alt.Step(12),
    width=800
).resolve_scale(
    color='independent'
)
chunk_cross_book_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# Version without any chunk letters
no_chunk_cross_book_heatmap = alt.Chart(
    full_sample.filter(
        (pl.col("appears_in_other_books")) & 
        (~pl.col("filename").str.contains("chunk"))
    )
).mark_rect(height=8).encode(
    y=alt.Y("letter_with_class:O", title="Letters from non-chunk files also in other books"),
    x=alt.X("gathering"),
    color=alt.Color("other_books:N", legend=alt.Legend(
        title="Also appears in books",
        labelLimit=300,
        symbolLimit=50,
        labelExpr="split(datum.label, ' + ')"
    )),
    opacity=alt.value(0.9),
    tooltip=["letter_with_class", "gathering", "other_books"]
).properties(
    title="Letters from non-chunk files that appear in both fortysermons and other books",
    height=alt.Step(12),
    width=800
).resolve_scale(
    color='independent'
)
no_chunk_cross_book_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# Create new versions using cluster information
# All data with clusters
clustered_cross_book_heatmap = alt.Chart(
    full_sample_with_clusters.filter(pl.col("appears_in_other_books"))
).mark_rect(height=8).encode(
    y=alt.Y("letter_with_class:O", title="Letters also in other books"),
    x=alt.X("gathering"),
    color=alt.Color("other_books:N", legend=alt.Legend(
        title="Also appears in books",
        labelLimit=300,
        symbolLimit=50,
        labelExpr="split(datum.label, ' + ')"
    )),
    facet=alt.Facet("cluster:N", columns=2),
    opacity=alt.value(0.9),
    tooltip=["letter_with_class", "gathering", "other_books", "cluster"]
).properties(
    title="Letters that appear in both fortysermons and other books (by cluster)",
    height=alt.Step(12),
    width=400
).resolve_scale(
    color='independent'
)
clustered_cross_book_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# Only chunk data with clusters
clustered_chunk_cross_book_heatmap = alt.Chart(
    full_sample_with_clusters.filter(
        (pl.col("appears_in_other_books")) & 
        (pl.col("filename").str.contains("chunk"))
    )
).mark_rect(height=8).encode(
    y=alt.Y("letter_with_class:O", title="Letters from chunk files also in other books"),
    x=alt.X("gathering"),
    color=alt.Color("other_books:N", legend=alt.Legend(
        title="Also appears in books",
        labelLimit=300,
        symbolLimit=50,
        labelExpr="split(datum.label, ' + ')"
    )),
    facet=alt.Facet("cluster:N", columns=2),
    opacity=alt.value(0.9),
    tooltip=["letter_with_class", "gathering", "other_books", "cluster"]
).properties(
    title="Letters from chunk files that appear in both fortysermons and other books (by cluster)",
    height=alt.Step(12),
    width=400
).resolve_scale(
    color='independent'
)
clustered_chunk_cross_book_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# No chunk data with no-chunks clustering
clustered_no_chunk_cross_book_heatmap = alt.Chart(
    full_sample_no_chunks_with_clusters.filter(pl.col("appears_in_other_books"))
).mark_rect(height=8).encode(
    y=alt.Y("letter_with_class:O", title="Letters from non-chunk files also in other books"),
    x=alt.X("gathering"),
    color=alt.Color("other_books:N", legend=alt.Legend(
        title="Also appears in books",
        labelLimit=300,
        symbolLimit=50,
        labelExpr="split(datum.label, ' + ')"
    )),
    facet=alt.Facet("cluster:N", columns=2),
    opacity=alt.value(0.9),
    tooltip=["letter_with_class", "gathering", "other_books", "cluster"]
).properties(
    title="Letters from non-chunk files that appear in both fortysermons and other books (by cluster)",
    height=alt.Step(12),
    width=400
).resolve_scale(
    color='independent'
)
clustered_no_chunk_cross_book_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# Create aggregated versions with other_books on y-axis
print("="*60)
print("CREATING AGGREGATED OTHER_BOOKS Y-AXIS VERSIONS")
print("="*60)

# Create expanded data where each letter occurrence is duplicated for each book it appears in
def expand_other_books_data(df):
    """Expand the data so each book gets its own row instead of being joined with +"""
    expanded_rows = []
    
    for row in df.iter_rows(named=True):
        if row["appears_in_other_books"]:
            other_books = row["other_books"]
            if other_books != "fortysermons only":
                # Split the combined book names and create a row for each book
                individual_books = [book.strip() for book in other_books.split(" + ")]
                for book in individual_books:
                    new_row = row.copy()
                    # Merge A_treatise with spinozatheologicalpoliticalREDO1689
                    if book.lower() == "a_treatise":
                        book = "spinozatheologicalpoliticalREDO1689"
                    if book.lower() == "two_treatises":
                        book = "twotreatisesofgov1690"
                    new_row["individual_book"] = book
                    expanded_rows.append(new_row)
    
    return pl.DataFrame(expanded_rows)

# Create expanded datasets for each filter condition
expanded_all = expand_other_books_data(full_sample.filter(pl.col("appears_in_other_books")))
expanded_chunk = expand_other_books_data(
    full_sample.filter(
        (pl.col("appears_in_other_books")) & 
        (pl.col("filename").str.contains("chunk"))
    )
)
expanded_no_chunk = expand_other_books_data(
    full_sample.filter(
        (pl.col("appears_in_other_books")) & 
        (~pl.col("filename").str.contains("chunk"))
    )
)

# Aggregated version - all cross-book letters
agg_cross_book_heatmap = alt.Chart(expanded_all).mark_rect(height=8).encode(
    y=alt.Y("individual_book:O", title="Other books with shared letters", sort="ascending"),
    x=alt.X("gathering", sort=original_gathering_order),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="greys"), legend=alt.Legend(title="Number of shared letters")),
    opacity=alt.value(0.9),
    tooltip=["individual_book", "gathering", "count()"]
).properties(
    title="Damaged type presence in other books across fortysermons gatherings",
    height=alt.Step(15),
    width=800
).resolve_scale(
    color='independent'
)
agg_cross_book_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# Aggregated version with only chunk letters
agg_chunk_cross_book_heatmap = alt.Chart(expanded_chunk).mark_rect(height=8).encode(
    y=alt.Y("individual_book:O", title="Other books with shared chunk letters", sort="ascending"),
    x=alt.X("gathering", sort=original_gathering_order),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="greys"), legend=alt.Legend(title="Number of shared letters")),
    opacity=alt.value(0.9),
    tooltip=["individual_book", "gathering", "count()"]
).properties(
    title="Chunk damaged type presence in other books across fortysermons gatherings",
    height=alt.Step(15),
    width=800
).resolve_scale(
    color='independent'
)
agg_chunk_cross_book_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# Aggregated version without any chunk letters
agg_no_chunk_cross_book_heatmap = alt.Chart(expanded_no_chunk).mark_rect(height=8).encode(
    y=alt.Y("individual_book:O", title="Other books with shared non-chunk letters", sort="ascending", axis=alt.Axis(grid=True, gridOpacity=0.5)),
    x=alt.X("gathering", sort=original_gathering_order, axis=alt.Axis(grid=True, gridOpacity=0.5)),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="blues"), legend=alt.Legend(title="Number of shared letters")),
    opacity=alt.value(0.9),
    tooltip=["individual_book", "gathering", "count()"]
).properties(
    title="Non-chunk damaged type presence in other books across fortysermons gatherings",
    height=alt.Step(15),
    width=800
).resolve_scale(
    color='independent'
)
agg_no_chunk_cross_book_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# Aggregated version with section background shading
if section_background is not None:
    # Create background layer for aggregated plot - match encoding exactly with main layer
    background_layer_agg = alt.Chart(section_background).mark_rect().encode(
        x=alt.X("gathering", sort=original_gathering_order),
        color=alt.Color("section:N", 
            scale=alt.Scale(scheme="pastel1"),
            legend=alt.Legend(title="Part")
        ),
        opacity=alt.value(0.3)
    )
    
    # Create the main aggregated heatmap layer
    heatmap_layer_agg = alt.Chart(expanded_no_chunk).mark_rect(height=8).encode(
        y=alt.Y("individual_book:O", title="Other books with shared non-chunk letters", sort="ascending", axis=alt.Axis(grid=True, gridOpacity=0.5)),
        x=alt.X("gathering", sort=original_gathering_order, axis=alt.Axis(grid=True, gridOpacity=0.5)),
        color=alt.Color("count():Q", scale=alt.Scale(scheme="blues"), legend=alt.Legend(title="Number of shared letters")),
        opacity=alt.value(0.9),
        tooltip=["individual_book", "gathering", "count()"]
    )
    
    # Layer them together
    agg_no_chunk_cross_book_heatmap_shaded = alt.layer(
        background_layer_agg,
        heatmap_layer_agg
    ).properties(
        title="Non-chunk damaged type presence in other books across fortysermons gatherings (with section shading)",
        height=alt.Step(15),
        width=800
    ).resolve_scale(
        color='independent'
    ).configure_axis(
        grid=True, tickBand="extent"
    )
else:
    agg_no_chunk_cross_book_heatmap_shaded = agg_no_chunk_cross_book_heatmap

# Create expanded datasets for clustered versions
expanded_clustered_all = expand_other_books_data(full_sample_with_clusters.filter(pl.col("appears_in_other_books")))
expanded_clustered_chunk = expand_other_books_data(
    full_sample_with_clusters.filter(
        (pl.col("appears_in_other_books")) & 
        (pl.col("filename").str.contains("chunk"))
    )
)
expanded_clustered_no_chunk = expand_other_books_data(
    full_sample_no_chunks_with_clusters.filter(pl.col("appears_in_other_books"))
)

# Aggregated clustered versions
agg_clustered_cross_book_heatmap = alt.Chart(expanded_clustered_all).mark_rect(height=8).encode(
    y=alt.Y("individual_book:O", title="Other books with shared letters", sort="ascending"),
    x=alt.X("gathering", sort=original_gathering_order),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="greys"), legend=alt.Legend(title="Number of shared letters")),
    facet=alt.Facet("cluster:N", columns=2),
    opacity=alt.value(0.9),
    tooltip=["individual_book", "gathering", "count()", "cluster"]
).properties(
    title="Damaged type presence in other books across fortysermons gatherings (by cluster)",
    height=alt.Step(15),
    width=400
).resolve_scale(
    color='independent'
)
agg_clustered_cross_book_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

agg_clustered_chunk_cross_book_heatmap = alt.Chart(expanded_clustered_chunk).mark_rect(height=8).encode(
    y=alt.Y("individual_book:O", title="Other books with shared chunk letters", sort="ascending"),
    x=alt.X("gathering", sort=original_gathering_order),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="greys"), legend=alt.Legend(title="Number of shared letters")),
    facet=alt.Facet("cluster:N", columns=2),
    opacity=alt.value(0.9),
    tooltip=["individual_book", "gathering", "count()", "cluster"]
).properties(
    title="Chunk damaged type presence in other books across fortysermons gatherings (by cluster)",
    height=alt.Step(15),
    width=400
).resolve_scale(
    color='independent'
)
agg_clustered_chunk_cross_book_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

agg_clustered_no_chunk_cross_book_heatmap = alt.Chart(expanded_clustered_no_chunk).mark_rect(height=8).encode(
    y=alt.Y("individual_book:O", title="Other books with shared non-chunk letters", sort="ascending"),
    x=alt.X("gathering", sort=original_gathering_order),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="greys"), legend=alt.Legend(title="Number of shared letters")),
    facet=alt.Facet("cluster:N", columns=2),
    opacity=alt.value(0.9),
    tooltip=["individual_book", "gathering", "count()", "cluster"]
).properties(
    title="Non-chunk damaged type presence in other books across fortysermons gatherings (by cluster)",
    height=alt.Step(15),
    width=400
).resolve_scale(
    color='independent'
)
agg_clustered_no_chunk_cross_book_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# Save the new clustered versions
clustered_cross_book_heatmap.save("fortysermons_clustered_crossbook_heatmap_latex_bigletter.png", scale_factor=2)
clustered_chunk_cross_book_heatmap.save("fortysermons_clustered_chunk_crossbook_heatmap_latex_bigletter.png", scale_factor=2)
clustered_no_chunk_cross_book_heatmap.save("fortysermons_clustered_nochunk_crossbook_heatmap_latex_bigletter.png", scale_factor=2)

# Save the new aggregated versions
agg_cross_book_heatmap.save("fortysermons_agg_crossbook_heatmap_latex_bigletter.png", scale_factor=2)
agg_chunk_cross_book_heatmap.save("fortysermons_agg_chunk_crossbook_heatmap_latex_bigletter.png", scale_factor=2)
agg_no_chunk_cross_book_heatmap.save("fortysermons_agg_nochunk_crossbook_heatmap_latex_bigletter.png", scale_factor=2)
agg_clustered_cross_book_heatmap.save("fortysermons_agg_clustered_crossbook_heatmap_latex_bigletter.png", scale_factor=2)
agg_clustered_chunk_cross_book_heatmap.save("fortysermons_agg_clustered_chunk_crossbook_heatmap_latex_bigletter.png", scale_factor=2)
agg_clustered_no_chunk_cross_book_heatmap.save("fortysermons_agg_clustered_nochunk_crossbook_heatmap_latex_bigletter.png", scale_factor=2)


# Get unique letters
unique_letters = full_sample["letter_with_class"].unique().sort().to_list()

# Create co-occurrence matrix data
adjacency_data = []

for i, letter_i in enumerate(unique_letters):
    for j, letter_j in enumerate(unique_letters):
        # Only include lower triangular part (i >= j)
        if i >= j:
            # Get gatherings for each letter
            filter_i = full_sample.filter(pl.col("letter_with_class") == letter_i)
            filter_j = full_sample.filter(pl.col("letter_with_class") == letter_j)
            
            gatherings_with_i = set(filter_i["gathering"].unique().to_list())
            gatherings_with_j = set(filter_j["gathering"].unique().to_list())
            
            # Count gatherings where both letters appear
            common_gatherings = len(gatherings_with_i.intersection(gatherings_with_j))
            
            adjacency_data.append({
                "letter_i": letter_i,
                "letter_j": letter_j,
                "common_gatherings": common_gatherings
            })

# Convert to DataFrame
adjacency_df = pl.DataFrame(adjacency_data)

# Create the lower triangular adjacency matrix heatmap
lower_triangle_heatmap = alt.Chart(adjacency_df).mark_rect().encode(
    x=alt.X("letter_j:O", title="Letter"),
    y=alt.Y("letter_i:O", title="Letter"),
    color=alt.Color("common_gatherings:Q", scale=alt.Scale(scheme="viridis")),
    tooltip=["letter_i", "letter_j", "common_gatherings"]
).properties(
    title="Letter Co-occurrence Adjacency Matrix",
    width=400,
    height=400
)

lower_triangle_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# Create versions that show ALL gatherings from the gathering map (including those without data)
print("="*60)
print("CREATING COMPLETE DOMAIN VERSIONS (ALL GATHERINGS)")
print("="*60)

# Get all gatherings from the gathering map (using custom sorted order)
all_gatherings_from_map = all_gatherings_sorted
print(f"Total gatherings in gathering map: {len(all_gatherings_from_map)}")
print(f"Gatherings with data: {len(full_sample['gathering'].unique())}")
missing_gatherings = set(all_gatherings_from_map) - set(full_sample['gathering'].unique())
print(f"Gatherings without data: {len(missing_gatherings)}")
if missing_gatherings:
    print(f"Missing gatherings: {sorted(list(missing_gatherings), key=gathering_sort_key)}")

# Create complete domain versions of key plots
# Complete domain heatmap
heatmap_complete = alt.Chart(full_sample).mark_rect(height=5).encode(
    y=alt.Y("letter_with_class:O", title="letter type"),
    x=alt.X("gathering", scale=alt.Scale(domain=all_gatherings_from_map)),
    color=alt.condition(
        alt.datum.appears_in_other_books,
        alt.Color("other_books:N", legend=alt.Legend(
            title="Also appears in books",
            labelLimit=300,
            symbolLimit=50,
            labelExpr="split(datum.label, ' + ')"
        )),
        alt.value("lightgray")
    ),
    opacity=alt.condition(
        alt.datum.appears_in_other_books,
        alt.value(0.8),
        alt.value(0.3)
    ),
    stroke=alt.condition(
        alt.datum.appears_in_other_books,
        alt.value("red"),
        alt.value("transparent")
    ),
    strokeWidth=alt.condition(
        alt.datum.appears_in_other_books,
        alt.value(1),
        alt.value(0)
    ),
    tooltip=["letter_with_class", "gathering", "other_books", "appears_in_other_books"]
).properties(
    title="Letter heatmap across all of fortysermons gatherings",
    height=alt.Step(10),
    width=800
).resolve_scale(
    color='independent'
)

# Complete domain cross-book heatmap
cross_book_heatmap_complete = alt.Chart(full_sample.filter(pl.col("appears_in_other_books"))).mark_rect(height=8).encode(
    y=alt.Y("letter_with_class:O", title="Letters also in other books"),
    x=alt.X("gathering", scale=alt.Scale(domain=all_gatherings_from_map)),
    color=alt.Color("other_books:N", legend=alt.Legend(
        title="Also appears in books",
        labelLimit=300,
        symbolLimit=50,
        labelExpr="split(datum.label, ' + ')"
    )),
    opacity=alt.value(0.9),
    tooltip=["letter_with_class", "gathering", "other_books"]
).properties(
    title="Cross-book letters across all of fortysermons gatherings",
    height=alt.Step(12),
    width=800
).resolve_scale(
    color='independent'
)

# Complete domain aggregated cross-book heatmap
agg_cross_book_heatmap_complete = alt.Chart(expanded_all).mark_rect(height=8).encode(
    y=alt.Y("individual_book:O", title="Other books with shared letters", sort="ascending"),
    x=alt.X("gathering", sort=all_gatherings_from_map, scale=alt.Scale(domain=all_gatherings_from_map)),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="greys"), legend=alt.Legend(title="Number of shared letters")),
    opacity=alt.value(0.9),
    tooltip=["individual_book", "gathering", "count()"]
).properties(
    title="Damaged type presence across all of fortysermons gatherings",
    height=alt.Step(15),
    width=800
).resolve_scale(
    color='independent'
)

# Complete domain aggregated no-chunk cross-book heatmap  
agg_no_chunk_cross_book_heatmap_complete = alt.Chart(expanded_no_chunk).mark_rect(height=8).encode(
    y=alt.Y("individual_book:O", title="Other books with shared non-chunk letters", sort="ascending", axis=alt.Axis(grid=True, gridOpacity=0.5)),
    x=alt.X("gathering", sort=all_gatherings_from_map, scale=alt.Scale(domain=all_gatherings_from_map), axis=alt.Axis(grid=True, gridOpacity=0.5)),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="blues"), legend=alt.Legend(title="Number of shared letters")),
    opacity=alt.value(0.9),
    tooltip=["individual_book", "gathering", "count()"]
).properties(
    title="Non-chunk damaged type presence across all of fortysermons gatherings",
    height=alt.Step(15),
    width=800
).resolve_scale(
    color='independent'
)

print("Created complete domain versions of key plots")

# save fig
images_l.save("fortysermons_letters_latex_bigletter.png", scale_factor=2)
heatmap.save("fortysermons_heatmap_latex_bigletter.png", scale_factor=2)
cross_book_heatmap.save("fortysermons_crossbook_heatmap_latex_bigletter.png", scale_factor=2)
if section_background is not None:
    cross_book_heatmap_shaded.save("fortysermons_crossbook_heatmap_shaded_latex_bigletter.png", scale_factor=2)
chunk_cross_book_heatmap.save("fortysermons_chunk_crossbook_heatmap_latex_bigletter.png", scale_factor=2)
no_chunk_cross_book_heatmap.save("fortysermons_nochunk_crossbook_heatmap_latex_bigletter.png", scale_factor=2)
lower_triangle_heatmap.save("fortysermons_adjacency_latex_bigletter.png", scale_factor=2)

# Save complete domain versions
heatmap_complete.save("fortysermons_heatmap_complete_domain_latex_bigletter.png", scale_factor=2)
cross_book_heatmap_complete.save("fortysermons_crossbook_complete_domain_latex_bigletter.png", scale_factor=2)
agg_cross_book_heatmap_complete.save("fortysermons_agg_crossbook_complete_domain_latex_bigletter.png", scale_factor=2)
agg_no_chunk_cross_book_heatmap_complete.save("fortysermons_agg_nochunk_crossbook_complete_domain_latex_bigletter.png", scale_factor=2)

# Save shaded versions
if section_background is not None:
    agg_no_chunk_cross_book_heatmap_shaded.save("fortysermons_agg_nochunk_crossbook_shaded_latex_bigletter.png", scale_factor=2)

print("Saved plots including cross-book letter highlighting")
print("Saved complete domain versions showing data gaps across all gatherings")
if section_background is not None:
    print("Saved shaded versions showing Part splits")
