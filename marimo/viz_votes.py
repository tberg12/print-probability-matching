import polars as pl
import altair as alt
from utils import load_and_prepare_sample

# Configuration: Set to True to use manually defined clusters from Chris's CSV
USE_MANUAL_CLUSTERS = True
print(f"USE_MANUAL_CLUSTERS = {USE_MANUAL_CLUSTERS}")
# Configuration: Set to True to only show high confidence matches in plots
only_hi_matches = True
print(f"only_hi_matches = {only_hi_matches}")
print("="*60)
print("="*60)

# sample = pl.read_excel("sample_review_updated_latex_bigletter_dedup.xlsx")
all_sample = load_and_prepare_sample("LockeSpinozaMatches_processed.csv", only_hi_matches=only_hi_matches)
print(all_sample.select(pl.col("book_name").unique()).to_series().to_list())
issue_map = pl.read_csv("votes_map_nikolai.csv")

# Load manual cluster assignments if available
# Load manual cluster assignments if available
manual_clusters_df = None
manual_cluster_map = {}  # Maps issue -> cluster_id from CSV
manual_empty_cluster_issues = []  # Issues with empty cluster values in CSV
if USE_MANUAL_CLUSTERS:
    try:
        manual_clusters_df = pl.read_csv("votes_issue_clusters_chris.csv")
        print("Loaded manual clusters from votes_issue_clusters_chris.csv")
        print(f"Manual cluster assignments: {len(manual_clusters_df)} issues")
        
        # Build a map of issue -> cluster_id for ground truth
        for row in manual_clusters_df.iter_rows(named=True):
            issue_val = str(row['issue'])
            cluster_val = row['cluster']
            
            # Store non-empty cluster assignments
            if cluster_val is not None and not (isinstance(cluster_val, str) and cluster_val.strip() == ''):
                manual_cluster_map[issue_val] = str(cluster_val)
            else:
                # Track issues with empty cluster values
                manual_empty_cluster_issues.append(issue_val)
        
        print(f"Ground truth clusters loaded: {len(manual_cluster_map)} issues with cluster assignments")
        print(f"Issues without manual cluster: {len(manual_empty_cluster_issues)}")
    except Exception as e:
        print(f"Warning: Could not load manual clusters: {e}")
        manual_clusters_df = None
        manual_cluster_map = {}
        manual_empty_cluster_issues = []

print(all_sample.select(pl.col("book_name").unique()).to_series().to_list())

# Find letters that appear in non-votes books
non_votes_letters = set(
    all_sample
    .filter(~pl.col("filename").str.contains("votes"))
    ["letter_with_class"]
    .unique()
    .to_list()
)

print(f"Found {len(non_votes_letters)} letter types in non-votes books")

# Print book names found in the dataset
all_book_names = set(all_sample["book_name"].unique().to_list())
votes_book_names = set(
    all_sample
    .filter(pl.col("filename").str.contains("votes"))
    ["book_name"]
    .unique()
    .to_list()
)
non_votes_book_names = all_book_names - votes_book_names

print(f"All book names found: {sorted(list(all_book_names))}")
print(f"votes book names: {sorted(list(votes_book_names))}")
print(f"non-votes book names: {sorted(list(non_votes_book_names))}")

# Helper function to create background shading for clusters
def create_cluster_background(issue_to_cluster_dict):
    """Create background rectangles for cluster identification"""
    rows = []
    for issue, cluster in issue_to_cluster_dict.items():
        formatted_issue = f"{int(issue):02d}" if str(issue).isdigit() else str(issue)
        rows.append({
            "issue": formatted_issue,
            "cluster": cluster
        })
    
    if not rows:
        return None
    
    return pl.DataFrame(rows)

# load issue map - votes map is simpler with page_num -> issue mapping
# Convert issues to strings and create enum
unique_issues = sorted(issue_map["issue"].unique().to_list())
unique_issues_str = [f"{i:02d}" for i in unique_issues]
# unique_issues_str = [str(i) for i in unique_issues]
issue_enum = pl.Enum(unique_issues_str)

# order by page number and join asof to get issue info for each page
full_sample = (
    all_sample
    .filter(pl.col("filename").str.contains("votes"))
    .with_columns(page_number=pl.col("page_number").cast(pl.Int64))
    .sort("page_number")
    .join_asof(
        issue_map,
        left_on="page_number",
        right_on="page_num",
        strategy="backward",
    )
    .select(
        "root_image",
        "page_number",
        "filename",
        "image",
        "is_root",
        "issue",
        "letter",
        "letter_class",
        "letter_with_class",
        "book_name"
    )
    .with_columns(
        # issue=pl.col("issue").cast(pl.Utf8).cast(issue_enum),
        issue=pl.col("issue").cast(pl.Utf8).str.zfill(2).cast(issue_enum),
        appears_in_other_books=pl.col("letter_with_class").is_in(list(non_votes_letters))
    )
)
full_sample = full_sample.sort("letter_with_class")

# Create a mapping of letter_with_class to non-votes books where they appear
letter_to_other_books = {}
for letter in full_sample["letter_with_class"].unique().to_list():
    other_books = set(
        all_sample
        .filter(
            (pl.col("letter_with_class") == letter) & 
            (~pl.col("filename").str.contains("votes"))
        )
        ["book_name"]
        .unique()
        .to_list()
    )
    if other_books:
        # Join multiple books with " + " if letter appears in multiple non-votes books
        letter_to_other_books[letter] = " + ".join(sorted(list(other_books)))
    else:
        letter_to_other_books[letter] = "votes only"

# Add the other_books column to full_sample
full_sample = full_sample.with_columns(
    other_books=pl.col("letter_with_class").map_elements(
        lambda x: letter_to_other_books[x], return_dtype=pl.Utf8
    )
)

# Print statistics about cross-book letter occurrences
votes_letters = set(full_sample["letter_with_class"].unique().to_list())
cross_book_letters = votes_letters.intersection(non_votes_letters)
votes_only_letters = votes_letters - non_votes_letters

print(f"Letters in votes: {len(votes_letters)}")
print(f"Letters that also appear in other books: {len(cross_book_letters)}")
print(f"Letters only in votes: {len(votes_only_letters)}")
print(f"Cross-book letter examples: {sorted(list(cross_book_letters))[:10]}")
print(f"votes-only letter examples: {sorted(list(votes_only_letters))[:10]}")
print()

# ========== PRINTER NAME EXTRACTION (from graph.py logic) ==========
# Extract printer name from root_image column
all_sample = all_sample.with_columns(
    printer_name=pl.col("root_image").str.split("_").list.get(0)
)

# Apply specific printer name rules based on keywords in root_image
all_sample = all_sample.with_columns(
    printer_name=pl.when(pl.col("root_image").str.contains("fortysermons1685"))
    .then(pl.lit("everingham"))
    .when(pl.col("root_image").str.contains("treatises"))
    .then(pl.lit("anon"))
    .when(pl.col("root_image").str.contains("sermons"))
    .then(pl.lit("everingham"))
    .when(pl.col("root_image").str.contains("theological"))
    .then(pl.lit("anon"))
    .otherwise(pl.col("printer_name"))
)

# Normalize printer names - treat "reveringham" and "everingham" as the same
all_sample = all_sample.with_columns(
    printer_name=pl.when(pl.col("printer_name").str.to_lowercase() == "reveringham")
    .then(pl.lit("everingham"))
    .otherwise(pl.col("printer_name"))
)

# Create a mapping from book_name to printer_name
book_to_printer = {}
for book in all_sample["book_name"].unique().to_list():
    # Check for specific patterns first
    if "fortysermons1685" in book:
        book_to_printer[book] = "everingham"
    elif "twotreatises" in book.lower():
        book_to_printer[book] = "anon"
    elif "religionandreason1688" in book:
        book_to_printer[book] = "tbraddyll"
    else:
        # Get the most common printer for this book
        printers = all_sample.filter(pl.col("book_name") == book)["printer_name"].unique().to_list()
        if printers:
            book_to_printer[book] = printers[0]
        else:
            book_to_printer[book] = "unknown"

print("="*60)
print("BOOK TO PRINTER MAPPING")
print("="*60)
for book, printer in sorted(book_to_printer.items()):
    print(f"  {book}: {printer}")
print()

# Create a mapping of letter_with_class to printers where they appear (in non-votes books)
letter_to_other_printers = {}
for letter in full_sample["letter_with_class"].unique().to_list():
    other_books = set(
        all_sample
        .filter(
            (pl.col("letter_with_class") == letter) &
            (~pl.col("filename").str.contains("votes"))
        )
        ["book_name"]
        .unique()
        .to_list()
    )
    if other_books:
        # Map books to printers and deduplicate
        printers = set(book_to_printer.get(book, "unknown") for book in other_books)
        letter_to_other_printers[letter] = " + ".join(sorted(list(printers)))
    else:
        letter_to_other_printers[letter] = "votes only"

# Add the other_printers column to full_sample
full_sample = full_sample.with_columns(
    other_printers=pl.col("letter_with_class").map_elements(
        lambda x: letter_to_other_printers[x], return_dtype=pl.Utf8
    )
)

# Print printer statistics
print("="*60)
print("PRINTER STATISTICS FOR CROSS-BOOK LETTERS")
print("="*60)
printer_counts = {}
for letter, printers in letter_to_other_printers.items():
    if printers != "votes only":
        for printer in printers.split(" + "):
            printer_counts[printer] = printer_counts.get(printer, 0) + 1
for printer, count in sorted(printer_counts.items(), key=lambda x: -x[1]):
    print(f"  {printer}: {count} letter types")
print()

# ========== DETAILED PRINTER MAPPING (anon shows book name) ==========
# Create a mapping that shows "anon (book_name)" for anonymous printers
# but just the printer name for known printers
letter_to_other_printers_detailed = {}
for letter in full_sample["letter_with_class"].unique().to_list():
    other_books = set(
        all_sample
        .filter(
            (pl.col("letter_with_class") == letter) &
            (~pl.col("filename").str.contains("votes"))
        )
        ["book_name"]
        .unique()
        .to_list()
    )
    if other_books:
        # For each book, get printer and create detailed label
        detailed_labels = set()
        for book in other_books:
            printer = book_to_printer.get(book, "unknown")
            if printer == "anon":
                # Show anon with book name for more detail
                detailed_labels.add(f"anon ({book})")
            else:
                detailed_labels.add(printer)
        letter_to_other_printers_detailed[letter] = " + ".join(sorted(list(detailed_labels)))
    else:
        letter_to_other_printers_detailed[letter] = "votes only"

# Create printer_bookname format (printer_bookname for all books)
letter_to_printer_bookname = {}
for letter in full_sample["letter_with_class"].unique().to_list():
    other_books = set(
        all_sample
        .filter(
            (pl.col("letter_with_class") == letter) &
            (~pl.col("filename").str.contains("votes"))
        )
        ["book_name"]
        .unique()
        .to_list()
    )
    if other_books:
        # For each book, create printer_bookname label
        printer_bookname_labels = set()
        for book in other_books:
            printer = book_to_printer.get(book, "unknown")
            printer_bookname_labels.add(f"{printer}_{book}")
        letter_to_printer_bookname[letter] = " + ".join(sorted(list(printer_bookname_labels)))
    else:
        letter_to_printer_bookname[letter] = "votes only"

# Add the other_printers_detailed column to full_sample
full_sample = full_sample.with_columns(
    other_printers_detailed=pl.col("letter_with_class").map_elements(
        lambda x: letter_to_other_printers_detailed[x], return_dtype=pl.Utf8
    ),
    other_printer_bookname=pl.col("letter_with_class").map_elements(
        lambda x: letter_to_printer_bookname[x], return_dtype=pl.Utf8
    )
)

# Print detailed printer statistics
print("="*60)
print("DETAILED PRINTER STATISTICS (anon with book names)")
print("="*60)
detailed_printer_counts = {}
for letter, printers in letter_to_other_printers_detailed.items():
    if printers != "votes only":
        for printer in printers.split(" + "):
            detailed_printer_counts[printer] = detailed_printer_counts.get(printer, 0) + 1
for printer, count in sorted(detailed_printer_counts.items(), key=lambda x: -x[1]):
    print(f"  {printer}: {count} letter types")
print()



# breakpoint()
# print full_sample with letter_with_class == O83
# print(full_sample.filter(pl.col("letter_with_class") == "O83"))

# plot letters by issue
images_l = (
    alt.Chart(full_sample)
    .mark_image()
    .encode(
        y=alt.Y("letter_with_class:O"), 
        x=alt.X("issue"), 
        url="image",
        # stroke=alt.condition(
        #     alt.datum.appears_in_other_books,
        #     alt.value("red"),
        #     alt.value("transparent")
        # ),
        # strokeWidth=alt.condition(
        #     alt.datum.appears_in_other_books,
        #     alt.value(2),
        #     alt.value(0)
        # )
    )
).properties(
    width=900,
    # height=180,
)
images_l.configure_axis(
    grid=True, tickBand="extent"
)

# heatmap of letters by issue with cross-book highlighting
heatmap = alt.Chart(full_sample).mark_rect(height=5).encode(
    y=alt.Y("letter_with_class:O", title="letter type"),
    x=alt.X("issue"),
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
    tooltip=["letter_with_class", "issue", "other_books", "appears_in_other_books"]
).properties(
    height=alt.Step(10),
    width=800
).resolve_scale(
    color='independent'
)


# Iterative clustering of issues based on shared letters
print("="*60)
if USE_MANUAL_CLUSTERS and manual_cluster_map:
    print("ITERATIVE CLUSTERING WITH MANUAL GROUND TRUTH")
else:
    print("ITERATIVE CLUSTERING OF ISSUES")
print("="*60)

# Get unique issues and create a mapping of issue -> letters
unique_issues = full_sample["issue"].unique().sort().to_list()
issue_to_letters = {}

for issue in unique_issues:
    letters = set(full_sample.filter(pl.col("issue") == issue)["letter_with_class"].unique().to_list())
    issue_to_letters[issue] = letters

print(f"Total issues in data: {len(unique_issues)}")

# Initialize with manual clusters if available
if USE_MANUAL_CLUSTERS and (manual_cluster_map or manual_empty_cluster_issues):
    print(f"Loaded {len(manual_cluster_map)} manual cluster assignments from CSV")
    
    # Add issues from manual clusters (including empty cluster values) that aren't in the data yet
    unique_issues_str_set = set([str(issue) for issue in unique_issues])
    
    # Issues with cluster assignments not in data
    issues_not_in_data = set(manual_cluster_map.keys()) - unique_issues_str_set
    
    # Issues with empty cluster values not in data
    empty_cluster_issues_not_in_data = set(manual_empty_cluster_issues) - unique_issues_str_set
    
    # Combine all issues not in data
    all_issues_not_in_data = issues_not_in_data | empty_cluster_issues_not_in_data
    
    if all_issues_not_in_data:
        print(f"Note: {len(all_issues_not_in_data)} issues in CSV not found in character data: {sorted(all_issues_not_in_data)}")
        print("  These issues will be included but have no letter associations")
        
        # Add these issues to issue_to_letters with empty letter sets
        for issue in all_issues_not_in_data:
            issue_to_letters[issue] = set()
        
        # Add them to unique_issues list
        unique_issues.extend([issue for issue in all_issues_not_in_data])
    
    print(f"Total issues to cluster (including manual): {len(unique_issues)}")
    print(f"Using {len(manual_cluster_map)} manual cluster assignments as seed clusters")
    print(f"Treating {len(manual_empty_cluster_issues)} empty-cluster issues as single-issue seed clusters")
    print("Automatic clustering will merge seed clusters if they share letters...\n")
    
    # Pre-initialize clusters from manual assignments as SEED clusters
    # Group issues by their manual cluster ID
    cluster_id_to_issues = {}
    for issue, cluster_id in manual_cluster_map.items():
        if cluster_id not in cluster_id_to_issues:
            cluster_id_to_issues[cluster_id] = set()
        cluster_id_to_issues[cluster_id].add(issue)
    
    # Start with manual seed clusters
    clusters = list(cluster_id_to_issues.values())
    
    # Also add each empty-cluster issue as its own single-issue seed cluster
    # This allows them to be merged with other clusters during the merging phase
    for issue in manual_empty_cluster_issues:
        # Only add if the issue exists in the data
        if issue in [str(i) for i in unique_issues]:
            clusters.append({issue})
    
    # Don't mark manual issues as "assigned" - let them participate in merging
    assigned_issues = set()
    
    print(f"Initialized {len(clusters)} seed clusters from ground truth (including {len(manual_empty_cluster_issues)} single-issue clusters)")
    for i, cluster in enumerate(clusters, 1):
        print(f"  Seed cluster {i}: {sorted([str(g) for g in cluster])}")
    print()
    
    # Now run automatic clustering to:
    # 1. Merge seed clusters if they share letters
    # 2. Add unassigned issues to appropriate clusters
    # 3. Create new clusters for isolated issues
    
    print("Running automatic clustering to merge seed clusters and add remaining issues...")
    
    # Track which cluster each issue belongs to for merging
    issue_to_cluster_idx = {}
    for idx, cluster in enumerate(clusters):
        for issue in cluster:
            issue_to_cluster_idx[issue] = idx
    
    # Function to merge two clusters
    def merge_clusters(idx1, idx2):
        """Merge cluster idx2 into cluster idx1"""
        if idx1 == idx2:
            return
        clusters[idx1] = clusters[idx1].union(clusters[idx2])
        # Update all issues in cluster idx2 to point to idx1
        for issue in clusters[idx2]:
            issue_to_cluster_idx[issue] = idx1
        clusters[idx2] = set()  # Clear the merged cluster
    
    # Check for merges between existing seed clusters
    for i in range(len(clusters)):
        if not clusters[i]:  # Skip empty clusters
            continue
        for j in range(i + 1, len(clusters)):
            if not clusters[j]:  # Skip empty clusters
                continue
            
            # Check if any issues in cluster i share letters with any issues in cluster j
            for issue_i in clusters[i]:
                if issue_i not in issue_to_letters:
                    continue
                letters_i = issue_to_letters[issue_i]
                
                for issue_j in clusters[j]:
                    if issue_j not in issue_to_letters:
                        continue
                    letters_j = issue_to_letters[issue_j]
                    shared = letters_i.intersection(letters_j)
                    
                    if shared:
                        print(f"  Merging seed clusters: {issue_i} (cluster {i+1}) shares {sorted(list(shared))} with {issue_j} (cluster {j+1})")
                        merge_clusters(i, j)
                        break
                if not clusters[j]:  # If we merged, break outer loop too
                    break
    
    # Remove empty clusters
    clusters = [c for c in clusters if c]
    
    # Update assigned_issues to include all issues in merged clusters
    assigned_issues = set()
    for cluster in clusters:
        assigned_issues.update(cluster)
    
    print(f"After merging: {len(clusters)} clusters")
    for i, cluster in enumerate(clusters, 1):
        print(f"  Merged cluster {i}: {sorted([str(g) for g in cluster])}")
    print()
    
else:
    print(f"Total issues to cluster: {len(unique_issues)}")
    print("Building clusters iteratively...\n")
    assigned_issues = set()
    clusters = []

def find_connected_issues(start_issue, current_cluster, assigned_issues_ref):
    """Iteratively find all issues connected to the start issue through shared letters"""
    to_process = [start_issue]
    processed = set()
    
    while to_process:
        current_issue = to_process.pop(0)
        
        if current_issue in processed:
            continue
            
        processed.add(current_issue)
        current_cluster.add(current_issue)
        assigned_issues_ref.add(current_issue)
        
        # Get letters for current issue
        current_letters = issue_to_letters[current_issue]
        
        # Find all issues that share any letter with current issue
        for other_issue in unique_issues:
            if other_issue not in processed and other_issue not in assigned_issues_ref:
                other_letters = issue_to_letters[other_issue]
                shared_letters = current_letters.intersection(other_letters)
                
                if shared_letters:
                    print(f"  Adding {other_issue} to cluster (shares {sorted(list(shared_letters))} with {current_issue})")
                    to_process.append(other_issue)

# Build clusters iteratively for unassigned issues
cluster_id = len(clusters) + 1
for issue in unique_issues:
    if issue not in assigned_issues:
        print(f"\nStarting Cluster {cluster_id} with issue: {issue}")
        
        current_cluster = set()
        find_connected_issues(issue, current_cluster, assigned_issues)
        
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
            print(f"  Cluster {cluster_id} completed with {len(current_cluster)} issues: {sorted([str(g) for g in current_cluster])}")
        else:
            print(f"  {issue} is isolated (no shared letters with other issues)")
            
        cluster_id += 1

# Count isolated issues
isolated = []
for issue in unique_issues:
    if not any(issue in cluster for cluster in clusters):
        isolated.append(issue)

print("\n" + "="*60)
print("CLUSTERING RESULTS")
print("="*60)

if clusters:
    print(f"Found {len(clusters)} clusters with multiple issues:")
    
    for i, cluster in enumerate(clusters, 1):
        cluster_list = sorted([str(g) for g in cluster])
        print(f"\nCluster {i} ({len(cluster)} issues):")
        print(f"  Issues: {', '.join(cluster_list)}")
        
        # Find all letters that appear in multiple issues within this cluster
        all_shared_pairs = []
        cluster_issues = list(cluster)
        
        for j in range(len(cluster_issues)):
            for k in range(j + 1, len(cluster_issues)):
                g1, g2 = cluster_issues[j], cluster_issues[k]
                shared = issue_to_letters[g1].intersection(issue_to_letters[g2])
                if shared:
                    all_shared_pairs.append(f"{g1}↔{g2}: {sorted(list(shared))}")
        
        if all_shared_pairs:
            print("  Shared letter pairs (sample of first 5):")
            for pair in all_shared_pairs[:5]:
                print(f"    {pair}")
            if len(all_shared_pairs) > 5:
                print(f"    ... and {len(all_shared_pairs) - 5} more pairs")
else:
    print("No clusters with multiple issues found.")

if isolated:
    print(f"\nIsolated issues ({len(isolated)}): {sorted([str(g) for g in isolated])}")

print("="*60)

# Clustering without 'chunk' files
print("\n" + "="*60)
if USE_MANUAL_CLUSTERS and manual_clusters_df is not None:
    print("USING MANUAL CLUSTER ASSIGNMENTS (EXCLUDING 'CHUNK' FILES)")
else:
    print("ITERATIVE CLUSTERING OF ISSUES (EXCLUDING 'CHUNK' FILES)")
print("="*60)

# Filter out rows where filename contains 'chunk'
full_sample_no_chunks = full_sample.filter(~pl.col("filename").str.contains("chunk"))

print(f"Original sample size: {len(full_sample)}")
print(f"After filtering out 'chunk' files: {len(full_sample_no_chunks)}")

# Get unique issues and create a mapping of issue -> letters (without chunks)
unique_issues_no_chunks = full_sample_no_chunks["issue"].unique().sort().to_list()
issue_to_letters_no_chunks = {}

for issue in unique_issues_no_chunks:
    letters = set(full_sample_no_chunks.filter(pl.col("issue") == issue)["letter_with_class"].unique().to_list())
    issue_to_letters_no_chunks[issue] = letters

print(f"Total issues in data (no chunks): {len(unique_issues_no_chunks)}")

# Initialize with manual clusters if available
if USE_MANUAL_CLUSTERS and (manual_cluster_map or manual_empty_cluster_issues):
    # Add issues from manual clusters (including empty cluster values) that aren't in the no-chunks data yet
    unique_issues_no_chunks_str_set = set([str(issue) for issue in unique_issues_no_chunks])
    
    # Issues with cluster assignments not in no-chunks data
    issues_not_in_no_chunks_data = set(manual_cluster_map.keys()) - unique_issues_no_chunks_str_set
    
    # Issues with empty cluster values not in no-chunks data
    empty_cluster_issues_not_in_no_chunks = set(manual_empty_cluster_issues) - unique_issues_no_chunks_str_set
    
    # Combine all issues not in no-chunks data
    all_issues_not_in_no_chunks = issues_not_in_no_chunks_data | empty_cluster_issues_not_in_no_chunks
    
    if all_issues_not_in_no_chunks:
        print(f"Note: {len(all_issues_not_in_no_chunks)} issues in CSV not found in no-chunks data")
        
        # Add these issues to issue_to_letters_no_chunks with empty letter sets
        for issue in all_issues_not_in_no_chunks:
            issue_to_letters_no_chunks[issue] = set()
        
        # Add them to unique_issues_no_chunks list
        unique_issues_no_chunks.extend([issue for issue in all_issues_not_in_no_chunks])
    
    print(f"Total issues to cluster (no chunks, including manual): {len(unique_issues_no_chunks)}")
    print(f"Using {len(manual_cluster_map)} manual cluster assignments as seed clusters")
    print(f"Treating {len(manual_empty_cluster_issues)} empty-cluster issues as single-issue seed clusters")
    print("Automatic clustering will merge seed clusters if they share letters (no chunks)...\n")
    
    # Pre-initialize clusters from manual assignments as SEED clusters
    cluster_id_to_issues_no_chunks = {}
    for issue, cluster_id in manual_cluster_map.items():
        if cluster_id not in cluster_id_to_issues_no_chunks:
            cluster_id_to_issues_no_chunks[cluster_id] = set()
        cluster_id_to_issues_no_chunks[cluster_id].add(issue)
    
    # Start with manual seed clusters
    clusters_no_chunks = list(cluster_id_to_issues_no_chunks.values())
    
    # Also add each empty-cluster issue as its own single-issue seed cluster
    for issue in manual_empty_cluster_issues:
        # Only add if the issue exists in the data
        if issue in [str(i) for i in unique_issues_no_chunks]:
            clusters_no_chunks.append({issue})
    
    # Don't mark manual issues as "assigned" - let them participate in merging
    assigned_issues_no_chunks = set()
    
    print(f"Initialized {len(clusters_no_chunks)} seed clusters from ground truth (no chunks, including {len(manual_empty_cluster_issues)} single-issue clusters)")
    for i, cluster in enumerate(clusters_no_chunks, 1):
        print(f"  Seed cluster {i}: {sorted([str(g) for g in cluster])}")
    print()
    
    # Run automatic clustering to merge seed clusters
    print("Running automatic clustering to merge seed clusters and add remaining issues (no chunks)...")
    
    # Track which cluster each issue belongs to for merging
    issue_to_cluster_idx_no_chunks = {}
    for idx, cluster in enumerate(clusters_no_chunks):
        for issue in cluster:
            issue_to_cluster_idx_no_chunks[issue] = idx
    
    # Function to merge two clusters
    def merge_clusters_no_chunks(idx1, idx2):
        """Merge cluster idx2 into cluster idx1"""
        if idx1 == idx2:
            return
        clusters_no_chunks[idx1] = clusters_no_chunks[idx1].union(clusters_no_chunks[idx2])
        for issue in clusters_no_chunks[idx2]:
            issue_to_cluster_idx_no_chunks[issue] = idx1
        clusters_no_chunks[idx2] = set()
    
    # Check for merges between existing seed clusters
    for i in range(len(clusters_no_chunks)):
        if not clusters_no_chunks[i]:
            continue
        for j in range(i + 1, len(clusters_no_chunks)):
            if not clusters_no_chunks[j]:
                continue
            
            # Check if any issues in cluster i share letters with any issues in cluster j
            for issue_i in clusters_no_chunks[i]:
                if issue_i not in issue_to_letters_no_chunks:
                    continue
                letters_i = issue_to_letters_no_chunks[issue_i]
                
                for issue_j in clusters_no_chunks[j]:
                    if issue_j not in issue_to_letters_no_chunks:
                        continue
                    letters_j = issue_to_letters_no_chunks[issue_j]
                    shared = letters_i.intersection(letters_j)
                    
                    if shared:
                        print(f"  Merging seed clusters (no chunks): {issue_i} (cluster {i+1}) shares {sorted(list(shared))} with {issue_j} (cluster {j+1})")
                        merge_clusters_no_chunks(i, j)
                        break
                if not clusters_no_chunks[j]:
                    break
    
    # Remove empty clusters
    clusters_no_chunks = [c for c in clusters_no_chunks if c]
    
    # Update assigned_issues to include all issues in merged clusters
    assigned_issues_no_chunks = set()
    for cluster in clusters_no_chunks:
        assigned_issues_no_chunks.update(cluster)
    
    print(f"After merging (no chunks): {len(clusters_no_chunks)} clusters")
    for i, cluster in enumerate(clusters_no_chunks, 1):
        print(f"  Merged cluster {i}: {sorted([str(g) for g in cluster])}")
    print()
    
else:
    print("Building clusters iteratively (excluding chunk files)...\n")
    
    # Track which issues have been assigned to clusters
    assigned_issues_no_chunks = set()
    clusters_no_chunks = []

def find_connected_issues_no_chunks(start_issue, current_cluster, assigned_issues_ref):
    """Iteratively find all issues connected to the start issue through shared letters (no chunks)"""
    to_process = [start_issue]
    processed = set()
    
    while to_process:
        current_issue = to_process.pop(0)
        
        if current_issue in processed:
            continue
            
        processed.add(current_issue)
        current_cluster.add(current_issue)
        assigned_issues_ref.add(current_issue)
        
        # Get letters for current issue
        current_letters = issue_to_letters_no_chunks[current_issue]
        
        # Find all issues that share any letter with current issue
        for other_issue in unique_issues_no_chunks:
            if other_issue not in processed and other_issue not in assigned_issues_ref:
                other_letters = issue_to_letters_no_chunks[other_issue]
                shared_letters = current_letters.intersection(other_letters)
                
                if shared_letters:
                    print(f"  Adding {other_issue} to cluster (shares {sorted(list(shared_letters))} with {current_issue})")
                    to_process.append(other_issue)

# Build clusters iteratively (no chunks)
cluster_id = len(clusters_no_chunks) + 1
for issue in unique_issues_no_chunks:
    if issue not in assigned_issues_no_chunks:
        print(f"\nStarting Cluster {cluster_id} with issue: {issue}")
        
        current_cluster = set()
        find_connected_issues_no_chunks(issue, current_cluster, assigned_issues_no_chunks)
        
        if len(current_cluster) > 1:
            clusters_no_chunks.append(current_cluster)
            print(f"  Cluster {cluster_id} completed with {len(current_cluster)} issues: {sorted([str(g) for g in current_cluster])}")
        else:
            print(f"  {issue} is isolated (no shared letters with other issues)")
            
        cluster_id += 1

# Count isolated issues (no chunks)
isolated_no_chunks = []
for issue in unique_issues_no_chunks:
    if not any(issue in cluster for cluster in clusters_no_chunks):
        isolated_no_chunks.append(issue)

print("\n" + "="*60)
print("CLUSTERING RESULTS (NO CHUNKS)")
print("="*60)

if clusters_no_chunks:
    print(f"Found {len(clusters_no_chunks)} clusters with multiple issues:")
    
    for i, cluster in enumerate(clusters_no_chunks, 1):
        cluster_list = sorted([str(g) for g in cluster])
        print(f"\nCluster {i} ({len(cluster)} issues):")
        print(f"  Issues: {', '.join(cluster_list)}")
        
        # Find all letters that appear in multiple issues within this cluster
        all_shared_pairs_no_chunks = []
        cluster_issues = list(cluster)
        
        for j in range(len(cluster_issues)):
            for k in range(j + 1, len(cluster_issues)):
                g1, g2 = cluster_issues[j], cluster_issues[k]
                shared = issue_to_letters_no_chunks[g1].intersection(issue_to_letters_no_chunks[g2])
                if shared:
                    all_shared_pairs_no_chunks.append(f"{g1}↔{g2}: {sorted(list(shared))}")
        
        if all_shared_pairs_no_chunks:
            print("  Shared letter pairs (sample of first 5):")
            for pair in all_shared_pairs_no_chunks[:5]:
                print(f"    {pair}")
            if len(all_shared_pairs_no_chunks) > 5:
                print(f"    ... and {len(all_shared_pairs_no_chunks) - 5} more pairs")
else:
    print("No clusters with multiple issues found.")

if isolated_no_chunks:
    print(f"\nIsolated issues ({len(isolated_no_chunks)}): {sorted([str(g) for g in isolated_no_chunks])}")

# Compare results
print("\nComparison:")
print(f"  With chunks: {len(clusters)} clusters, {len(isolated)} isolated")
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

# Create mapping from issue to cluster ID
issue_to_cluster = {}
for i, cluster in enumerate(clusters, 1):
    for issue in cluster:
        issue_to_cluster[issue] = f"Cluster {i}"
for issue in isolated:
    issue_to_cluster[issue] = "Isolated"

# Same for no-chunks clustering
issue_to_cluster_no_chunks = {}
for i, cluster in enumerate(clusters_no_chunks, 1):
    for issue in cluster:
        issue_to_cluster_no_chunks[issue] = f"Cluster {i}"
for issue in isolated_no_chunks:
    issue_to_cluster_no_chunks[issue] = "Isolated"

# Get the original issue order from full_sample to preserve in plots
original_issue_order = full_sample["issue"].unique().to_list()

# Add cluster information to full_sample
full_sample_with_clusters = full_sample.with_columns(
    cluster=pl.col("issue").map_elements(
        lambda g: issue_to_cluster.get(g, "Unknown"), 
        return_dtype=pl.Utf8
    )
)

# No-chunks version
full_sample_no_chunks_with_clusters = full_sample.filter(
    ~pl.col("filename").str.contains("chunk")
).with_columns(
    cluster=pl.col("issue").map_elements(
        lambda g: issue_to_cluster_no_chunks.get(g, "Unknown"), 
        return_dtype=pl.Utf8
    )
)

# Create the original heatmaps (preserved from original code)
# Original version - all cross-book letters
cross_book_heatmap = alt.Chart(full_sample.filter(pl.col("appears_in_other_books"))).mark_rect(height=8).encode(
    y=alt.Y("letter_with_class:O", title="Letters also in other books"),
    x=alt.X("issue"),
    color=alt.Color("other_books:N", legend=alt.Legend(
        title="Also appears in books",
        labelLimit=300,
        symbolLimit=50,
        labelExpr="split(datum.label, ' + ')"
    )),
    opacity=alt.value(0.9),
    tooltip=["letter_with_class", "issue", "other_books"]
).properties(
    title="Letters that appear in both votes and other books",
    height=alt.Step(12),
    width=800
).resolve_scale(
    color='independent'
)
cross_book_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# Create cluster background data for shading
cluster_background = create_cluster_background(issue_to_cluster)
print(f"Created cluster background data with {len(cluster_background) if cluster_background is not None else 0} rows")

# Create a plot showing ALL issues shaded by cluster (including those without characters)
if cluster_background is not None:
    # Sort issues by issue number
    all_issues_with_clusters = sorted(list(issue_to_cluster.keys()), key=lambda x: int(x) if x.isdigit() else float('inf'))
    
    # Create a dataframe with all issues and their cluster assignments
    issue_cluster_data = pl.DataFrame({
        "issue": all_issues_with_clusters,
        "cluster": [issue_to_cluster[issue] for issue in all_issues_with_clusters]
    })
    
    # Create a simple bar chart showing all issues colored by cluster
    all_issues_cluster_plot = alt.Chart(issue_cluster_data).mark_rect(
        height=20
    ).encode(
        x=alt.X("issue:N", 
            title="Issue",
            axis=alt.Axis(labelAngle=-45),
            sort=all_issues_with_clusters
        ),
        color=alt.Color("cluster:N", 
            scale=alt.Scale(scheme="set3"),
            legend=alt.Legend(title="Cluster Assignment")
        ),
        tooltip=["issue", "cluster"]
    ).properties(
        title="All Issues Shaded by Cluster Assignment (including issues without characters)",
        width=800,
        height=100
    )
    
    print(f"Created cluster overview plot with {len(all_issues_with_clusters)} issues")
else:
    all_issues_cluster_plot = None
    print("No cluster background data available for overview plot")

# Version with cluster background shading
if cluster_background is not None:
    # Create background layer - each issue gets colored by its cluster
    background_layer = alt.Chart(cluster_background).mark_rect(
            stroke='lightgray', 
            strokeWidth=0.5,
            strokeOpacity=0.5
        ).encode(
        x=alt.X("issue"),
        color=alt.Color("cluster:N", 
            scale=alt.Scale(scheme="tableau20"),
            legend=alt.Legend(title="Issue Cluster")
        ),
        opacity=alt.value(0.3)
    )
    
    # Create the main heatmap layer
    heatmap_layer = alt.Chart(full_sample.filter(pl.col("appears_in_other_books"))).mark_rect(height=8).encode(
        y=alt.Y("letter_with_class:O", title="Letters also in other books"),
        x=alt.X("issue"),
        color=alt.Color("other_books:N", legend=alt.Legend(
            title="Also appears in books",
            labelLimit=300,
            symbolLimit=50,
            labelExpr="split(datum.label, ' + ')"
        )),
        opacity=alt.value(0.9),
        tooltip=["letter_with_class", "issue", "other_books"]
    )
    
    # Layer them together
    cross_book_heatmap_shaded = alt.layer(
        background_layer,
        heatmap_layer
    ).properties(
        title="Letters that appear in both votes and other books (with cluster shading)",
        height=alt.Step(12),
        width=800
    ).resolve_scale(
        color='independent'
    ).configure_axis(
        grid=True, tickBand="extent"
    )
else:
    cross_book_heatmap_shaded = cross_book_heatmap

# ========== PRINTER-COLORED CROSS-BOOK HEATMAPS ==========
# Cross-book heatmap colored by printer (anon shows book name for detail)
cross_printer_heatmap = alt.Chart(full_sample.filter(pl.col("appears_in_other_books"))).mark_rect(height=8).encode(
    y=alt.Y("letter_with_class:O", title="Letters also in other books"),
    x=alt.X("issue"),
    color=alt.Color("other_printers_detailed:N", legend=alt.Legend(
        title="Also appears in printers",
        labelLimit=300,
        symbolLimit=50,
        labelExpr="split(datum.label, ' + ')"
    )),
    opacity=alt.value(0.9),
    tooltip=["letter_with_class", "issue", "other_printers_detailed", "other_books"]
).properties(
    title="Letters that appear in both votes and other books (colored by printer)",
    height=alt.Step(12),
    width=800
).resolve_scale(
    color='independent'
)
cross_printer_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# Shaded versions with cluster background
if cluster_background is not None:
    # Printer version with cluster shading (anon shows book name)
    heatmap_layer_printer = alt.Chart(full_sample.filter(pl.col("appears_in_other_books"))).mark_rect(height=8).encode(
        y=alt.Y("letter_with_class:O", title="Letters also in other books"),
        x=alt.X("issue"),
        color=alt.Color("other_printers_detailed:N", legend=alt.Legend(
            title="Also appears in printers",
            labelLimit=300,
            symbolLimit=50,
            labelExpr="split(datum.label, ' + ')"
        )),
        opacity=alt.value(0.9),
        tooltip=["letter_with_class", "issue", "other_printers_detailed", "other_books"]
    )

    cross_printer_heatmap_shaded = alt.layer(
        background_layer,
        heatmap_layer_printer
    ).properties(
        title="Letters in both votes and other books (by printer, with cluster shading)",
        height=alt.Step(12),
        width=800
    ).resolve_scale(
        color='independent'
    ).configure_axis(
        grid=True, tickBand="extent"
    )
else:
    cross_printer_heatmap_shaded = cross_printer_heatmap

# Non-chunk versions colored by printer (anon shows book name)
cross_printer_no_chunk_heatmap = alt.Chart(
    full_sample.filter(
        (pl.col("appears_in_other_books")) &
        (~pl.col("filename").str.contains("chunk"))
    )
).mark_rect(height=8).encode(
    y=alt.Y("letter_with_class:O", title="Letters from non-chunk files also in other books"),
    x=alt.X("issue"),
    color=alt.Color("other_printers_detailed:N", legend=alt.Legend(
        title="Also appears in printers",
        labelLimit=300,
        symbolLimit=50,
        labelExpr="split(datum.label, ' + ')"
    )),
    opacity=alt.value(0.9),
    tooltip=["letter_with_class", "issue", "other_printers_detailed", "other_books"]
).properties(
    title="Non-chunk letters in both votes and other books (colored by printer)",
    height=alt.Step(12),
    width=800
).resolve_scale(
    color='independent'
)
cross_printer_no_chunk_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# Version with only chunk letters
chunk_cross_book_heatmap = alt.Chart(
    full_sample.filter(
        (pl.col("appears_in_other_books")) & 
        (pl.col("filename").str.contains("chunk"))
    )
).mark_rect(height=8).encode(
    y=alt.Y("letter_with_class:O", title="Letters from chunk files also in other books"),
    x=alt.X("issue"),
    color=alt.Color("other_books:N", legend=alt.Legend(
        title="Also appears in books",
        labelLimit=300,
        symbolLimit=50,
        labelExpr="split(datum.label, ' + ')"
    )),
    opacity=alt.value(0.9),
    tooltip=["letter_with_class", "issue", "other_books"]
).properties(
    title="Letters from chunk files that appear in both votes and other books",
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
    x=alt.X("issue"),
    color=alt.Color("other_books:N", legend=alt.Legend(
        title="Also appears in books",
        labelLimit=300,
        symbolLimit=50,
        labelExpr="split(datum.label, ' + ')"
    )),
    opacity=alt.value(0.9),
    tooltip=["letter_with_class", "issue", "other_books"]
).properties(
    title="Letters from non-chunk files that appear in both votes and other books",
    height=alt.Step(12),
    width=800
).resolve_scale(
    color='independent'
)
no_chunk_cross_book_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# Same plot but with printer_bookname format and color-coded by printer
# Create a custom color scale where books from the same printer get different shades of the same hue
import colorsys

# Collect ALL unique values (including combined ones like "book1 + book2")
all_printer_bookname_values = set()
for val in letter_to_printer_bookname.values():
    if val != "votes only":
        all_printer_bookname_values.add(val)

# Categorize values by their printer(s)
single_printer_values = {}  # printer -> [values]
mixed_printer_values = []   # values with multiple different printers

for val in all_printer_bookname_values:
    items = val.split(" + ")
    if len(items) == 1:
        # Single printer_bookname
        printer = items[0].split("_")[0]
        if printer not in single_printer_values:
            single_printer_values[printer] = []
        single_printer_values[printer].append(val)
    else:
        # Multiple printer_booknames - check if all from same printer
        printers = set([item.split("_")[0] for item in items])
        if len(printers) == 1:
            # All from same printer - treat as single printer value
            printer = printers.pop()
            if printer not in single_printer_values:
                single_printer_values[printer] = []
            single_printer_values[printer].append(val)
        else:
            # Multiple different printers combined
            mixed_printer_values.append(val)

# Define base hues for each printer
printer_base_hues = {
    "everingham": 0.55,  # Blue
    "tbraddyll": 0.08,   # Orange/red
    "anon": 0.33,        # Green
    "unknown": 0.0,      # Red
}

color_domain = []
color_range = []

# Assign colors for single-printer values (different shades of same hue)
for printer in sorted(single_printer_values.keys()):
    values = sorted(single_printer_values[printer])
    base_hue = printer_base_hues.get(printer, 0.75)  # Default purple
    num_values = len(values)

    for i, val in enumerate(values):
        color_domain.append(val)
        # Vary saturation and lightness to create different shades
        if num_values == 1:
            saturation = 0.7
            lightness = 0.5
        else:
            # Spread across saturation and lightness range
            saturation = 0.4 + (0.5 * i / max(1, num_values - 1))
            lightness = 0.3 + (0.4 * i / max(1, num_values - 1))

        r, g, b = colorsys.hls_to_rgb(base_hue, lightness, saturation)
        color_hex = '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
        color_range.append(color_hex)

# Assign colors for mixed-printer combined values (use purple/magenta hues)
for i, val in enumerate(sorted(mixed_printer_values)):
    color_domain.append(val)
    # Use purple/magenta hues for combinations of different printers
    base_hue = 0.75  # Purple
    num_values = len(mixed_printer_values)
    if num_values == 1:
        saturation = 0.6
        lightness = 0.5
    else:
        saturation = 0.4 + (0.4 * i / max(1, num_values - 1))
        lightness = 0.35 + (0.35 * i / max(1, num_values - 1))

    r, g, b = colorsys.hls_to_rgb(base_hue, lightness, saturation)
    color_hex = '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
    color_range.append(color_hex)

# Add "votes only" color
color_domain.append("votes only")
color_range.append("#cccccc")

no_chunk_cross_book_heatmap_printer_bookname = alt.Chart(
    full_sample.filter(
        (pl.col("appears_in_other_books")) &
        (~pl.col("filename").str.contains("chunk"))
    )
).mark_rect(height=8).encode(
    y=alt.Y("letter_with_class:O", title="Letters from non-chunk files also in other books"),
    x=alt.X("issue"),
    color=alt.Color("other_printer_bookname:N",
        scale=alt.Scale(domain=color_domain, range=color_range),
        legend=alt.Legend(
            title="Also appears in (printer_bookname)",
            labelLimit=300,
            symbolLimit=50,
            labelExpr="split(datum.label, ' + ')"
        )
    ),
    opacity=alt.value(0.9),
    tooltip=["letter_with_class", "issue", "other_printer_bookname"]
).properties(
    title="Letters from non-chunk files that appear in both votes and other books (printer_bookname, color-coded by printer)",
    height=alt.Step(12),
    width=800
).resolve_scale(
    color='independent'
)
no_chunk_cross_book_heatmap_printer_bookname.configure_axis(
    grid=True, tickBand="extent"
)

# Create new versions using cluster information
# All data with clusters
clustered_cross_book_heatmap = alt.Chart(
    full_sample_with_clusters.filter(pl.col("appears_in_other_books"))
).mark_rect(height=8).encode(
    y=alt.Y("letter_with_class:O", title="Letters also in other books"),
    x=alt.X("issue"),
    color=alt.Color("other_books:N", legend=alt.Legend(
        title="Also appears in books",
        labelLimit=300,
        symbolLimit=50,
        labelExpr="split(datum.label, ' + ')"
    )),
    facet=alt.Facet("cluster:N", columns=2),
    opacity=alt.value(0.9),
    tooltip=["letter_with_class", "issue", "other_books", "cluster"]
).properties(
    title="Letters that appear in both votes and other books (by cluster)",
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
    x=alt.X("issue"),
    color=alt.Color("other_books:N", legend=alt.Legend(
        title="Also appears in books",
        labelLimit=300,
        symbolLimit=50,
        labelExpr="split(datum.label, ' + ')"
    )),
    facet=alt.Facet("cluster:N", columns=2),
    opacity=alt.value(0.9),
    tooltip=["letter_with_class", "issue", "other_books", "cluster"]
).properties(
    title="Letters from chunk files that appear in both votes and other books (by cluster)",
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
    x=alt.X("issue"),
    color=alt.Color("other_books:N", legend=alt.Legend(
        title="Also appears in books",
        labelLimit=300,
        symbolLimit=50,
        labelExpr="split(datum.label, ' + ')"
    )),
    facet=alt.Facet("cluster:N", columns=2),
    opacity=alt.value(0.9),
    tooltip=["letter_with_class", "issue", "other_books", "cluster"]
).properties(
    title="Letters from non-chunk files that appear in both votes and other books (by cluster)",
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
            if other_books != "votes only":
                # Split the combined book names and create a row for each book
                individual_books = [book.strip() for book in other_books.split(" + ")]
                for book in individual_books:
                    new_row = row.copy()
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
    x=alt.X("issue", sort=original_issue_order),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="greys"), legend=alt.Legend(title="Number of shared letters")),
    opacity=alt.value(0.9),
    tooltip=["individual_book", "issue", "count()"]
).properties(
    title="Damaged type presence in other books across votes issues",
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
    x=alt.X("issue", sort=original_issue_order),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="greys"), legend=alt.Legend(title="Number of shared letters")),
    opacity=alt.value(0.9),
    tooltip=["individual_book", "issue", "count()"]
).properties(
    title="Chunk damaged type presence in other books across votes issues",
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
    x=alt.X("issue", sort=original_issue_order, axis=alt.Axis(grid=True, gridOpacity=0.5)),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="blues"), legend=alt.Legend(title="Number of shared letters")),
    opacity=alt.value(0.9),
    tooltip=["individual_book", "issue", "count()"]
).properties(
    title="Non-chunk damaged type presence in other books across votes issues",
    height=alt.Step(15),
    width=800
).resolve_scale(
    color='independent'
)
agg_no_chunk_cross_book_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

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
    x=alt.X("issue", sort=original_issue_order),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="greys"), legend=alt.Legend(title="Number of shared letters")),
    facet=alt.Facet("cluster:N", columns=2),
    opacity=alt.value(0.9),
    tooltip=["individual_book", "issue", "count()", "cluster"]
).properties(
    title="Damaged type presence in other books across votes issues (by cluster)",
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
    x=alt.X("issue", sort=original_issue_order),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="greys"), legend=alt.Legend(title="Number of shared letters")),
    facet=alt.Facet("cluster:N", columns=2),
    opacity=alt.value(0.9),
    tooltip=["individual_book", "issue", "count()", "cluster"]
).properties(
    title="Chunk damaged type presence in other books across votes issues (by cluster)",
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
    x=alt.X("issue", sort=original_issue_order),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="greys"), legend=alt.Legend(title="Number of shared letters")),
    facet=alt.Facet("cluster:N", columns=2),
    opacity=alt.value(0.9),
    tooltip=["individual_book", "issue", "count()", "cluster"]
).properties(
    title="Non-chunk damaged type presence in other books across votes issues (by cluster)",
    height=alt.Step(15),
    width=400
).resolve_scale(
    color='independent'
)
agg_clustered_no_chunk_cross_book_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# ========== PRINTER-BASED AGGREGATED VERSIONS ==========
print("="*60)
print("CREATING AGGREGATED PRINTER Y-AXIS VERSIONS")
print("="*60)

# Create expanded data where each letter occurrence is duplicated for each printer it appears in
def expand_other_printers_data(df):
    """Expand the data so each printer gets its own row instead of being joined with +"""
    expanded_rows = []

    for row in df.iter_rows(named=True):
        if row["appears_in_other_books"]:
            other_printers = row["other_printers"]
            if other_printers != "votes only":
                # Split the combined printer names and create a row for each printer
                individual_printers = [printer.strip() for printer in other_printers.split(" + ")]
                for printer in individual_printers:
                    new_row = row.copy()
                    new_row["individual_printer"] = printer
                    expanded_rows.append(new_row)

    return pl.DataFrame(expanded_rows)

# Create expanded printer datasets for each filter condition
expanded_printers_all = expand_other_printers_data(full_sample.filter(pl.col("appears_in_other_books")))
expanded_printers_chunk = expand_other_printers_data(
    full_sample.filter(
        (pl.col("appears_in_other_books")) &
        (pl.col("filename").str.contains("chunk"))
    )
)
expanded_printers_no_chunk = expand_other_printers_data(
    full_sample.filter(
        (pl.col("appears_in_other_books")) &
        (~pl.col("filename").str.contains("chunk"))
    )
)

print(f"Expanded printer data sizes: all={len(expanded_printers_all)}, chunk={len(expanded_printers_chunk)}, no_chunk={len(expanded_printers_no_chunk)}")

# Aggregated printer version - all cross-book letters
agg_printer_heatmap = alt.Chart(expanded_printers_all).mark_rect(height=8).encode(
    y=alt.Y("individual_printer:O", title="Printers with shared letters", sort="ascending"),
    x=alt.X("issue", sort=original_issue_order),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="oranges"), legend=alt.Legend(title="Number of shared letters")),
    opacity=alt.value(0.9),
    tooltip=["individual_printer", "issue", "count()"]
).properties(
    title="Damaged type presence by printer across votes issues",
    height=alt.Step(20),
    width=800
).resolve_scale(
    color='independent'
)
agg_printer_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# Aggregated printer version with only chunk letters
agg_printer_chunk_heatmap = alt.Chart(expanded_printers_chunk).mark_rect(height=8).encode(
    y=alt.Y("individual_printer:O", title="Printers with shared chunk letters", sort="ascending"),
    x=alt.X("issue", sort=original_issue_order),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="oranges"), legend=alt.Legend(title="Number of shared letters")),
    opacity=alt.value(0.9),
    tooltip=["individual_printer", "issue", "count()"]
).properties(
    title="Chunk damaged type presence by printer across votes issues",
    height=alt.Step(20),
    width=800
).resolve_scale(
    color='independent'
)
agg_printer_chunk_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# Aggregated printer version without any chunk letters
agg_printer_no_chunk_heatmap = alt.Chart(expanded_printers_no_chunk).mark_rect(height=8).encode(
    y=alt.Y("individual_printer:O", title="Printers with shared non-chunk letters", sort="ascending", axis=alt.Axis(grid=True, gridOpacity=0.5)),
    x=alt.X("issue", sort=original_issue_order, axis=alt.Axis(grid=True, gridOpacity=0.5)),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="purples"), legend=alt.Legend(title="Number of shared letters")),
    opacity=alt.value(0.9),
    tooltip=["individual_printer", "issue", "count()"]
).properties(
    title="Non-chunk damaged type presence by printer across votes issues",
    height=alt.Step(20),
    width=800
).resolve_scale(
    color='independent'
)
agg_printer_no_chunk_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# Create expanded printer datasets for clustered versions
expanded_printers_clustered_all = expand_other_printers_data(full_sample_with_clusters.filter(pl.col("appears_in_other_books")))
expanded_printers_clustered_chunk = expand_other_printers_data(
    full_sample_with_clusters.filter(
        (pl.col("appears_in_other_books")) &
        (pl.col("filename").str.contains("chunk"))
    )
)
expanded_printers_clustered_no_chunk = expand_other_printers_data(
    full_sample_no_chunks_with_clusters.filter(pl.col("appears_in_other_books"))
)

# Aggregated clustered printer versions
agg_clustered_printer_heatmap = alt.Chart(expanded_printers_clustered_all).mark_rect(height=8).encode(
    y=alt.Y("individual_printer:O", title="Printers with shared letters", sort="ascending"),
    x=alt.X("issue", sort=original_issue_order),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="oranges"), legend=alt.Legend(title="Number of shared letters")),
    facet=alt.Facet("cluster:N", columns=2),
    opacity=alt.value(0.9),
    tooltip=["individual_printer", "issue", "count()", "cluster"]
).properties(
    title="Damaged type presence by printer across votes issues (by cluster)",
    height=alt.Step(20),
    width=400
).resolve_scale(
    color='independent'
)
agg_clustered_printer_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

agg_clustered_printer_chunk_heatmap = alt.Chart(expanded_printers_clustered_chunk).mark_rect(height=8).encode(
    y=alt.Y("individual_printer:O", title="Printers with shared chunk letters", sort="ascending"),
    x=alt.X("issue", sort=original_issue_order),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="oranges"), legend=alt.Legend(title="Number of shared letters")),
    facet=alt.Facet("cluster:N", columns=2),
    opacity=alt.value(0.9),
    tooltip=["individual_printer", "issue", "count()", "cluster"]
).properties(
    title="Chunk damaged type presence by printer across votes issues (by cluster)",
    height=alt.Step(20),
    width=400
).resolve_scale(
    color='independent'
)
agg_clustered_printer_chunk_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

agg_clustered_printer_no_chunk_heatmap = alt.Chart(expanded_printers_clustered_no_chunk).mark_rect(height=8).encode(
    y=alt.Y("individual_printer:O", title="Printers with shared non-chunk letters", sort="ascending"),
    x=alt.X("issue", sort=original_issue_order),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="purples"), legend=alt.Legend(title="Number of shared letters")),
    facet=alt.Facet("cluster:N", columns=2),
    opacity=alt.value(0.9),
    tooltip=["individual_printer", "issue", "count()", "cluster"]
).properties(
    title="Non-chunk damaged type presence by printer across votes issues (by cluster)",
    height=alt.Step(20),
    width=400
).resolve_scale(
    color='independent'
)
agg_clustered_printer_no_chunk_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# ========== DETAILED PRINTER-BASED AGGREGATED VERSIONS (anon with book names) ==========
print("="*60)
print("CREATING DETAILED PRINTER Y-AXIS VERSIONS (anon with book names)")
print("="*60)

# Create expanded data where each letter occurrence is duplicated for each detailed printer it appears in
def expand_other_printers_detailed_data(df):
    """Expand the data so each detailed printer gets its own row (anon shows book name)"""
    expanded_rows = []

    for row in df.iter_rows(named=True):
        if row["appears_in_other_books"]:
            other_printers_detailed = row["other_printers_detailed"]
            if other_printers_detailed != "votes only":
                # Split the combined printer names and create a row for each printer
                individual_printers = [printer.strip() for printer in other_printers_detailed.split(" + ")]
                for printer in individual_printers:
                    new_row = row.copy()
                    new_row["individual_printer_detailed"] = printer
                    expanded_rows.append(new_row)

    return pl.DataFrame(expanded_rows)

# Create expanded detailed printer datasets for each filter condition
expanded_printers_detailed_all = expand_other_printers_detailed_data(full_sample.filter(pl.col("appears_in_other_books")))
expanded_printers_detailed_chunk = expand_other_printers_detailed_data(
    full_sample.filter(
        (pl.col("appears_in_other_books")) &
        (pl.col("filename").str.contains("chunk"))
    )
)
expanded_printers_detailed_no_chunk = expand_other_printers_detailed_data(
    full_sample.filter(
        (pl.col("appears_in_other_books")) &
        (~pl.col("filename").str.contains("chunk"))
    )
)

print(f"Expanded detailed printer data sizes: all={len(expanded_printers_detailed_all)}, chunk={len(expanded_printers_detailed_chunk)}, no_chunk={len(expanded_printers_detailed_no_chunk)}")

# Aggregated detailed printer version - all cross-book letters
agg_printer_detailed_heatmap = alt.Chart(expanded_printers_detailed_all).mark_rect(height=8).encode(
    y=alt.Y("individual_printer_detailed:O", title="Printers (anon with book) with shared letters", sort="ascending"),
    x=alt.X("issue", sort=original_issue_order),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="oranges"), legend=alt.Legend(title="Number of shared letters")),
    opacity=alt.value(0.9),
    tooltip=["individual_printer_detailed", "issue", "count()"]
).properties(
    title="Damaged type presence by printer (anon with book) across votes issues",
    height=alt.Step(15),
    width=800
).resolve_scale(
    color='independent'
)
agg_printer_detailed_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# Aggregated detailed printer version with only chunk letters
agg_printer_detailed_chunk_heatmap = alt.Chart(expanded_printers_detailed_chunk).mark_rect(height=8).encode(
    y=alt.Y("individual_printer_detailed:O", title="Printers (anon with book) with shared chunk letters", sort="ascending"),
    x=alt.X("issue", sort=original_issue_order),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="oranges"), legend=alt.Legend(title="Number of shared letters")),
    opacity=alt.value(0.9),
    tooltip=["individual_printer_detailed", "issue", "count()"]
).properties(
    title="Chunk damaged type presence by printer (anon with book) across votes issues",
    height=alt.Step(15),
    width=800
).resolve_scale(
    color='independent'
)
agg_printer_detailed_chunk_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# Aggregated detailed printer version without any chunk letters
agg_printer_detailed_no_chunk_heatmap = alt.Chart(expanded_printers_detailed_no_chunk).mark_rect(height=8).encode(
    y=alt.Y("individual_printer_detailed:O", title="Printers (anon with book) with shared non-chunk letters", sort="ascending", axis=alt.Axis(grid=True, gridOpacity=0.5)),
    x=alt.X("issue", sort=original_issue_order, axis=alt.Axis(grid=True, gridOpacity=0.5)),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="teals"), legend=alt.Legend(title="Number of shared letters")),
    opacity=alt.value(0.9),
    tooltip=["individual_printer_detailed", "issue", "count()"]
).properties(
    title="Non-chunk damaged type presence by printer (anon with book) across votes issues",
    height=alt.Step(15),
    width=800
).resolve_scale(
    color='independent'
)
agg_printer_detailed_no_chunk_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# Create expanded detailed printer datasets for clustered versions
expanded_printers_detailed_clustered_all = expand_other_printers_detailed_data(full_sample_with_clusters.filter(pl.col("appears_in_other_books")))
expanded_printers_detailed_clustered_chunk = expand_other_printers_detailed_data(
    full_sample_with_clusters.filter(
        (pl.col("appears_in_other_books")) &
        (pl.col("filename").str.contains("chunk"))
    )
)
expanded_printers_detailed_clustered_no_chunk = expand_other_printers_detailed_data(
    full_sample_no_chunks_with_clusters.filter(pl.col("appears_in_other_books"))
)

# Aggregated clustered detailed printer versions
agg_clustered_printer_detailed_heatmap = alt.Chart(expanded_printers_detailed_clustered_all).mark_rect(height=8).encode(
    y=alt.Y("individual_printer_detailed:O", title="Printers (anon with book) with shared letters", sort="ascending"),
    x=alt.X("issue", sort=original_issue_order),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="oranges"), legend=alt.Legend(title="Number of shared letters")),
    facet=alt.Facet("cluster:N", columns=2),
    opacity=alt.value(0.9),
    tooltip=["individual_printer_detailed", "issue", "count()", "cluster"]
).properties(
    title="Damaged type presence by printer (anon with book) across votes issues (by cluster)",
    height=alt.Step(15),
    width=400
).resolve_scale(
    color='independent'
)
agg_clustered_printer_detailed_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

agg_clustered_printer_detailed_chunk_heatmap = alt.Chart(expanded_printers_detailed_clustered_chunk).mark_rect(height=8).encode(
    y=alt.Y("individual_printer_detailed:O", title="Printers (anon with book) with shared chunk letters", sort="ascending"),
    x=alt.X("issue", sort=original_issue_order),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="oranges"), legend=alt.Legend(title="Number of shared letters")),
    facet=alt.Facet("cluster:N", columns=2),
    opacity=alt.value(0.9),
    tooltip=["individual_printer_detailed", "issue", "count()", "cluster"]
).properties(
    title="Chunk damaged type presence by printer (anon with book) across votes issues (by cluster)",
    height=alt.Step(15),
    width=400
).resolve_scale(
    color='independent'
)
agg_clustered_printer_detailed_chunk_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

agg_clustered_printer_detailed_no_chunk_heatmap = alt.Chart(expanded_printers_detailed_clustered_no_chunk).mark_rect(height=8).encode(
    y=alt.Y("individual_printer_detailed:O", title="Printers (anon with book) with shared non-chunk letters", sort="ascending"),
    x=alt.X("issue", sort=original_issue_order),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="teals"), legend=alt.Legend(title="Number of shared letters")),
    facet=alt.Facet("cluster:N", columns=2),
    opacity=alt.value(0.9),
    tooltip=["individual_printer_detailed", "issue", "count()", "cluster"]
).properties(
    title="Non-chunk damaged type presence by printer (anon with book) across votes issues (by cluster)",
    height=alt.Step(15),
    width=400
).resolve_scale(
    color='independent'
)
agg_clustered_printer_detailed_no_chunk_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# Save the new clustered versions
clustered_cross_book_heatmap.save("votes_clustered_crossbook_heatmap_latex_bigletter.png", scale_factor=2)
if len(full_sample_with_clusters.filter((pl.col("appears_in_other_books")) & (pl.col("filename").str.contains("chunk")))) > 0:
    clustered_chunk_cross_book_heatmap.save("votes_clustered_chunk_crossbook_heatmap_latex_bigletter.png", scale_factor=2)
else:
    print("Skipping clustered_chunk_cross_book_heatmap (no data)")
clustered_no_chunk_cross_book_heatmap.save("votes_clustered_nochunk_crossbook_heatmap_latex_bigletter.png", scale_factor=2)

# Save the new aggregated versions
if len(expanded_all) > 0:
    agg_cross_book_heatmap.save("votes_agg_crossbook_heatmap_latex_bigletter.png", scale_factor=2)
else:
    print("Skipping agg_cross_book_heatmap (no data)")
    
if len(expanded_chunk) > 0:
    agg_chunk_cross_book_heatmap.save("votes_agg_chunk_crossbook_heatmap_latex_bigletter.png", scale_factor=2)
else:
    print("Skipping agg_chunk_cross_book_heatmap (no data)")
    
if len(expanded_no_chunk) > 0:
    agg_no_chunk_cross_book_heatmap.save("votes_agg_nochunk_crossbook_heatmap_latex_bigletter.png", scale_factor=2)
else:
    print("Skipping agg_no_chunk_cross_book_heatmap (no data)")
    
if len(expanded_clustered_all) > 0:
    agg_clustered_cross_book_heatmap.save("votes_agg_clustered_crossbook_heatmap_latex_bigletter.png", scale_factor=2)
else:
    print("Skipping agg_clustered_cross_book_heatmap (no data)")
    
if len(expanded_clustered_chunk) > 0:
    agg_clustered_chunk_cross_book_heatmap.save("votes_agg_clustered_chunk_crossbook_heatmap_latex_bigletter.png", scale_factor=2)
else:
    print("Skipping agg_clustered_chunk_cross_book_heatmap (no data)")
    
if len(expanded_clustered_no_chunk) > 0:
    agg_clustered_no_chunk_cross_book_heatmap.save("votes_agg_clustered_nochunk_crossbook_heatmap_latex_bigletter.png", scale_factor=2)
else:
    print("Skipping agg_clustered_no_chunk_cross_book_heatmap (no data)")

# Save the new printer-based aggregated versions
print("="*60)
print("SAVING PRINTER-BASED PLOTS")
print("="*60)

if len(expanded_printers_all) > 0:
    agg_printer_heatmap.save("votes_agg_printer_heatmap_latex_bigletter.png", scale_factor=2)
    print("Saved votes_agg_printer_heatmap_latex_bigletter.png")
else:
    print("Skipping agg_printer_heatmap (no data)")

if len(expanded_printers_chunk) > 0:
    agg_printer_chunk_heatmap.save("votes_agg_printer_chunk_heatmap_latex_bigletter.png", scale_factor=2)
    print("Saved votes_agg_printer_chunk_heatmap_latex_bigletter.png")
else:
    print("Skipping agg_printer_chunk_heatmap (no data)")

if len(expanded_printers_no_chunk) > 0:
    agg_printer_no_chunk_heatmap.save("votes_agg_printer_nochunk_heatmap_latex_bigletter.png", scale_factor=2)
    print("Saved votes_agg_printer_nochunk_heatmap_latex_bigletter.png")
else:
    print("Skipping agg_printer_no_chunk_heatmap (no data)")

if len(expanded_printers_clustered_all) > 0:
    agg_clustered_printer_heatmap.save("votes_agg_clustered_printer_heatmap_latex_bigletter.png", scale_factor=2)
    print("Saved votes_agg_clustered_printer_heatmap_latex_bigletter.png")
else:
    print("Skipping agg_clustered_printer_heatmap (no data)")

if len(expanded_printers_clustered_chunk) > 0:
    agg_clustered_printer_chunk_heatmap.save("votes_agg_clustered_printer_chunk_heatmap_latex_bigletter.png", scale_factor=2)
    print("Saved votes_agg_clustered_printer_chunk_heatmap_latex_bigletter.png")
else:
    print("Skipping agg_clustered_printer_chunk_heatmap (no data)")

if len(expanded_printers_clustered_no_chunk) > 0:
    agg_clustered_printer_no_chunk_heatmap.save("votes_agg_clustered_printer_nochunk_heatmap_latex_bigletter.png", scale_factor=2)
    print("Saved votes_agg_clustered_printer_nochunk_heatmap_latex_bigletter.png")
else:
    print("Skipping agg_clustered_printer_no_chunk_heatmap (no data)")

# Save the new detailed printer-based aggregated versions (anon with book names)
print("="*60)
print("SAVING DETAILED PRINTER-BASED PLOTS (anon with book names)")
print("="*60)

if len(expanded_printers_detailed_all) > 0:
    agg_printer_detailed_heatmap.save("votes_agg_printer_detailed_heatmap_latex_bigletter.png", scale_factor=2)
    print("Saved votes_agg_printer_detailed_heatmap_latex_bigletter.png")
else:
    print("Skipping agg_printer_detailed_heatmap (no data)")

if len(expanded_printers_detailed_chunk) > 0:
    agg_printer_detailed_chunk_heatmap.save("votes_agg_printer_detailed_chunk_heatmap_latex_bigletter.png", scale_factor=2)
    print("Saved votes_agg_printer_detailed_chunk_heatmap_latex_bigletter.png")
else:
    print("Skipping agg_printer_detailed_chunk_heatmap (no data)")

if len(expanded_printers_detailed_no_chunk) > 0:
    agg_printer_detailed_no_chunk_heatmap.save("votes_agg_printer_detailed_nochunk_heatmap_latex_bigletter.png", scale_factor=2)
    print("Saved votes_agg_printer_detailed_nochunk_heatmap_latex_bigletter.png")
else:
    print("Skipping agg_printer_detailed_no_chunk_heatmap (no data)")

if len(expanded_printers_detailed_clustered_all) > 0:
    agg_clustered_printer_detailed_heatmap.save("votes_agg_clustered_printer_detailed_heatmap_latex_bigletter.png", scale_factor=2)
    print("Saved votes_agg_clustered_printer_detailed_heatmap_latex_bigletter.png")
else:
    print("Skipping agg_clustered_printer_detailed_heatmap (no data)")

if len(expanded_printers_detailed_clustered_chunk) > 0:
    agg_clustered_printer_detailed_chunk_heatmap.save("votes_agg_clustered_printer_detailed_chunk_heatmap_latex_bigletter.png", scale_factor=2)
    print("Saved votes_agg_clustered_printer_detailed_chunk_heatmap_latex_bigletter.png")
else:
    print("Skipping agg_clustered_printer_detailed_chunk_heatmap (no data)")

if len(expanded_printers_detailed_clustered_no_chunk) > 0:
    agg_clustered_printer_detailed_no_chunk_heatmap.save("votes_agg_clustered_printer_detailed_nochunk_heatmap_latex_bigletter.png", scale_factor=2)
    print("Saved votes_agg_clustered_printer_detailed_nochunk_heatmap_latex_bigletter.png")
else:
    print("Skipping agg_clustered_printer_detailed_no_chunk_heatmap (no data)")


# Get unique letters
unique_letters = full_sample["letter_with_class"].unique().sort().to_list()

# Create co-occurrence matrix data
adjacency_data = []

for i, letter_i in enumerate(unique_letters):
    for j, letter_j in enumerate(unique_letters):
        # Only include lower triangular part (i >= j)
        if i >= j:
            # Get issues for each letter
            filter_i = full_sample.filter(pl.col("letter_with_class") == letter_i)
            filter_j = full_sample.filter(pl.col("letter_with_class") == letter_j)
            
            issues_with_i = set(filter_i["issue"].unique().to_list())
            issues_with_j = set(filter_j["issue"].unique().to_list())
            
            # Count issues where both letters appear
            common_issues = len(issues_with_i.intersection(issues_with_j))
            
            adjacency_data.append({
                "letter_i": letter_i,
                "letter_j": letter_j,
                "common_issues": common_issues
            })

# Convert to DataFrame
adjacency_df = pl.DataFrame(adjacency_data)

# Create the lower triangular adjacency matrix heatmap
lower_triangle_heatmap = alt.Chart(adjacency_df).mark_rect().encode(
    x=alt.X("letter_j:O", title="Letter"),
    y=alt.Y("letter_i:O", title="Letter"),
    color=alt.Color("common_issues:Q", scale=alt.Scale(scheme="viridis")),
    tooltip=["letter_i", "letter_j", "common_issues"]
).properties(
    title="Letter Co-occurrence Adjacency Matrix",
    width=400,
    height=400
)

lower_triangle_heatmap.configure_axis(
    grid=True, tickBand="extent"
)

# Create versions that show ALL issues from the issue map (including those without data)
print("="*60)
print("CREATING COMPLETE DOMAIN VERSIONS (ALL ISSUES)")
print("="*60)

# Get all issues from the issue map and convert to strings to match the enum (format to 2 digits with leading zeros)
all_issues_from_map = [f"{i:02d}" for i in sorted(issue_map["issue"].unique().to_list())]
print(f"Total issues in issue map: {len(all_issues_from_map)}")
print(f"Issues with data: {len(full_sample['issue'].unique())}")
missing_issues = set(all_issues_from_map) - set(full_sample['issue'].unique().to_list())
print(f"Issues without data: {len(missing_issues)}")
if missing_issues:
    print(f"Missing issues: {sorted(list(missing_issues))}")

# Create complete domain versions of key plots
# Complete domain heatmap
heatmap_complete = alt.Chart(full_sample).mark_rect(height=5).encode(
    y=alt.Y("letter_with_class:O", title="letter type"),
    x=alt.X("issue", scale=alt.Scale(domain=all_issues_from_map)),
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
    tooltip=["letter_with_class", "issue", "other_books", "appears_in_other_books"]
).properties(
    title="Letter heatmap with complete issue domain (shows data gaps)",
    height=alt.Step(10),
    width=800
).resolve_scale(
    color='independent'
)

# Complete domain cross-book heatmap
cross_book_heatmap_complete = alt.Chart(full_sample.filter(pl.col("appears_in_other_books"))).mark_rect(height=8).encode(
    y=alt.Y("letter_with_class:O", title="Letters also in other books"),
    x=alt.X("issue", scale=alt.Scale(domain=all_issues_from_map)),
    color=alt.Color("other_books:N", legend=alt.Legend(
        title="Also appears in books",
        labelLimit=300,
        symbolLimit=50,
        labelExpr="split(datum.label, ' + ')"
    )),
    opacity=alt.value(0.9),
    tooltip=["letter_with_class", "issue", "other_books"]
).properties(
    title="Cross-book letters across all of votes issues",
    height=alt.Step(12),
    width=800
).resolve_scale(
    color='independent'
)

# Complete domain aggregated cross-book heatmap
agg_cross_book_heatmap_complete = alt.Chart(expanded_all).mark_rect(height=8).encode(
    y=alt.Y("individual_book:O", title="Other books with shared letters", sort="ascending"),
    x=alt.X("issue", sort=all_issues_from_map, scale=alt.Scale(domain=all_issues_from_map)),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="greys"), legend=alt.Legend(title="Number of shared letters")),
    opacity=alt.value(0.9),
    tooltip=["individual_book", "issue", "count()"]
).properties(
    title="Damaged type presence across all of votes issues",
    height=alt.Step(15),
    width=800
).resolve_scale(
    color='independent'
)

# Complete domain aggregated no-chunk cross-book heatmap  
agg_no_chunk_cross_book_heatmap_complete = alt.Chart(expanded_no_chunk).mark_rect(height=8).encode(
    y=alt.Y("individual_book:O", title="Other books with shared non-chunk letters", sort="ascending", axis=alt.Axis(grid=True, gridOpacity=0.5)),
    x=alt.X("issue", sort=all_issues_from_map, scale=alt.Scale(domain=all_issues_from_map), axis=alt.Axis(grid=True, gridOpacity=0.5)),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="blues"), legend=alt.Legend(title="Number of shared letters")),
    opacity=alt.value(0.9),
    tooltip=["individual_book", "issue", "count()"]
).properties(
    title="Non-chunk damaged type presence across all of votes issues",
    height=alt.Step(15),
    width=800
).resolve_scale(
    color='independent'
)

# Shaded version with cluster background
if cluster_background is not None:
    # Create background layer for cluster shading
    background_layer_agg_no_chunk = alt.Chart(cluster_background).mark_rect().encode(
        x=alt.X("issue", sort=all_issues_from_map, scale=alt.Scale(domain=all_issues_from_map)),
        color=alt.Color("cluster:N", 
            scale=alt.Scale(scheme="set3"),
            legend=alt.Legend(title="Issue Cluster")
        ),
        opacity=alt.value(0.3)
    )
    
    # Create the main heatmap layer
    heatmap_layer_agg_no_chunk = alt.Chart(expanded_no_chunk).mark_rect(height=8).encode(
        y=alt.Y("individual_book:O", title="Other books with shared non-chunk letters", sort="ascending", axis=alt.Axis(grid=True, gridOpacity=0.5)),
        x=alt.X("issue", sort=all_issues_from_map, scale=alt.Scale(domain=all_issues_from_map), axis=alt.Axis(grid=True, gridOpacity=0.5)),
        color=alt.Color("count():Q", scale=alt.Scale(scheme="warmgreys"), legend=alt.Legend(title="Number of shared letters")),
        opacity=alt.value(0.9),
        tooltip=["individual_book", "issue", "count()"]
    )
    
    # Layer them together
    agg_no_chunk_cross_book_heatmap_complete_shaded = alt.layer(
        background_layer_agg_no_chunk,
        heatmap_layer_agg_no_chunk
    ).properties(
        title="Non-chunk damaged type presence across all of votes issues (with cluster shading)",
        height=alt.Step(15),
        width=800
    ).resolve_scale(
        color='independent'
    )
else:
    agg_no_chunk_cross_book_heatmap_complete_shaded = agg_no_chunk_cross_book_heatmap_complete

# ========== COMPLETE DOMAIN PRINTER VERSIONS ==========
# Complete domain aggregated printer heatmap
agg_printer_heatmap_complete = alt.Chart(expanded_printers_all).mark_rect(height=8).encode(
    y=alt.Y("individual_printer:O", title="Printers with shared letters", sort="ascending"),
    x=alt.X("issue", sort=all_issues_from_map, scale=alt.Scale(domain=all_issues_from_map)),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="oranges"), legend=alt.Legend(title="Number of shared letters")),
    opacity=alt.value(0.9),
    tooltip=["individual_printer", "issue", "count()"]
).properties(
    title="Damaged type presence by printer across all votes issues",
    height=alt.Step(20),
    width=800
).resolve_scale(
    color='independent'
)

# Complete domain aggregated no-chunk printer heatmap
agg_printer_no_chunk_heatmap_complete = alt.Chart(expanded_printers_no_chunk).mark_rect(height=8).encode(
    y=alt.Y("individual_printer:O", title="Printers with shared non-chunk letters", sort="ascending", axis=alt.Axis(grid=True, gridOpacity=0.5)),
    x=alt.X("issue", sort=all_issues_from_map, scale=alt.Scale(domain=all_issues_from_map), axis=alt.Axis(grid=True, gridOpacity=0.5)),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="purples"), legend=alt.Legend(title="Number of shared letters")),
    opacity=alt.value(0.9),
    tooltip=["individual_printer", "issue", "count()"]
).properties(
    title="Non-chunk damaged type presence by printer across all votes issues",
    height=alt.Step(20),
    width=800
).resolve_scale(
    color='independent'
)

# Shaded version with cluster background for printer heatmap
if cluster_background is not None:
    # Create background layer for cluster shading
    background_layer_agg_printer = alt.Chart(cluster_background).mark_rect().encode(
        x=alt.X("issue", sort=all_issues_from_map, scale=alt.Scale(domain=all_issues_from_map)),
        color=alt.Color("cluster:N",
            scale=alt.Scale(scheme="set3"),
            legend=alt.Legend(title="Issue Cluster")
        ),
        opacity=alt.value(0.3)
    )

    # Create the main heatmap layer
    heatmap_layer_agg_printer = alt.Chart(expanded_printers_no_chunk).mark_rect(height=8).encode(
        y=alt.Y("individual_printer:O", title="Printers with shared non-chunk letters", sort="ascending", axis=alt.Axis(grid=True, gridOpacity=0.5)),
        x=alt.X("issue", sort=all_issues_from_map, scale=alt.Scale(domain=all_issues_from_map), axis=alt.Axis(grid=True, gridOpacity=0.5)),
        color=alt.Color("count():Q", scale=alt.Scale(scheme="reds"), legend=alt.Legend(title="Number of shared letters")),
        opacity=alt.value(0.9),
        tooltip=["individual_printer", "issue", "count()"]
    )

    # Layer them together
    agg_printer_no_chunk_heatmap_complete_shaded = alt.layer(
        background_layer_agg_printer,
        heatmap_layer_agg_printer
    ).properties(
        title="Non-chunk damaged type presence by printer across all votes issues (with cluster shading)",
        height=alt.Step(20),
        width=800
    ).resolve_scale(
        color='independent'
    )
else:
    agg_printer_no_chunk_heatmap_complete_shaded = agg_printer_no_chunk_heatmap_complete

# ========== COMPLETE DOMAIN DETAILED PRINTER VERSIONS (anon with book names) ==========
# Complete domain aggregated detailed printer heatmap
agg_printer_detailed_heatmap_complete = alt.Chart(expanded_printers_detailed_all).mark_rect(height=8).encode(
    y=alt.Y("individual_printer_detailed:O", title="Printers (anon with book) with shared letters", sort="ascending"),
    x=alt.X("issue", sort=all_issues_from_map, scale=alt.Scale(domain=all_issues_from_map)),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="oranges"), legend=alt.Legend(title="Number of shared letters")),
    opacity=alt.value(0.9),
    tooltip=["individual_printer_detailed", "issue", "count()"]
).properties(
    title="Damaged type presence by printer (anon with book) across all votes issues",
    height=alt.Step(15),
    width=800
).resolve_scale(
    color='independent'
)

# Complete domain aggregated no-chunk detailed printer heatmap
agg_printer_detailed_no_chunk_heatmap_complete = alt.Chart(expanded_printers_detailed_no_chunk).mark_rect(height=8).encode(
    y=alt.Y("individual_printer_detailed:O", title="Printers (anon with book) with shared non-chunk letters", sort="ascending", axis=alt.Axis(grid=True, gridOpacity=0.5)),
    x=alt.X("issue", sort=all_issues_from_map, scale=alt.Scale(domain=all_issues_from_map), axis=alt.Axis(grid=True, gridOpacity=0.5)),
    color=alt.Color("count():Q", scale=alt.Scale(scheme="teals"), legend=alt.Legend(title="Number of shared letters")),
    opacity=alt.value(0.9),
    tooltip=["individual_printer_detailed", "issue", "count()"]
).properties(
    title="Non-chunk damaged type presence by printer (anon with book) across all votes issues",
    height=alt.Step(15),
    width=800
).resolve_scale(
    color='independent'
)

# Shaded version with cluster background for detailed printer heatmap
if cluster_background is not None:
    # Create background layer for cluster shading
    background_layer_agg_printer_detailed = alt.Chart(cluster_background).mark_rect().encode(
        x=alt.X("issue", sort=all_issues_from_map, scale=alt.Scale(domain=all_issues_from_map)),
        color=alt.Color("cluster:N",
            scale=alt.Scale(scheme="set3"),
            legend=alt.Legend(title="Issue Cluster")
        ),
        opacity=alt.value(0.3)
    )

    # Create the main heatmap layer
    heatmap_layer_agg_printer_detailed = alt.Chart(expanded_printers_detailed_no_chunk).mark_rect(height=8).encode(
        y=alt.Y("individual_printer_detailed:O", title="Printers (anon with book) with shared non-chunk letters", sort="ascending", axis=alt.Axis(grid=True, gridOpacity=0.5)),
        x=alt.X("issue", sort=all_issues_from_map, scale=alt.Scale(domain=all_issues_from_map), axis=alt.Axis(grid=True, gridOpacity=0.5)),
        color=alt.Color("count():Q", scale=alt.Scale(scheme="reds"), legend=alt.Legend(title="Number of shared letters")),
        opacity=alt.value(0.9),
        tooltip=["individual_printer_detailed", "issue", "count()"]
    )

    # Layer them together
    agg_printer_detailed_no_chunk_heatmap_complete_shaded = alt.layer(
        background_layer_agg_printer_detailed,
        heatmap_layer_agg_printer_detailed
    ).properties(
        title="Non-chunk damaged type presence by printer (anon with book) across all votes issues (with cluster shading)",
        height=alt.Step(15),
        width=800
    ).resolve_scale(
        color='independent'
    )
else:
    agg_printer_detailed_no_chunk_heatmap_complete_shaded = agg_printer_detailed_no_chunk_heatmap_complete

print("Created complete domain versions of key plots")

# save fig
images_l.save("votes_letters_latex_bigletter.png", scale_factor=2)
heatmap.save("votes_heatmap_latex_bigletter.png", scale_factor=2)
cross_book_heatmap.save("votes_crossbook_heatmap_latex_bigletter.png", scale_factor=2)
if cluster_background is not None:
    cross_book_heatmap_shaded.save("votes_crossbook_heatmap_shaded_latex_bigletter.png", scale_factor=2)
    if all_issues_cluster_plot is not None:
        all_issues_cluster_plot.save("votes_all_issues_cluster_overview_latex_bigletter.png", scale_factor=2)
chunk_cross_book_heatmap.save("votes_chunk_crossbook_heatmap_latex_bigletter.png", scale_factor=2)
no_chunk_cross_book_heatmap.save("votes_nochunk_crossbook_heatmap_latex_bigletter.png", scale_factor=2)
no_chunk_cross_book_heatmap_printer_bookname.save("votes_nochunk_crossbook_heatmap_printer_bookname_latex_bigletter.png", scale_factor=2)
print("Saved votes_nochunk_crossbook_heatmap_printer_bookname_latex_bigletter.png")

# Save printer-colored cross-book heatmaps (anon shows book name)
cross_printer_heatmap.save("votes_cross_printer_heatmap_latex_bigletter.png", scale_factor=2)
print("Saved votes_cross_printer_heatmap_latex_bigletter.png")
cross_printer_no_chunk_heatmap.save("votes_cross_printer_nochunk_heatmap_latex_bigletter.png", scale_factor=2)
print("Saved votes_cross_printer_nochunk_heatmap_latex_bigletter.png")
if cluster_background is not None:
    cross_printer_heatmap_shaded.save("votes_cross_printer_heatmap_shaded_latex_bigletter.png", scale_factor=2)
    print("Saved votes_cross_printer_heatmap_shaded_latex_bigletter.png")
lower_triangle_heatmap.save("votes_adjacency_latex_bigletter.png", scale_factor=2)

# Save complete domain versions
heatmap_complete.save("votes_heatmap_complete_domain_latex_bigletter.png", scale_factor=2)
cross_book_heatmap_complete.save("votes_crossbook_complete_domain_latex_bigletter.png", scale_factor=2)
agg_cross_book_heatmap_complete.save("votes_agg_crossbook_complete_domain_latex_bigletter.png", scale_factor=2)
agg_no_chunk_cross_book_heatmap_complete.save("votes_agg_nochunk_crossbook_complete_domain_latex_bigletter.png", scale_factor=2)
if cluster_background is not None:
    agg_no_chunk_cross_book_heatmap_complete_shaded.save("votes_agg_nochunk_crossbook_complete_domain_shaded_latex_bigletter.png", scale_factor=2)

# Save complete domain printer versions
if len(expanded_printers_all) > 0:
    agg_printer_heatmap_complete.save("votes_agg_printer_complete_domain_latex_bigletter.png", scale_factor=2)
    print("Saved votes_agg_printer_complete_domain_latex_bigletter.png")
if len(expanded_printers_no_chunk) > 0:
    agg_printer_no_chunk_heatmap_complete.save("votes_agg_printer_nochunk_complete_domain_latex_bigletter.png", scale_factor=2)
    print("Saved votes_agg_printer_nochunk_complete_domain_latex_bigletter.png")
if cluster_background is not None and len(expanded_printers_no_chunk) > 0:
    agg_printer_no_chunk_heatmap_complete_shaded.save("votes_agg_printer_nochunk_complete_domain_shaded_latex_bigletter.png", scale_factor=2)
    print("Saved votes_agg_printer_nochunk_complete_domain_shaded_latex_bigletter.png")

# Save complete domain detailed printer versions (anon with book names)
if len(expanded_printers_detailed_all) > 0:
    agg_printer_detailed_heatmap_complete.save("votes_agg_printer_detailed_complete_domain_latex_bigletter.png", scale_factor=2)
    print("Saved votes_agg_printer_detailed_complete_domain_latex_bigletter.png")
if len(expanded_printers_detailed_no_chunk) > 0:
    agg_printer_detailed_no_chunk_heatmap_complete.save("votes_agg_printer_detailed_nochunk_complete_domain_latex_bigletter.png", scale_factor=2)
    print("Saved votes_agg_printer_detailed_nochunk_complete_domain_latex_bigletter.png")
if cluster_background is not None and len(expanded_printers_detailed_no_chunk) > 0:
    agg_printer_detailed_no_chunk_heatmap_complete_shaded.save("votes_agg_printer_detailed_nochunk_complete_domain_shaded_latex_bigletter.png", scale_factor=2)
    print("Saved votes_agg_printer_detailed_nochunk_complete_domain_shaded_latex_bigletter.png")

print("Saved plots including cross-book letter highlighting")
print("Saved complete domain versions showing data gaps across all issues")
print("Saved printer-based versions showing which printers printed which issues")
print("Saved detailed printer versions (anon with book names) for more granular analysis")
if cluster_background is not None:
    print("Saved shaded versions showing issue cluster groupings")
    if all_issues_cluster_plot is not None:
        print("Saved cluster overview plot showing all issues (including those without characters)")