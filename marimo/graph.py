import polars as pl
import networkx as nx
import matplotlib.pyplot as plt
from utils import load_and_prepare_sample

# Load data similarly to viz.py
all_sample = load_and_prepare_sample('LockeSpinozaMatches_processed.csv')

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

print(f"Total records loaded: {len(all_sample)}")
print(f"Columns: {all_sample.columns}")
print(f"Unique books: {sorted(all_sample['book_name'].unique().to_list())}")

# Filter out "no" confidence matches
print(f"\nFiltering out 'no' confidence matches...")
original_count = len(all_sample)
all_sample = all_sample.filter(pl.col("confidence_level") != "no")
filtered_count = len(all_sample)
print(f"Removed {original_count - filtered_count} 'no' confidence matches")
print(f"Remaining records: {filtered_count}")

# Check confidence levels
if "confidence_level" in all_sample.columns:
    confidence_counts = all_sample.group_by("confidence_level").agg(pl.count()).sort("confidence_level")
    print(f"\nConfidence level distribution:")
    for row in confidence_counts.iter_rows(named=True):
        print(f"  {row['confidence_level']}: {row['count']}")

# Determine if a match is a "chunk" or "non-chunk" based on filename
all_sample = all_sample.with_columns(
    is_chunk=pl.col("filename").str.contains("chunk")
)

# Get unique letter types and books
unique_letters = all_sample["letter_with_class"].unique().to_list()
unique_books = all_sample["book_name"].unique().to_list()

print(f"\nUnique letter types: {len(unique_letters)}")
print(f"Unique books: {len(unique_books)}")

# Create a graph showing which books share which letters
# Nodes: books
# Edges: exist when books share at least one letter
# Edge attributes:
#   - solid line if they share non-chunk letters
#   - dashed line if they only share chunk letters
#   - color based on confidence level

G = nx.Graph()

# Add all books as nodes
for book in unique_books:
    G.add_node(book)

# For each pair of books, determine if they share letters
edge_data = []

for i, book1 in enumerate(unique_books):
    for j, book2 in enumerate(unique_books):
        if i < j:  # Only process each pair once
            # Get letters in each book
            letters1 = set(
                all_sample
                .filter(pl.col("book_name") == book1)
                ["letter_with_class"]
                .unique()
                .to_list()
            )

            letters2 = set(
                all_sample
                .filter(pl.col("book_name") == book2)
                ["letter_with_class"]
                .unique()
                .to_list()
            )

            # Find shared letters
            shared_letters = letters1.intersection(letters2)

            if shared_letters:
                # Check confidence levels for shared letters
                # Get all matches between these two books
                shared_matches = all_sample.filter(
                    (pl.col("book_name") == book1) | (pl.col("book_name") == book2)
                ).filter(
                    pl.col("letter_with_class").is_in(list(shared_letters))
                )

                # Check if any shared letters have "hi" confidence
                has_hi_confidence = "hi" in shared_matches["confidence_level"].unique().to_list()

                # Get confidence distribution for this edge
                confidence_counts = shared_matches.group_by("confidence_level").agg(pl.count())
                confidence_dist = {row['confidence_level']: row['count'] for row in confidence_counts.iter_rows(named=True)}

                # Get unique letters by confidence level for accurate counting
                hi_conf_letters = set(
                    shared_matches
                    .filter(pl.col("confidence_level") == "hi")
                    ["letter_with_class"]
                    .unique()
                    .to_list()
                )
                lo_conf_letters = set(
                    shared_matches
                    .filter(pl.col("confidence_level") == "lo")
                    ["letter_with_class"]
                    .unique()
                    .to_list()
                )

                # Check if shared letters come from non-chunk files
                # Get letters from non-chunk files for both books
                nonchunk_letters1 = set(
                    all_sample
                    .filter(
                        (pl.col("book_name") == book1) &
                        (~pl.col("filename").str.contains("chunk"))
                    )
                    ["letter_with_class"]
                    .unique()
                    .to_list()
                )

                nonchunk_letters2 = set(
                    all_sample
                    .filter(
                        (pl.col("book_name") == book2) &
                        (~pl.col("filename").str.contains("chunk"))
                    )
                    ["letter_with_class"]
                    .unique()
                    .to_list()
                )

                # Check for shared non-chunk letters
                shared_nonchunk = nonchunk_letters1.intersection(nonchunk_letters2)

                # Get chunk-only letters (letters that appear only in chunk files)
                chunk_letters1 = set(
                    all_sample
                    .filter(
                        (pl.col("book_name") == book1) &
                        (pl.col("filename").str.contains("chunk"))
                    )
                    ["letter_with_class"]
                    .unique()
                    .to_list()
                )

                chunk_letters2 = set(
                    all_sample
                    .filter(
                        (pl.col("book_name") == book2) &
                        (pl.col("filename").str.contains("chunk"))
                    )
                    ["letter_with_class"]
                    .unique()
                    .to_list()
                )

                # Find letters that are ONLY in chunks (not in non-chunk files)
                chunk_only_letters1 = chunk_letters1 - nonchunk_letters1
                chunk_only_letters2 = chunk_letters2 - nonchunk_letters2
                shared_chunk_only = chunk_only_letters1.intersection(chunk_only_letters2)

                # Create edges based on what types of connections exist
                # We can have both a solid edge (non-chunk) and dashed edge (chunk-only)

                if shared_nonchunk:
                    # Calculate counts for non-chunk letters by confidence
                    nonchunk_hi = shared_nonchunk.intersection(hi_conf_letters)
                    nonchunk_not_lo = shared_nonchunk - lo_conf_letters

                    # Add solid edge for non-chunk matches
                    G.add_edge(book1, book2,
                              style='solid',
                              type='non-chunk',
                              count=len(shared_nonchunk),
                              letters=shared_nonchunk,
                              has_hi_confidence=has_hi_confidence,
                              confidence_dist=confidence_dist,
                              key='solid')  # Add key to allow multiple edges

                    edge_data.append({
                        'book1': book1,
                        'book2': book2,
                        'type': 'non-chunk',
                        'count': len(shared_nonchunk),
                        'count_hi_only': len(nonchunk_hi),
                        'count_not_lo': len(nonchunk_not_lo),
                        'has_hi_confidence': has_hi_confidence,
                        'confidence_dist': confidence_dist,
                        'sample_letters': ', '.join(sorted(list(shared_nonchunk))[:5])
                    })

                if shared_chunk_only:
                    # Calculate counts for chunk-only letters by confidence
                    chunk_hi = shared_chunk_only.intersection(hi_conf_letters)
                    chunk_not_lo = shared_chunk_only - lo_conf_letters

                    # Add dashed edge for chunk-only matches
                    # Note: Using MultiGraph would be cleaner, but we'll handle this in visualization
                    edge_data.append({
                        'book1': book1,
                        'book2': book2,
                        'type': 'chunk-only',
                        'count': len(shared_chunk_only),
                        'count_hi_only': len(chunk_hi),
                        'count_not_lo': len(chunk_not_lo),
                        'has_hi_confidence': has_hi_confidence,
                        'confidence_dist': confidence_dist,
                        'sample_letters': ', '.join(sorted(list(shared_chunk_only))[:5]),
                        'style': 'dashed'
                    })

# Print edge statistics
print(f"\n{'='*60}")
print(f"GRAPH STATISTICS")
print(f"{'='*60}")
print(f"Number of nodes (books): {G.number_of_nodes()}")
print(f"Total edge connections: {len(edge_data)}")

solid_edges = [e for e in edge_data if e['type'] == 'non-chunk']
dashed_edges = [e for e in edge_data if e['type'] == 'chunk-only']
hi_confidence_edges = [e for e in edge_data if e.get('has_hi_confidence', False)]
lo_confidence_edges = [e for e in edge_data if 'lo' in e.get('confidence_dist', {})]

print(f"Non-chunk connections (solid lines): {len(solid_edges)}")
print(f"Chunk-only connections (dashed lines): {len(dashed_edges)}")
print(f"High confidence connections: {len(hi_confidence_edges)}")
print(f"Low confidence connections: {len(lo_confidence_edges)}")

# Count unique book pairs
unique_pairs = set()
for e in edge_data:
    pair = tuple(sorted([e['book1'], e['book2']]))
    unique_pairs.add(pair)
print(f"Unique book pairs with connections: {len(unique_pairs)}")

# Print edge details
print(f"\n{'='*60}")
print(f"EDGE DETAILS")
print(f"{'='*60}")
for edge_info in sorted(edge_data, key=lambda x: x['count'], reverse=True):
    print(f"{edge_info['book1']} <-> {edge_info['book2']}")
    print(f"  Type: {edge_info['type']}, Count: {edge_info['count']}")
    conf_str = ', '.join([f"{k}: {v}" for k, v in edge_info['confidence_dist'].items()])
    hi_marker = " [HIGH CONFIDENCE]" if edge_info['has_hi_confidence'] else ""
    print(f"  Confidence: {conf_str}{hi_marker}")
    print(f"  Sample letters: {edge_info['sample_letters']}")
    print()

# ========== INTERACTIVE BROWSER-BASED VISUALIZATION ==========
from pyvis.network import Network

# Create pyvis network
net = Network(height='900px', width='100%', bgcolor='#ffffff', font_color='black')

# Configure physics for better interactivity
# Using force-atlas based layout for better node distribution
net.set_options("""
{
  "physics": {
    "enabled": true,
    "stabilization": {
      "enabled": true,
      "iterations": 500,
      "updateInterval": 25
    },
    "forceAtlas2Based": {
      "gravitationalConstant": -200,
      "centralGravity": 0.015,
      "springLength": 300,
      "springConstant": 0.08,
      "damping": 0.4,
      "avoidOverlap": 1
    },
    "solver": "forceAtlas2Based"
  },
  "interaction": {
    "dragNodes": true,
    "dragView": true,
    "zoomView": true,
    "hover": true,
    "tooltipDelay": 100
  },
  "nodes": {
    "shape": "box",
    "margin": 10,
    "widthConstraint": {
      "maximum": 200
    },
    "fixed": {
      "x": false,
      "y": false
    }
  },
  "manipulation": {
    "enabled": false
  },
  "edges": {
    "smooth": {
      "enabled": true,
      "type": "continuous"
    }
  }
}
""")

# Get printer names for each book
book_to_printers = {}
for book in unique_books:
    # Check if book name contains specific patterns and override
    if "fortysermons1685" in book:
        book_to_printers[book] = "everingham"
    elif "twotreatises" in book.lower():
        book_to_printers[book] = "anon"
    elif "religionandreason1688" in book:
        book_to_printers[book] = "tbraddyll"
    else:
        printers = all_sample.filter(pl.col("book_name") == book)["printer_name"].unique().to_list()
        # Use the first/most common printer for the book's label
        if printers:
            book_to_printers[book] = printers[0]

unique_printers = sorted(set(book_to_printers.values()))

# Create color mapping for printers
# Color palette - using distinct colors
color_palette = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
    '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#90EE90',
    '#DDA0DD', '#F0E68C', '#87CEEB', '#FFB6C1', '#98FB98',
    '#DEB887', '#5F9EA0', '#FF69B4', '#BA55D3', '#20B2AA',
    '#FFD700', '#FF7F50', '#6495ED', '#DC143C', '#00CED1',
    '#FF1493', '#00FA9A', '#FFE4B5', '#ADFF2F', '#FF6347'
]

printer_colors = {}
for i, printer in enumerate(unique_printers):
    printer_colors[printer] = color_palette[i % len(color_palette)]

print(f"\n{'='*60}")
print("PRINTER COLOR CODING")
print(f"{'='*60}")
for printer, color in sorted(printer_colors.items()):
    books_with_printer = [b for b, p in book_to_printers.items() if p == printer]
    print(f"{printer}: {color} ({len(books_with_printer)} books)")

# Add nodes with styling
for node in G.nodes():
    # Create abbreviated label for display
    label = node
    if len(label) > 40:
        if '_part' in label:
            base = label.split('_part')[0]
            part = label.split('_part')[1]
            if len(base) > 25:
                label = f"{base[:22]}..._part{part}"
        elif len(label) > 40:
            label = f"{label[:37]}..."

    # Get printer name for this node
    printer = book_to_printers.get(node, 'unknown')

    # Get color based on printer name
    node_color = printer_colors.get(printer, '#CCCCCC')

    # Add printer name to the label
    label_with_printer = f"{label}\n[{printer}]"

    # Create detailed title (tooltip) with full name
    title = f"<b>{node}</b><br>Printer: {printer}"

    net.add_node(node,
                label=label_with_printer,
                title=title,
                color={'background': node_color, 'border': node_color},
                borderWidth=1,
                size=25,
                font={'size': 12, 'face': 'arial', 'color': '#000000', 'multi': 'html'})

# Add edges with styling - iterate through edge_data to support multiple edges between nodes
for edge_info in edge_data:
    u = edge_info['book1']
    v = edge_info['book2']
    edge_type = edge_info['type']
    edge_count = edge_info['count']
    has_hi_confidence = edge_info.get('has_hi_confidence', False)
    confidence_dist = edge_info.get('confidence_dist', {})
    sample_letters_str = edge_info.get('sample_letters', '')
    edge_style = edge_info.get('style', 'solid' if edge_type == 'non-chunk' else 'dashed')

    # Create tooltip showing edge details
    conf_str = ', '.join([f"{k}: {v}" for k, v in confidence_dist.items()])
    hi_marker = " ⭐ HIGH CONFIDENCE" if has_hi_confidence else ""

    title = f"""<b>{u}</b> ↔ <b>{v}</b><br>
Type: {edge_type}<br>
Shared letters: {edge_count}<br>
Confidence: {conf_str}{hi_marker}<br>
Examples: {sample_letters_str}"""

    # Determine edge color based on confidence
    if has_hi_confidence:
        # High confidence - green
        edge_color = '#00AA00'
    else:
        # Regular confidence - black
        edge_color = '#000000'

    if edge_style == 'solid':
        # Solid line for non-chunk matches
        net.add_edge(u, v,
                    title=title,
                    label=str(edge_count),
                    color=edge_color,
                    width=4 if has_hi_confidence else 3,
                    dashes=False,
                    font={'size': 24, 'color': '#000000', 'face': 'arial', 'strokeWidth': 4, 'strokeColor': '#ffffff', 'bold': True})
    else:
        # Dashed line for chunk-only matches
        net.add_edge(u, v,
                    title=title,
                    label=str(edge_count),
                    color=edge_color,
                    width=4 if has_hi_confidence else 3,
                    dashes=[5, 5],
                    font={'size': 24, 'color': '#000000', 'face': 'arial', 'strokeWidth': 4, 'strokeColor': '#ffffff', 'bold': True})

# Save and display
output_file = 'book_matching_network.html'
layout_file = 'book_matching_network_layout.json'
net.save_graph(output_file)

# Add custom controls to toggle physics on/off and save layout
import re
import json

# Check if saved layout exists
saved_layout = None
import os
if os.path.exists(layout_file):
    with open(layout_file, 'r') as f:
        saved_layout = json.load(f)
    print(f"\n✓ Found saved layout: {layout_file}")
    print(f"  Loaded positions for {len(saved_layout)} nodes")
else:
    print(f"\n✗ No saved layout found")

with open(output_file, 'r') as f:
    html_content = f.read()

# Add custom button and script to toggle physics and save layout
saved_layout_json = json.dumps(saved_layout) if saved_layout else "null"

# Store edge data with confidence info for browser-side filtering
edge_data_for_js = []
for edge_info in edge_data:
    edge_data_for_js.append({
        'from': edge_info['book1'],
        'to': edge_info['book2'],
        'type': edge_info['type'],
        'count': edge_info['count'],
        'count_not_lo': edge_info.get('count_not_lo', edge_info['count']),
        'has_hi_confidence': edge_info.get('has_hi_confidence', False),
        'confidence_dist': edge_info.get('confidence_dist', {}),
        'has_lo_confidence': 'lo' in edge_info.get('confidence_dist', {})
    })
edge_data_json = json.dumps(edge_data_for_js)

custom_controls = f"""
<style>
  .control-button {{
    position: fixed;
    right: 20px;
    z-index: 1000;
    padding: 12px 24px;
    font-size: 16px;
    font-weight: bold;
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    transition: background-color 0.3s;
  }}
  #physics-toggle {{
    top: 20px;
    background-color: #4CAF50;
  }}
  #physics-toggle:hover {{
    background-color: #45a049;
  }}
  #physics-toggle.off {{
    background-color: #f44336;
  }}
  #physics-toggle.off:hover {{
    background-color: #da190b;
  }}
  #save-layout {{
    top: 70px;
    background-color: #2196F3;
  }}
  #save-layout:hover {{
    background-color: #0b7dda;
  }}
  #confidence-toggle {{
    top: 120px;
    background-color: #FF9800;
  }}
  #confidence-toggle:hover {{
    background-color: #F57C00;
  }}
  #confidence-toggle.hiding {{
    background-color: #9C27B0;
  }}
  #confidence-toggle.hiding:hover {{
    background-color: #7B1FA2;
  }}
  #info-box {{
    position: fixed;
    top: 170px;
    right: 20px;
    z-index: 1000;
    padding: 12px;
    background-color: rgba(255, 255, 255, 0.95);
    border: 2px solid #333;
    border-radius: 8px;
    font-size: 14px;
    max-width: 300px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
  }}
</style>
<button id="physics-toggle" class="control-button" onclick="togglePhysics()">Physics: ON (Click to freeze nodes)</button>
<button id="save-layout" class="control-button" onclick="saveLayout()">💾 Save Layout</button>
<button id="confidence-toggle" class="control-button" onclick="toggleLowConfidence()">Show Low Confidence</button>
<div id="info-box">
  <strong>Controls:</strong><br>
  • Drag nodes to reposition<br>
  • Scroll to zoom<br>
  • Drag background to pan<br>
  • Toggle physics OFF to freeze nodes<br>
  • Toggle low confidence matches<br>
  • Save layout to preserve positions<br>
  <br>
  <strong>Legend:</strong><br>
  • Node colors = printer names<br>
  • Node labels show: book name [printer]<br>
  • Edge numbers = shared letter count<br>
  • Green edges = HIGH confidence<br>
  • Black edges = regular confidence<br>
  • Solid = non-chunk matches<br>
  • Dashed = chunk-only matches<br>
</div>
<script type="text/javascript">
  var physicsEnabled = true;
  var savedLayout = {saved_layout_json};
  var showLowConfidence = true;
  var edgeData = {edge_data_json};
  var allEdges = null;

  // Store all edges after network is created
  network.once("stabilizationIterationsDone", function() {{
    // Save all edges for filtering
    allEdges = network.body.data.edges.get();
    console.log("Total edges stored:", allEdges.length);

    // Apply saved layout if available
    if (savedLayout) {{
      console.log("Loading saved layout for", Object.keys(savedLayout).length, "nodes");

      // Disable physics first to prevent nodes from moving
      network.setOptions({{ physics: {{ enabled: false }} }});
      physicsEnabled = false;

      // Update button state
      var button = document.getElementById('physics-toggle');
      button.textContent = 'Physics: OFF (Nodes frozen)';
      button.classList.add('off');

      // Apply saved positions
      Object.keys(savedLayout).forEach(function(nodeId) {{
        var pos = savedLayout[nodeId];
        network.moveNode(nodeId, pos.x, pos.y);
      }});

      console.log("Saved layout applied and physics disabled");
    }}
  }});

  function togglePhysics() {{
    physicsEnabled = !physicsEnabled;
    network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});

    var button = document.getElementById('physics-toggle');
    if (physicsEnabled) {{
      button.textContent = 'Physics: ON (Click to freeze nodes)';
      button.classList.remove('off');
    }} else {{
      button.textContent = 'Physics: OFF (Nodes frozen)';
      button.classList.add('off');
    }}
  }}

  function toggleLowConfidence() {{
    showLowConfidence = !showLowConfidence;
    var button = document.getElementById('confidence-toggle');

    if (!allEdges) {{
      console.warn("Edges not yet loaded, waiting...");
      setTimeout(toggleLowConfidence, 500);
      return;
    }}

    if (showLowConfidence) {{
      // Show all edges with original counts
      button.textContent = 'Show Low Confidence';
      button.classList.remove('hiding');
      network.body.data.edges.update(allEdges);
      console.log("Showing all edges:", allEdges.length);
    }} else {{
      // Hide low confidence edges and adjust counts
      button.textContent = 'Hide Low Confidence';
      button.classList.add('hiding');

      // Filter out edges that are only low confidence
      // Also update labels to show count without low confidence matches
      var filteredEdges = allEdges.filter(function(edge) {{
        // Find corresponding edge in edgeData
        var edgeInfo = edgeData.find(function(e) {{
          return (e.from === edge.from && e.to === edge.to) ||
                 (e.from === edge.to && e.to === edge.from);
        }});

        if (!edgeInfo) return true; // Keep if not found (shouldn't happen)

        // Hide edge if count_not_lo is 0 (meaning all matches are low confidence)
        // Otherwise keep it and update the label
        return edgeInfo.count_not_lo > 0;
      }}).map(function(edge) {{
        // Update label to show count without low confidence matches
        var edgeInfo = edgeData.find(function(e) {{
          return (e.from === edge.from && e.to === edge.to) ||
                 (e.from === edge.to && e.to === edge.from);
        }});

        if (edgeInfo && edgeInfo.count_not_lo < edgeInfo.count) {{
          // Create a copy and update the label to show count without low confidence
          var updatedEdge = Object.assign({{}}, edge);
          updatedEdge.label = String(edgeInfo.count_not_lo);
          return updatedEdge;
        }}
        return edge;
      }});

      // Update edges
      var edgeIds = allEdges.map(e => e.id);
      network.body.data.edges.remove(edgeIds);
      network.body.data.edges.update(filteredEdges);

      console.log("Filtered edges:", filteredEdges.length, "of", allEdges.length);
    }}
  }}

  function saveLayout() {{
    var positions = network.getPositions();
    var layoutData = JSON.stringify(positions, null, 2);

    // Create a download link
    var blob = new Blob([layoutData], {{ type: 'application/json' }});
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'book_matching_network_layout.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    alert('Layout saved! The file will be downloaded.\\n\\nPlace it in the same directory as graph.py to use it next time.');
  }}

  // Auto-disable physics after stabilization
  network.on("stabilizationIterationsDone", function () {{
    console.log("Network stabilized");
  }});

  // Fix node position when dragged
  network.on("dragEnd", function (params) {{
    if (params.nodes.length > 0 && !physicsEnabled) {{
      params.nodes.forEach(function(nodeId) {{
        network.moveNode(nodeId, params.pointer.canvas.x, params.pointer.canvas.y);
      }});
    }}
  }});
</script>
"""

# Insert the custom controls before the closing body tag
html_content = html_content.replace('</body>', custom_controls + '</body>')

with open(output_file, 'w') as f:
    f.write(html_content)

print(f"\n{'='*60}")
print("INTERACTIVE BROWSER VISUALIZATION CREATED!")
print(f"{'='*60}")
print(f"HTML file saved to: {output_file}")
print("\nFeatures:")
print("  • Click 'Physics: ON' button to FREEZE nodes (stops snap-back)")
print("  • Click '💾 Save Layout' to download current node positions")
print("  • Click 'Show Low Confidence' to HIDE low confidence edges")
print("  • Drag nodes to reposition them")
print("  • Scroll to zoom in/out")
print("  • Hover over nodes and edges for details")
print("  • Drag the canvas to pan around")
print("\nNode Display:")
print("  • Node color: printer name")
print("  • Node label: book name [printer name]")
print("\nEdge Colors:")
print("  • Green: HIGH confidence matches")
print("  • Black: regular confidence matches")
print("  • Solid lines: non-chunk matches")
print("  • Dashed lines: chunk-only matches")
print(f"{'='*60}\n")

# Open in browser automatically
import webbrowser
import os
file_path = os.path.abspath(output_file)
webbrowser.open('file://' + file_path)

print("Opening in browser...")
print("\nDone!")
