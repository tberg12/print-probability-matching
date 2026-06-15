
"""
interactive_annotation.py
=========================
Utility for Jupyter / Colab

• renders one thumbnail‑rich table **per row of data**  
• lets the user pick **one column per table** (yellow highlight)  
• provides **one global “💾 Export” button** that dumps the full
  `{row → col}` mapping to timestamped JSON + CSV.

Usage
-----

```python
from interactive_annotation import build_annotation_ui

rows = [
    paths_for_row0,   # → table 0
    paths_for_row1,   # → table 1
    ...
]

ui = build_annotation_ui(rows)
display(ui)
```
"""

from __future__ import annotations

import json, csv, datetime
from pathlib import Path
from typing import Sequence

from IPython.display import HTML, display
import ipywidgets as W

# ──────────────────────────────────────────────────────────────
# 0  YOUR existing helpers
#     Paste or import `_img_tag` + `paths_to_html_table_with_details`
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
# 1  N tables  +  one global Export button
# ──────────────────────────────────────────────────────────────
def build_annotation_ui(
    rows: Sequence[Sequence[str]],
    *,
    cell_width: str = "15ch",
    table_attrs: str = 'border="1" cellpadding="4"',
):
    """Return a `widgets.Output` ready for `display()`."""

    wrappers = []
    tables_html = []

    for idx, paths in enumerate(rows):
        raw = paths_to_html_table_with_details(
            paths,
            cell_width=cell_width,
            table_attrs=table_attrs,
        )
        wid = f"annottbl-{idx}"
        wrappers.append(wid)
        tables_html.append(
            f'<div id="{wid}" style="margin-bottom:1rem;">{raw}</div>'
        )

    export_btn_id = "globalExportBtn"
    button_html   = f'<button id="{export_btn_id}">💾 Export</button>'

    # JavaScript (plain; no framework)
    import json as _json
    js = f"""
<script>
(function() {{
    const selections = {{}};   // {{row: col}}

    function setSelection(tblId, td) {{
        document.querySelectorAll(`#${{tblId}} td.annot-selected`)
                .forEach(el => el.classList.remove('annot-selected'));
        if (td) td.classList.add('annot-selected');
    }}

    const wrapperIds = {_json.dumps(wrappers)};
    wrapperIds.forEach((wid, rowIdx) => {{
        const tbl = document.querySelector(`#${{wid}} table`);
        tbl.addEventListener('click', ev => {{
            const td = ev.target.closest('td');
            if (!td) return;

            const colIdx = [...td.parentNode.children].indexOf(td) - 1; // ‑1: skip row‑label
            setSelection(wid, td);
            selections[rowIdx] = colIdx;
        }});
    }});

    document.getElementById('{export_btn_id}').addEventListener('click', () => {{
        const payload = JSON.stringify(selections)
                            .replace(/\\/g, '\\\\')      // escape for Python
                            .replace(/`/g, '\\`');
        const code = `
from __future__ import annotations
_save_annotations_json = r'''${{payload}}'''
_save_annotations(_save_annotations_json)
`;
        if (window.google && google.colab) {{
            google.colab.kernel.invokeFunction('notebook.run', [code], {{}});
        }} else {{
            Jupyter.notebook.kernel.execute(code);
        }}
    }});
}})();
</script>

<style>
td {{ cursor: pointer; }}
.annot-selected {{ background: #fffa9e; }}
</style>
"""

    out = W.Output()
    with out:
        display(HTML("".join(tables_html) + button_html + js))
    return out


# ──────────────────────────────────────────────────────────────
# 2  Python → disk
# ──────────────────────────────────────────────────────────────
def _save_annotations(cells_json: str | dict):
    """Write the mapping ``{row: col}`` to timestamped JSON and CSV."""
    mapping = json.loads(cells_json) if isinstance(cells_json, str) else cells_json

    ts   = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    stem = f'annot-{ts}'

    Path(f"{stem}.json").write_text(json.dumps(mapping, indent=2))

    with open(f"{stem}.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["row", "col"])
        for row, col in sorted(mapping.items(), key=lambda p: int(p[0])):
            wr.writerow([row, col])

    print(f"✔ saved {stem}.csv  and  {stem}.json")


# ──────────────────────────────────────────────────────────────
# 3  CLI hint
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(
        'Import this in a notebook:\n'
        '    from interactive_annotation import build_annotation_ui\n'
        '    ui = build_annotation_ui([...])\n'
        '    display(ui)'
    )
