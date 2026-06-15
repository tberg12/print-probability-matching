import pandas as pd
import re
import os
import base64
import io
from pathlib import Path
from PIL import Image

# --- 1. SET YOUR INPUTS HERE ---

# Paste your full LaTeX table code inside the triple quotes.
# You can include multiple tables one after another.
latex_code = """
\\begin{table}[t]
    \\centering
    \\scalebox{.7}{
    % \\resizebox{\\linewidth}{!}{
    \\begin{tabular}{|c|c|c|c|}
		\\hline
	 \\multicolumn{4}{|c|}{\\LARGE \\shortstack{Robert Everingham and Thomas Braddyll:\\\\Printers of Locke's \\emph{Two Treatises} \\\\ and Spinoza's \\emph{Theological-Political Treatise}}} \\\\ \\hline


\\makecell[t]{\\MEDIUM \\emph{Two Treatises of Government} (1690)} & \\makecell[t]{\\MEDIUM \\emph{Theological-Political Treatise} (1689)} &\\makecell[t]{\\MEDIUM Acknowledged output of\\\\Robert Everingham} & \\makecell[t]{\\MEDIUM Presumptive output of\\\\Thomas Braddyll}\\\\ \\hline

		\\includegraphics[height=2.75cm]{ims/F_uc - (96077) Two treatises of government in... p. 84-s l. 35 c. 2.jpg} & \\includegraphics[height=2.75cm]{ims/F_uc - (63390) A treatise partly theological,... p. 206-s l. 30 c. 8.jpg} &
        \\includegraphics[height=2.75cm]{ims/F_uc - (48072) Forty sermons. preached by the... p. 639-s l. 20 c. 31.jpg} & \\\\\\hline



  
        \\includegraphics[height=2.75cm]{ims/N_uc - (96077) Two treatises of government in... p. 104-s l. 12 c. 12.jpg} & \\includegraphics[height=2.75cm]{ims/N_uc - (63390) A treatise partly theological,... p. 57-s l. 1 c. 28.jpg}
        & \\includegraphics[height=2.75cm]{ims/N_uc - (48072) Forty sermons. preached by the... p. 657-s l. 35 c. 75.jpg} & \\\\\\hline
        
        \\includegraphics[height=2.75cm]{ims/P_uc - (96077) Two treatises of government in... p. 66-s l. 17 c. 8.jpg} & \\includegraphics[height=2.75cm]{ims/P_uc - (63390) A treatise partly theological,... p. 253-s l. 18 c. 37.jpg} & & \\includegraphics[height=2.75cm]{ims/tbraddyll_R23639_mdp_8_religionandreasonREDO1688-0103_page1rline14_char19_P_uc.jpg}\\\\\hline
        
        % \\includegraphics[height=2.75cm]{ims/Q_uc - (48072) Forty sermons. preached by the... p. 606-s l. 40 c. 46.jpg} & \\includegraphics[height=2.75cm]{ims/Q_uc - (96077) Two treatises of government in... p. 32-s l. 17 c. 20.jpg}\\\\\hline
        
        \\includegraphics[height=2.75cm]{ims/O_uc - (96077) Two treatises of government in... p. 175-s l. 28 c. 11.jpg} & \\includegraphics[height=2.75cm]
        {ims/O_uc - (63390) A treatise partly theological,... p. 15-s l. 18 c. 9.jpg} &  & \\\\\\hline

        \\includegraphics[height=2.75cm]{ims/H_uc - (96077) Two treatises of government in... p. 61-s l. 38 c. 18.jpg} &
        \\includegraphics[height=2.75cm]{ims/H_uc - (63390) A treatise partly theological,... p. 181-s l. 3 c. 37.jpg} & 
         \\includegraphics[height=2.75cm]{ims/reveringham_R30863_ctbtcml_2_fortysermons1685-594_page1rline30_char55_H_uc} & \\\\\\hline

        & \\includegraphics[height=2.75cm]{ims/N_uc - (63390) A treatise partly theological,... p. 261-s l. 2 c. 31.jpg} &  \\includegraphics[height=2.75cm]{ims/N_uc - (48072) Forty sermons. preached by the... p. 599-s l. 30 c. 26.jpg} & \\\\\\hline

        \\includegraphics[height=2.75cm]{ims/R_uc - (96077) Two treatises of government in... p. 96-s l. 16 c. 13.jpg} & & \\includegraphics[height=2.75cm]{ims/R_uc - (48072) Forty sermons. preached by the... p. 680-s l. 42 c. 12.jpg} & \\\\\\hline

        \\includegraphics[height=2.75cm]{ims/Q_uc - (96077) Two treatises of government in... p. 32-s l. 17 c. 20.jpg} & & \\includegraphics[height=2.75cm]{ims/Q_uc - (48072) Forty sermons. preached by the... p. 606-s l. 40 c. 46.jpg} & \\\\\\hline
        
        & \\includegraphics[height=2.75cm]{ims/H_uc - (63390) A treatise partly theological,... p. 404-s l. 26 c. 1.jpg}  & \\includegraphics[height=2.75cm]{ims/H_uc - (48072) Forty sermons. preached by the... p. 528-s l. 26 c. 63.jpg} & \\\\\\hline

        \\includegraphics[height=2.75cm]{ims/anon_R2930_iur_8_twotreatisesofgov1690-0244_page1rline23_char14_N_uc.jpg} & \\includegraphics[height=2.75cm]{ims/anon_R21627_gw_8_spinozatheologicalpolitical1689-0448_page1rline33_char18_N_uc.jpg} & & \\includegraphics[height=2.75cm]{ims/tbraddyll_R23639_mdp_8_religionandreasonREDO1688-0288_page1rline29_char2_N_uc.jpg}\\\\\hline
    \\end{tabular}}
\\end{table}

\\begin{table}[t]
    \\centering
    \\scalebox{.7}{
    \\begin{tabular}{|c|c|c|c|}
		\\hline
\\makecell[t]{\\MEDIUM \\emph{Two Treatises of Government} (1690)} & \\makecell[t]{\\MEDIUM \\emph{Theological-Political Treatise} (1689)} &\\makecell[t]{\\MEDIUM Acknowledged output of\\\\Robert Everingham} & \\makecell[t]{\\MEDIUM Presumptive output of\\\\Thomas Braddyll}\\\\ \\hline

		& \\includegraphics[height=2.75cm]{ims/anon_R21627_gw_8_spinozatheologicalpolitical1689-0017_page1rline20_char35_A_uc.jpg} & & \\includegraphics[height=2.75cm]{ims/tbraddyll_R23639_mdp_8_religionandreasonREDO1688-0031_page1rline10_char6_A_uc.jpg} \\\\\\hline

        & \\includegraphics[height=2.75cm]{ims/anon_R21627_gw_8_spinozatheologicalpolitical1689-0233_page1rline25_char12_B_uc.jpg} & & \\includegraphics[height=2.75cm]{ims/tbraddyll_R23639_mdp_8_religionandreasonREDO1688-0429_page1rline13_char16_B_uc.jpg} \\\\\\hline

        & \\includegraphics[height=2.75cm]{ims/anon_R21627_gw_8_spinozatheologicalpolitical1689-0103_page1rline2_char12_T_uc.jpg} & & \\includegraphics[height=2.75cm]{ims/tbraddyll_R23639_mdp_8_religionandreasonREDO1688-0271_page1rline24_char11_T_uc.jpg} \\\\\\hline

        & \\includegraphics[height=2.75cm]{ims/anon_R21627_gw_8_spinozatheologicalpolitical1689-0272_page1rline2_char25_T_uc.jpg} & & \\includegraphics[height=2.75cm]{ims/tbraddyll_R23639_mdp_8_religionandreasonREDO1688-0360_page1rline13_char16_T_uc.jpg} \\\\\\hline

        & \\includegraphics[height=2.75cm]{ims/anon_R21627_gw_8_spinozatheologicalpolitical1689-0166_page1rline10_char9_C_uc.jpg} & & \\includegraphics[height=2.75cm]{ims/tbraddyll_R23639_mdp_8_religionandreasonREDO1688-0084_page1rline24_char31_C_uc.jpg} \\\\\\hline

        \\includegraphics[height=2.75cm]{ims/anon_R2930_iur_8_twotreatisesofgov1690-0142_page1rline7_char30_F_uc.jpg} & & & \\includegraphics[height=2.75cm]{ims/tbraddyll_R23639_mdp_8_religionandreasonREDO1688-0129_page1rline25_char6_F_uc.jpg} \\\\\\hline

        \\includegraphics[height=2.75cm]{ims/anon_R2930_iur_8_twotreatisesofgov1690-0053_page1rline18_char24_G_uc.jpg} & \\includegraphics[height=2.75cm]{ims/anon_R21627_gw_8_spinozatheologicalpoliticalREDO1689-0196_page1rline12_char13_G_uc.jpg} & & \\includegraphics[height=2.75cm]{ims/tbraddyll_R23639_mdp_8_religionandreasonREDO1688-0160_page1rline26_char21_G_uc.jpg} \\\\\\hline

        \\includegraphics[height=2.75cm]{ims/anon_R2930_iur_8_twotreatisesofgov1690-0304_page1rline25_char32_F_uc.jpg} & & & \\includegraphics[height=2.75cm]{ims/tbraddyll_R14212_getty_8_voyageofitalyREDO1685-0034_page1rline13_char2_F_uc.jpg} \\\\\\hline

        \\includegraphics[height=2.75cm]{ims/anon_R2930_iur_8_twotreatisesofgovREDO1690-0025_page1rline1_char6_P_uc.jpg} & & & \\includegraphics[height=2.75cm]{ims/tbraddyll_R42222_uk_8_plutarchmoralsREDO1694-0114_page1rline39_char30_P_uc.jpg} \\\\\\hline

        \\includegraphics[height=2.75cm]{ims/anon_R2930_iur_8_twotreatisesofgovREDO1690-0360_page1rline25_char17_A_uc.jpg} & & & \\includegraphics[height=2.75cm]{ims/tbraddyll_R22245_uk_2_twobooksindefenceREDO1680-0038_page1rline24_char41_A_uc.jpg} \\\\\\hline
    \\end{tabular}}
\\end{table}
"""

# Specify the path to your existing spreadsheet and where to save the new one.
input_spreadsheet = "marimo/sample_review_updated.xlsx"
output_spreadsheet = "marimo/sample_review_updated_latex.xlsx"

# --- DO NOT EDIT BELOW THIS LINE ---

EXPECTED_COLUMNS = [
    'root_image', 'page_number', 'filename', 'image',
    'is_root', 'gathering', 'letter', 'letter_class', 'confidence_level'
]

def encode_image_to_base64(file_path):
    """Opens an image, converts it to PNG in memory, and returns a base64 string."""
    try:
        with Image.open(file_path) as img:
            # Convert to a format that supports transparency (like RGBA)
            img = img.convert("RGBA")
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{img_str}"
    except FileNotFoundError:
        print(f"    - ⚠️  Warning: Image file not found at '{file_path}'. Skipping encoding.")
        return None
    except Exception as e:
        print(f"    - ❌ Error encoding {file_path}: {e}")
        return None


def extract_filenames_from_latex(latex):
    """Parses LaTeX code to extract image paths from \\includegraphics commands, grouped by table row."""
    all_groups = []
    # Regex to find the full path inside \includegraphics{...}
    path_regex = re.compile(r'\\includegraphics.*?\{(.*?)\}')
    # Split content into individual table environments
    tables = latex.split('\\begin{table}')
    for table in tables:
        if not table.strip() or '\\begin{tabular}' not in table:
            continue
        # Split the table's content into rows using the row separator
        rows = table.split('\\\\')
        for row in rows:
            # Find all image paths in the current row
            image_paths = path_regex.findall(row)
            if image_paths:
                all_groups.append(image_paths)
    return all_groups


def main():
    """Main function to run the script."""
    # --- 2. Load Existing Spreadsheet ---
    try:
        # Check for both .xlsx and .csv extensions for flexibility
        if os.path.exists(input_spreadsheet):
             df = pd.read_excel(input_spreadsheet)
        elif os.path.exists(input_spreadsheet.replace('.xlsx', '.csv')):
             df = pd.read_csv(input_spreadsheet.replace('.xlsx', '.csv'))
        else:
            raise FileNotFoundError

        print(f"✅ Successfully loaded '{input_spreadsheet}' with {len(df)} existing entries.")
        if 'letter_class' in df.columns and not df['letter_class'].empty:
            current_class_id = pd.to_numeric(df['letter_class'], errors='coerce').max() + 1
        else:
            current_class_id = 1

    except FileNotFoundError:
        print(f"⚠️ Warning: Could not find '{input_spreadsheet}'. A new spreadsheet will be created.")
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
        current_class_id = 1
    except Exception as e:
        print(f"❌ Error loading spreadsheet: {e}")
        return

    # --- 3. Extract and Process Filenames ---
    groups = extract_filenames_from_latex(latex_code)
    if not groups:
        print("❌ No image file paths found in the provided LaTeX code. Please check your input.")
        return

    print(f"\nProcessing {len(groups)} new groups of images...")
    new_rows = []
    for group_index, group in enumerate(groups):
        print(f"  Processing Group {group_index + 1}/{len(groups)}...")
        root_image_path = group[0]
        root_filename = os.path.basename(root_image_path)

        for i, image_path in enumerate(group):
            filename = os.path.basename(image_path)

            # Encode the image to base64
            actual_image_path = Path('lists')/image_path
            if not actual_image_path.exists():
                # add .jpg if not present and check again
                actual_image_path = actual_image_path.with_suffix('.jpg')
            base64_image = encode_image_to_base64(actual_image_path)

            # Extract letter
            letter_match = re.search(r'_([A-Z])_uc', filename)
            if not letter_match:
                letter_match = re.search(r'^([A-Z])_uc', filename)
            letter = letter_match.group(1) if letter_match else None

            # Extract page number
            page_match = re.search(r'-(\d+)', filename)
            if not page_match:
                page_match = re.search(r'p\. (\d+)', filename)
            page_number = page_match.group(1) if page_match else None

            new_rows.append({
                'filename': filename,
                'image': base64_image,
                'letter': letter,
                'letter_class': current_class_id,
                'root_image': root_filename,
                'page_number': page_number,
                'is_root': i == 0,
                'gathering': None,
            })
        current_class_id += 1

    if not new_rows:
        print("No new files were processed to add.")
        return

    # --- 4. Combine and Save ---
    new_df = pd.DataFrame(new_rows)
    combined_df = pd.concat([df, new_df], ignore_index=True)

    # Ensure columns are in the correct order
    final_df = combined_df.reindex(columns=EXPECTED_COLUMNS)

    try:
        final_df.to_excel(output_spreadsheet, index=False)
        print(f"\n🎉 Success! Processed {len(groups)} new groups with a total of {len(new_rows)} files.")
        print(f"Updated spreadsheet saved to '{output_spreadsheet}'")
    except Exception as e:
        print(f"❌ Error saving file: {e}")


if __name__ == "__main__":
    main()

