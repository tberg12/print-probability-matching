import pandas as pd
import re
import os
import base64
import io
from PIL import Image

# --- 1. SET YOUR INPUTS HERE ---

# Paste the full list of new file paths inside the triple quotes.
# new_files_text = """
# /graft2/code/nvog/git/matching/char_images3/char_A_uc/reveringham_R30863_ctbtcml_2_fortysermons1685-675_page1rline23_char7_A_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_A_uc/reveringham_R30863_ctbtcml_2_fortysermonsREDO1685-0621_page1rline22_char52_A_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_A_uc/reveringham_R30863_ctbtcml_2_fortysermonsREDO1685-0604_page1rline27_char4_A_uc.tif

# /graft2/code/nvog/git/matching/char_images3/char_A_uc/reveringham_R30863_ctbtcml_2_fortysermons1685-681_page1rline52_char39_A_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_A_uc/reveringham_R30863_ctbtcml_2_fortysermonsREDO1685-0613_page1rline34_char47_A_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_A_uc/reveringham_R30863_ctbtcml_2_fortysermonsREDO1685-0599_page1rline21_char73_A_uc.tif

# /graft2/code/nvog/git/matching/char_images3/char_A_uc/reveringham_R30863_ctbtcml_2_fortysermons1685-591_page1rline10_char21_A_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_A_uc/reveringham_R30863_ctbtcml_2_fortysermonsREDO1685-0635_page1rline35_char13_A_uc.tif

# /graft2/code/nvog/git/matching/char_images3/char_C_uc/reveringham_R30863_ctbtcml_2_fortysermons1685-474_page1rline10_char67_C_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_C_uc/reveringham_R30863_ctbtcml_2_fortysermons1685-494_page1rline35_char45_C_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_C_uc/reveringham_R30863_ctbtcml_2_fortysermons1685-557_page1rline51_char78_C_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_C_uc/everingham_R3105_de12_4_ogygiaREDO1685-0075_page1rline13_char23_C_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_C_uc/everingham_R3105_de12_4_ogygiaREDO1685-0064_page1rline25_char34_C_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_C_uc/everingham_R3105_de12_4_ogygiaREDO1685-0168_page1rline35_char24_C_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_C_uc/everingham_R3105_de12_4_ogygiaREDO1685-0087_page1rline6_char24_C_uc.tif

# /graft2/code/nvog/git/matching/char_images3/char_C_uc/reveringham_R30863_ctbtcml_2_fortysermons1685-705_page1rline51_char47_C_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_C_uc/everingham_R3105_de12_4_ogygiaREDO1685-0521_page1rline4_char20_C_uc.tif

# /graft2/code/nvog/git/matching/char_images3/char_C_uc/reveringham_R30863_ctbtcml_2_fortysermons1685-722_page1rline10_char35_C_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_C_uc/everingham_R3105_de12_4_ogygiaREDO1685-0396_page1rline41_char52_C_uc.tif

# /graft2/code/nvog/git/matching/char_images3/char_C_uc/reveringham_R30863_ctbtcml_2_fortysermons1685-585_page1rline25_char11_C_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_C_uc/reveringham_R30863_ctbtcml_2_fortysermonsREDO1685-0678_page1rline50_char19_C_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_C_uc/reveringham_R30863_ctbtcml_2_fortysermonsREDO1685-0585_page1rline15_char13_C_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_C_uc/reveringham_R30863_ctbtcml_2_fortysermonsREDO1685-0694_page1rline14_char44_C_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_C_uc/reveringham_R30863_ctbtcml_2_fortysermonsREDO1685-0704_page1rline27_char2_C_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_C_uc/reveringham_R30863_ctbtcml_2_fortysermonsREDO1685-0608_page1rline38_char45_C_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_C_uc/reveringham_R30863_ctbtcml_2_fortysermonsREDO1685-0661_page1rline33_char24_C_uc.tif

# /graft2/code/nvog/git/matching/char_images3/char_H_uc/reveringham_R30863_ctbtcml_2_fortysermons1685-486_page1rline35_char14_H_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_H_uc/everingham_R3105_de12_4_ogygiaREDO1685-0070_page1rline34_char39_H_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_H_uc/everingham_R2059_uk_8_anekdotaheterouriakREDO1686-0526_page1rline17_char11_H_uc.tif

# /graft2/code/nvog/git/matching/char_images3/char_H_uc/reveringham_R30863_ctbtcml_2_fortysermons1685-490_page1rline4_char7_H_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_H_uc/everingham_R3105_de12_4_ogygiaREDO1685-0160_page1rline45_char39_H_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_H_uc/everingham_R3105_de12_4_ogygiaREDO1685-0142_page1rline14_char2_H_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_H_uc/everingham_R3105_de12_4_ogygiaREDO1685-0250_page1rline27_char34_H_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_H_uc/everingham_R3105_de12_4_ogygiaREDO1685-0424_page1rline21_char37_H_uc.tif

# /graft2/code/nvog/git/matching/char_images3/char_H_uc/reveringham_R30863_ctbtcml_2_fortysermons1685-578_page1rline31_char18_H_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_H_uc/everingham_R3105_de12_4_ogygiaREDO1685-0308_page1rline33_char1_H_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_H_uc/everingham_R3105_de12_4_ogygiaREDO1685-0219_page1rline40_char14_H_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_H_uc/everingham_R3105_de12_4_ogygiaREDO1685-0109_page1rline12_char38_H_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_H_uc/everingham_R3105_de12_4_ogygiaREDO1685-0107_page1rline17_char37_H_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_H_uc/everingham_R3105_de12_4_ogygiaREDO1685-0238_page1rline23_char27_H_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_H_uc/everingham_R3105_de12_4_ogygiaREDO1685-0554_page1rline32_char6_H_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_H_uc/everingham_R3105_de12_4_ogygiaREDO1685-0426_page1rline16_char26_H_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_H_uc/everingham_R3105_de12_4_ogygiaREDO1685-0472_page1rline11_char6_H_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_H_uc/everingham_R3105_de12_4_ogygiaREDO1685-0400_page1rline3_char25_H_uc.tif

# /graft2/code/nvog/git/matching/char_images3/char_M_uc/anon_R21627_gw_8_spinozatheologicalpoliticalREDO1689-0322_page1rline12_char24_M_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_M_uc/reveringham_R30863_ctbtcml_2_fortysermonsREDO1685-0480_page1rline31_char60_M_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_M_uc/everingham_R2059_uk_8_anekdotaheterouriakREDO1686-0174_page1rline2_char16_M_uc.tif

# /graft2/code/nvog/git/matching/char_images3/char_N_uc/anon_R21627_gw_8_spinozatheologicalpolitical1689-0365_page1rline36_char4_N_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_N_uc/everingham_R2059_uk_8_anekdotaheterouriakREDO1686-0405_page1rline19_char29_N_uc.tif

# /graft2/code/nvog/git/matching/char_images3/char_N_uc/reveringham_R30863_ctbtcml_2_fortysermons1685-083_page1rline5_char43_N_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_N_uc/everingham_R122_uklw_8_examenpoeticumREDO1693-0569_page1rline7_char5_N_uc.tif

# /graft2/code/nvog/git/matching/char_images3/char_N_uc/reveringham_R30863_ctbtcml_2_fortysermons1685-441_page1rline10_char32_N_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_N_uc/everingham_R122_uklw_8_examenpoeticumREDO1693-0559_page1rline5_char50_N_uc.tif

# /graft2/code/nvog/git/matching/char_images3/char_T_uc/reveringham_R30863_ctbtcml_2_fortysermons1685-426_page1rline42_char5_T_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_T_uc/reveringham_R30863_ctbtcml_2_fortysermons1685-487_page1rline45_char38_T_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_T_uc/everingham_R3105_de12_4_ogygiaREDO1685-0090_page1rline7_char1_T_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_T_uc/everingham_R3105_de12_4_ogygiaREDO1685-0076_page1rline12_char30_T_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_T_uc/everingham_R3105_de12_4_ogygiaREDO1685-0065_page1rline23_char40_T_uc.tif

# /graft2/code/nvog/git/matching/char_images3/char_T_uc/anon_R21627_gw_8_spinozatheologicalpoliticalREDO1689-0272_page1rline3_char27_T_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_T_uc/anon_R21627_gw_8_spinozatheologicalpoliticalREDO1689-0252_page1rline11_char13_T_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_T_uc/reveringham_R30863_ctbtcml_2_fortysermonsREDO1685-0614_page1rline37_char2_T_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_T_uc/reveringham_R30863_ctbtcml_2_fortysermonsREDO1685-0659_page1rline24_char58_T_uc.tif
# /graft2/code/nvog/git/matching/char_images3/char_T_uc/tbraddyll_R23639_mdp_8_religionandreasonREDO1688-0360_page1rline13_char16_T_uc.tif
# """

new_files_text = """
big_letters/t/anon_R21627_gw_8_spinozatheologicalpoliticalREDO1689-0028_chunk_001.jpg
big_letters/t/tbraddyll_R42222_uk_8_plutarchmorals1694-0010_chunk_000.jpg

big_letters/e/anon_R21627_gw_8_spinozatheologicalpolitical1689-0028_chunk_009.jpg
big_letters/e/anon_R17109_uk_2_ThePlotrevivdorAme1680-0000_chunk_014.jpg
big_letters/e/tbraddyll_R4480_munich_2_annalsofkingjamescharles1681-0012_chunk_001.jpg 
big_letters/e/tbraddyll_R4480_munich_2_annalsofkingjamescharles1681-0020_chunk_001.jpg

big_letters/e/tbraddyll_R12782_usnjpt_4_criticalenquiriesinto1684-5_chunk_023.jpg
big_letters/e/reveringham_R30863_ctbtcml_2_fortysermons1685-613_chunk_032.jpg
big_letters/e/reveringham_R30863_ctbtcml_2_fortysermons1685-604_chunk_001.jpg
big_letters/e/reveringham_R30863_ctbtcml_2_fortysermons1685-677_chunk_008.jpg
big_letters/e/reveringham_R30863_ctbtcml_2_fortysermons1685-688_chunk_004.jpg
big_letters/e/reveringham_R30863_ctbtcml_2_fortysermons1685-637_chunk_026.jpg
big_letters/e/reveringham_R30863_ctbtcml_2_fortysermons1685-668_chunk_005.jpg
big_letters/e/reveringham_R30863_ctbtcml_2_fortysermons1685-657_chunk_009.jpg
big_letters/e/reveringham_R30863_ctbtcml_2_fortysermons1685-592_chunk_005.jpg

big_letters/b/reveringham_R30863_ctbtcml_2_fortysermons1685-007_chunk_028.jpg
big_letters/b/reveringham_R30863_ctbtcml_2_fortysermons1685-457_chunk_027.jpg

big_letters/e/reveringham_R30863_ctbtcml_2_fortysermons1685-708_chunk_008.jpg
big_letters/e/reveringham_R30863_ctbtcml_2_fortysermons1685-314_chunk_004.jpg
big_letters/e/reveringham_R30863_ctbtcml_2_fortysermons1685-566_chunk_008.jpg

big_letters/e/reveringham_R30863_ctbtcml_2_fortysermons1685-162_chunk_019.jpg
big_letters/e/reveringham_R30863_ctbtcml_2_fortysermons1685-173_chunk_009.jpg

big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-465_chunk_026.jpg
big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-089_chunk_016.jpg
big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-052_chunk_027.jpg
big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-285_chunk_011.jpg
big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-213_chunk_025.jpg
big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-334_chunk_009.jpg
big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-486_chunk_019.jpg
big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-149_chunk_013.jpg
big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-021_chunk_059.jpg
big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-162_chunk_027.jpg
big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-260_chunk_032.jpg
big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-420_chunk_012.jpg

big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-457_chunk_049.jpg
big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-007_chunk_050.jpg

big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-604_chunk_005.jpg
big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-592_chunk_006.jpg

big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-637_chunk_029.jpg
big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-677_chunk_007.jpg
big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-668_chunk_008.jpg
big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-646_chunk_007.jpg
big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-688_chunk_007.jpg

big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-296_chunk_017.jpg
big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-708_chunk_006.jpg

big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-314_chunk_008.jpg
big_letters/m/reveringham_R30863_ctbtcml_2_fortysermons1685-566_chunk_005.jpg

big_letters/n/reveringham_R30863_ctbtcml_2_fortysermons1685-688_chunk_005.jpg
big_letters/n/reveringham_R30863_ctbtcml_2_fortysermons1685-668_chunk_009.jpg
big_letters/n/reveringham_R30863_ctbtcml_2_fortysermons1685-646_chunk_005.jpg
big_letters/n/reveringham_R30863_ctbtcml_2_fortysermons1685-637_chunk_027.jpg
big_letters/n/reveringham_R30863_ctbtcml_2_fortysermons1685-677_chunk_005.jpg
big_letters/n/reveringham_R30863_ctbtcml_2_fortysermons1685-604_chunk_003.jpg
big_letters/n/reveringham_R30863_ctbtcml_2_fortysermons1685-592_chunk_002.jpg
big_letters/n/reveringham_R30863_ctbtcml_2_fortysermons1685-613_chunk_037.jpg
big_letters/n/reveringham_R30863_ctbtcml_2_fortysermons1685-657_chunk_004.jpg

big_letters/n/reveringham_R30863_ctbtcml_2_fortysermons1685-323_chunk_006.jpg
big_letters/n/reveringham_R30863_ctbtcml_2_fortysermons1685-378_chunk_004.jpg
big_letters/n/reveringham_R30863_ctbtcml_2_fortysermons1685-343_chunk_002.jpg
big_letters/n/reveringham_R30863_ctbtcml_2_fortysermons1685-304_chunk_005.jpg

big_letters/o/reveringham_R30863_ctbtcml_2_fortysermons1685-052_chunk_028.jpg
big_letters/o/reveringham_R30863_ctbtcml_2_fortysermons1685-021_chunk_060.jpg
big_letters/o/reveringham_R30863_ctbtcml_2_fortysermons1685-162_chunk_029.jpg
big_letters/o/reveringham_R30863_ctbtcml_2_fortysermons1685-007_chunk_049.jpg
big_letters/o/reveringham_R30863_ctbtcml_2_fortysermons1685-457_chunk_048.jpg
big_letters/o/reveringham_R30863_ctbtcml_2_fortysermons1685-486_chunk_016.jpg

big_letters/r/reveringham_R30863_ctbtcml_2_fortysermons1685-213_chunk_026.jpg
big_letters/r/reveringham_R30863_ctbtcml_2_fortysermons1685-465_chunk_028.jpg
big_letters/r/reveringham_R30863_ctbtcml_2_fortysermons1685-052_chunk_024.jpg
big_letters/r/reveringham_R30863_ctbtcml_2_fortysermons1685-486_chunk_017.jpg

big_letters/r/reveringham_R30863_ctbtcml_2_fortysermons1685-334_chunk_011.jpg
big_letters/r/reveringham_R30863_ctbtcml_2_fortysermons1685-420_chunk_009.jpg

big_letters/r/reveringham_R30863_ctbtcml_2_fortysermons1685-457_chunk_050.jpg
big_letters/r/reveringham_R30863_ctbtcml_2_fortysermons1685-007_chunk_047.jpg

big_letters/s/reveringham_R30863_ctbtcml_2_fortysermons1685-604_chunk_000.jpg
big_letters/s/reveringham_R30863_ctbtcml_2_fortysermons1685-592_chunk_001.jpg

big_letters/s/reveringham_R30863_ctbtcml_2_fortysermons1685-677_chunk_006.jpg
big_letters/s/reveringham_R30863_ctbtcml_2_fortysermons1685-688_chunk_001.jpg

big_letters/s/reveringham_R30863_ctbtcml_2_fortysermons1685-304_chunk_010.jpg
big_letters/s/reveringham_R30863_ctbtcml_2_fortysermons1685-397_chunk_012.jpg
big_letters/s/reveringham_R30863_ctbtcml_2_fortysermons1685-378_chunk_000.jpg
big_letters/s/reveringham_R30863_ctbtcml_2_fortysermons1685-323_chunk_002.jpg
big_letters/s/reveringham_R30863_ctbtcml_2_fortysermons1685-358_chunk_009.jpg
big_letters/s/reveringham_R30863_ctbtcml_2_fortysermons1685-343_chunk_007.jpg
"""

# The path to your existing Excel file.
# excel_file_path = "marimo/sample_review.xlsx" 
excel_file_path = "marimo/sample_review_updated_latex.xlsx" 

# Define all expected columns to ensure consistency
EXPECTED_COLUMNS = [
    'root_image', 'page_number', 'filename', 'image', 'is_root', 
    'gathering', 'letter', 'letter_class'
]

# --- END OF INPUTS ---

def encode_image_to_base64(file_path):
    """
    Opens a TIFF image, converts it to PNG in memory, 
    and returns its Base64 encoded Data URI string.
    """
    # Monkeypatch Image.open to auto-resize images to max height 144 (proportional scale)
    MAX_HEIGHT = 88
    _original_image_open = Image.open

    def _open_and_maybe_resize(fp, *args, **kwargs):
        im = _original_image_open(fp, *args, **kwargs)
        try:
            if im.height > MAX_HEIGHT:
                scale = MAX_HEIGHT / im.height
                new_w = max(1, int(im.width * scale))
                im = im.resize((new_w, MAX_HEIGHT), Image.Resampling.LANCZOS)
            return im
        except Exception:
            im.close()
            raise

    Image.open = _open_and_maybe_resize
    try:
        # Open the TIFF image with Pillow
        with Image.open(file_path) as img:
            # Create an in-memory binary stream
            with io.BytesIO() as in_mem_file:
                # Save the image to the in-memory stream as a PNG
                img.save(in_mem_file, format="PNG")
                # Get the binary content of the PNG
                image_bytes = in_mem_file.getvalue()
        
        # Encode the PNG bytes to Base64
        base64_bytes = base64.b64encode(image_bytes)
        base64_string = base64_bytes.decode('utf-8')
        
        # Prepend the Data URI scheme for PNG images
        return f"data:image/png;base64,{base64_string}"

    except FileNotFoundError:
        print(f"⚠️ WARNING: File not found at '{file_path}'. Image column will be empty for this entry.")
        return None
    except Exception as e:
        print(f"❌ ERROR: Could not process file '{file_path}'. Reason: {e}")
        return None

def update_spreadsheet(file_path, text_block):
    """
    Reads an Excel file, adds new entries with Base64 images, and saves to a new file.
    """
    try:
        df = pd.read_excel(file_path)
        print(f"✅ Successfully loaded '{file_path}' with {len(df)} rows.")
    except FileNotFoundError:
        print(f"ℹ️  File '{file_path}' not found. A new DataFrame will be created.")
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)

    if 'letter_class' in df.columns and pd.api.types.is_numeric_dtype(df['letter_class']) and not df.empty:
        next_id = int(df['letter_class'].max()) + 1
    else:
        next_id = 0
    
    groups = re.split(r'\n\s*\n', text_block.strip())
    
    new_rows = []
    
    for i, group in enumerate(groups):
        current_class_id = next_id + i
        file_paths = [line.strip() for line in group.strip().split('\n') if line.strip()]
        
        for path in file_paths:
            filename = os.path.basename(path)
            
            page_match = re.search(r'-(\d+)', filename)
            page_number = page_match.group(1) if page_match else None
            
            base64_image = encode_image_to_base64(path)
            letter_match = re.search(r'_([A-Z])_uc\.tif$', filename)
            if not letter_match:
                letter_match = re.search(r'/([a-z])/', str(path))
            letter = letter_match.group(1).upper() if letter_match else None
            
            new_rows.append({
                'filename': filename,
                'image': base64_image,
                'letter': letter,
                'letter_class': current_class_id,
                'root_image': filename,
                'page_number': int(page_number),
                'is_root': 0,
                'gathering': None,
            })

    if not new_rows:
        print("No new files to add.")
        return

    new_df = pd.DataFrame(new_rows)
    combined_df = pd.concat([df, new_df], ignore_index=True)
    final_df = combined_df.reindex(columns=EXPECTED_COLUMNS)
    # fill nan in is_root with 0
    final_df['is_root'] = final_df['is_root'].fillna(0).astype(bool)
    # delete the gathering column
    final_df = final_df.drop(columns=['gathering'])

    output_file = "marimo/sample_review_updated_latex_bigletter.xlsx"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_excel(output_file, index=False)
    
    print(f"\n🎉 Processed {len(groups)} new groups with a total of {len(new_rows)} files.")
    print(f"   New `letter_class` IDs start from {next_id}.")
    print(f"   Updated data saved successfully to '{output_file}'.")

if __name__ == "__main__":
    update_spreadsheet(excel_file_path, new_files_text)
