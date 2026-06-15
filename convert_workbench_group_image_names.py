"""
Convert worbench group image names to the standard char_images3 format

Only works for locke, spinoza, and forty sermons matches now.
"""
from pathlib import Path
import glob
import shutil
import sys
from PIL import Image


# (base) ➜  ims git:(master) ✗ for f in *\(*.jpg; do echo $f | python -c "import sys; print(sys.stdin.readline().strip())"; done
workbench_filenames = """
A_uc - (48072) Forty sermons. preached by the... p. 606-s l. 13 c. 9.jpg
A_uc - (48072) Forty sermons. preached by the... p. 664-s l. 3 c. 22.jpg
A_uc - (48072) Forty sermons. preached by the... p. 675-s l. 23 c. 7.jpg
A_uc - (48072) Forty sermons. preached by the... p. 681-s l. 52 c. 39.jpg
A_uc - (48072) Forty sermons. preached by the... p. 690-s l. 35 c. 23.jpg
A_uc - (48072) Forty sermons. preached by the... p. 700-s l. 37 c. 53.jpg
A_uc - (63390) A treatise partly theological,... p. 224-s l. 28 c. 21.jpg
A_uc - (63390) A treatise partly theological,... p. 225-s l. 1 c. 17.jpg
A_uc - (96077) Two treatises of government in... p. 138-s l. 1 c. 19.jpg
A_uc - (96077) Two treatises of government in... p. 161-s l. 39 c. 19.jpg
A_uc - (96077) Two treatises of government in... p. 47-s l. 5 c. 1.jpg
B_uc - (48072) Forty sermons. preached by the... p. 532-s l. 37 c. 1.jpg
B_uc - (96077) Two treatises of government in... p. 178-s l. 44 c. 6.jpg
B_uc - (96077) Two treatises of government in... p. 360-s l. 30 c. 6.jpg
E_uc - (48072) Forty sermons. preached by the... p. 512-s l. 7 c. 37.jpg
E_uc - (63390) A treatise partly theological,... p. 166-s l. 36 c. 29.jpg
E_uc - (96077) Two treatises of government in... p. 111-s l. 7 c. 6.jpg
E_uc - (96077) Two treatises of government in... p. 187-s l. 40 c. 22.jpg
E_uc - (96077) Two treatises of government in... p. 391-s l. 20 c. 10.jpg
F_uc - (48072) Forty sermons. preached by the... p. 597-s l. 23 c. 21.jpg
F_uc - (48072) Forty sermons. preached by the... p. 639-s l. 20 c. 31.jpg
F_uc - (63390) A treatise partly theological,... p. 206-s l. 30 c. 8.jpg
F_uc - (96077) Two treatises of government in... p. 84-s l. 35 c. 2.jpg
G_uc - (48072) Forty sermons. preached by the... p. 202-s l. 13 c. 1.jpg
G_uc - (48072) Forty sermons. preached by the... p. 256-s l. 9 c. 61.jpg
G_uc - (48072) Forty sermons. preached by the... p. 333-s l. 29 c. 7.jpg
G_uc - (96077) Two treatises of government in... p. 343-s l. 49 c. 4.jpg
G_uc - (96077) Two treatises of government in... p. 350-s l. 19 c. 18.jpg
H_uc - (48072) Forty sermons. preached by the... p. 528-s l. 26 c. 63.jpg
H_uc - (48072) Forty sermons. preached by the... p. 55-s l. 38 c. 7.jpg
H_uc - (48072) Forty sermons. preached by the... p. 558-s l. 52 c. 27.jpg
H_uc - (48072) Forty sermons. preached by the... p. 619-s l. 41 c. 74.jpg
H_uc - (63390) A treatise partly theological,... p. 154-s l. 28 c. 1.jpg
H_uc - (63390) A treatise partly theological,... p. 166-s l. 7 c. 1.jpg
H_uc - (63390) A treatise partly theological,... p. 181-s l. 3 c. 37.jpg
H_uc - (63390) A treatise partly theological,... p. 368-s l. 1 c. 41.jpg
H_uc - (63390) A treatise partly theological,... p. 404-s l. 26 c. 1.jpg
H_uc - (96077) Two treatises of government in... p. 146-s l. 41 c. 23.jpg
H_uc - (96077) Two treatises of government in... p. 61-s l. 38 c. 18.jpg
N_uc - (48072) Forty sermons. preached by the... p. 317-s l. 41 c. 33.jpg
N_uc - (48072) Forty sermons. preached by the... p. 448-s l. 41 c. 30.jpg
N_uc - (48072) Forty sermons. preached by the... p. 599-s l. 30 c. 26.jpg
N_uc - (48072) Forty sermons. preached by the... p. 657-s l. 35 c. 75.jpg
N_uc - (63390) A treatise partly theological,... p. 202-s l. 11 c. 1.jpg
N_uc - (63390) A treatise partly theological,... p. 261-s l. 2 c. 31.jpg
N_uc - (63390) A treatise partly theological,... p. 57-s l. 1 c. 28.jpg
N_uc - (96077) Two treatises of government in... p. 104-s l. 12 c. 12.jpg
O_uc - (63390) A treatise partly theological,... p. 15-s l. 18 c. 9.jpg
O_uc - (96077) Two treatises of government in... p. 175-s l. 28 c. 11.jpg
P_uc - (63390) A treatise partly theological,... p. 253-s l. 18 c. 37.jpg
P_uc - (96077) Two treatises of government in... p. 66-s l. 17 c. 8.jpg
Q_uc - (48072) Forty sermons. preached by the... p. 606-s l. 40 c. 46.jpg
Q_uc - (96077) Two treatises of government in... p. 32-s l. 17 c. 20.jpg
R_uc - (48072) Forty sermons. preached by the... p. 680-s l. 42 c. 12.jpg
R_uc - (96077) Two treatises of government in... p. 96-s l. 16 c. 13.jpg
T_uc - (48072) Forty sermons. preached by the... p. 308-s l. 17 c. 3.jpg
T_uc - (63390) A treatise partly theological,... p. 353-s l. 9 c. 7.jpg
"""

# load in csv of matches per row separated by commas
csv_path = "data/lockespinoza_matching_test_set/matches_unconverted.csv"

csv_str = open(csv_path).read()


for line in workbench_filenames.split('\n'):
    line = line.strip()
    if not line:
        continue

    sl = line.split()
    c = sl[0][0]
    p = sl[sl.index('p.') + 1].replace('-s', '')
    l = sl[sl.index('l.') + 1]
    d = sl[sl.index('c.') + 1].replace('.jpg', '')
    
    if "Forty sermons" in line:
        book = 'reveringham_R30863_ctbtcml_2_fortysermons*1685'
    elif "A treatise partly" in line:
        book = 'anon_R21627_gw_8_spinozatheologicalpolitical*1689'
    elif "Two treatises" in line:
        book = 'anon_R2930_iur_8_twotreatisesofgov*1690'
    else:
        raise NotImplementedError
    
    converted_fname_wildcard = (
        f"{book}-*{p}_page1rline{l}_char{d}_{c}_uc*"
    )

    # print(converted_fname_wildcard)

    # TODO: look for converted_fname_wildcard at path_to_search
    glob_path = f'/graft2/code/nvog/git/matching/char_images3/char_*_uc/{converted_fname_wildcard}'
    dest_path = Path('lists/ims')
    matches = list(glob.glob(glob_path))
    if len(matches) == 0:
        print(f"No matches found for wildcard: {converted_fname_wildcard}", file=sys.stderr)
    elif len(matches) > 1:
        # print(f"Multiple matches found for wildcard: {converted_fname_wildcard}: {matches}", file=sys.stderr)
        # print one without 'REDO' in the name
        matches = [m for m in matches if 'REDO' not in m]
        if len(matches) > 1:
            print(f"Multiple matches found for wildcard: {converted_fname_wildcard}: {matches}", file=sys.stderr)
            continue
        else:
            print(matches[0])
    else:
        # print(f"Match found for wildcard {converted_fname_wildcard}: {matches[0]}")
        print(matches[0])

    # if 'N' in line:
    #     import ipdb; ipdb.set_trace()
    # # import ipdb; ipdb.set_trace()
    csv_str = csv_str.replace(line.replace('.jpg', ''), matches[0])

    # convert file to jpg with max quality and save to list/ims/
    # Image.open(matches[0]).save(str(
    #     dest_path / (
    #         Path(matches[0]).name.replace('tif', 'jpg')
    #     )
    # ), quality=100)

csv_str = csv_str.replace('.jpg', '.tif')
csv_str = csv_str.replace('"', '')

print('\n\n\n')
print('new csv:')
print(csv_str)
