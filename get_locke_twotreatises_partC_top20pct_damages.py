


damage_csv = 'damage_classifier_output/evaluate/char_images3_REDO_valid_2025-05-27_damagepredictions_top20pctbychar.csv'
#damage_csv = 'damage_classifier_output/evaluate/char_images3_REDO_valid_2025-05-27_damagepredictions_top5pctbychar.csv'

for line in open(damage_csv):
    if 'twotreatises' in line:
        pgnum = line.strip().split('-')[1].split('_')[0]
        if 257 <= int(pgnum) <= 464:
            if 'I_uc' not in line:
                print(line.strip())
