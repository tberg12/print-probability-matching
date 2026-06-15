import argparse
from pathlib import Path
from shutil import copy2


def parse_filename_identifier(filename):
    l = filename.split('_')
    # import ipdb; ipdb.set_trace()
    # perturb<id>, timestamp, damage_type, xloc, yloc 
    # return l[10], l[12], l[14], l[15], l[16]
    # print(len(l))
    assert len(l) >= 12, f'{filename} has less than 12 parts'
    return l[9], l[11]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('splits_dir', type=str)  #/projects/nvog/synthetic/data/synthetic/shakespeare_2021-09-28/${c}_twin_20_2021-10-08 , which contains train,valid,test
    parser.add_argument('dest_dir', type=str)  #/projects/nvog/synthetic/data/synthetic/shakespeare_2021-09-28/${c}_twin_20_2021-10-08/fake_match_sets , which will contain train,valid,test
    parser.add_argument('--num_pairs', type=int, default=None)
    parser.add_argument('--splits', nargs='+', type=str, choices=['train', 'valid', 'test'], default=['train', 'valid', 'test'])
    args = parser.parse_args()





    splits_dir = Path(args.splits_dir)
    dest_dir = Path(args.dest_dir)
    for split in args.splits:
        pairs = []
        split_dir = splits_dir/split
        # group the matches next to each other
        # images = sorted(split_dir.glob("*.tif"), key=lambda x: parse_filename_identifier(x.name)[:2])
        images = sorted(split_dir.glob("*.tif"), key=lambda x: parse_filename_identifier(x.name)[1])
        assert len(images) > 0, f'{split_dir} has no images'
        # use all images if num_pairs is None
        num_pairs = args.num_pairs if args.num_pairs is not None else len(images) // 2
        # get num_pairs matches, cp to dest dir, and dump to matches.csv
        for i in range(num_pairs * 2):
            if i % 2 != 0:
                # check that the damage is the same for the filename pairs
                # _, _, damage_type_a, xloc_a, yloc_a = parse_filename_identifier(images[i-1].name)
                # _, _, damage_type_b, xloc_b, yloc_b = parse_filename_identifier(images[i].name)
                perturb_id_a, timestamp_a = parse_filename_identifier(images[i-1].name)
                perturb_id_b, timestamp_b = parse_filename_identifier(images[i].name)
                assert perturb_id_a == perturb_id_b
                assert timestamp_a == timestamp_b
                # assert damage_type_a == damage_type_b
                # assert xloc_a == xloc_b
                # assert yloc_a == yloc_b
                pairs.append((images[i-1].name, images[i].name))
        # cp files over into the correct split in dest_dir
        split_dest_dir = dest_dir/split
        split_dest_dir.mkdir(exist_ok=True, parents=True)
        for pair in pairs:
            assert len(pair) == 2, f'{pair} != 2 elements'
            copy2(split_dir/pair[0], split_dest_dir/pair[0])
            copy2(split_dir/pair[1], split_dest_dir/pair[1])
        # dump pairs to matches.csv
        matches_csv = split_dest_dir/'matches.csv'
        with open(matches_csv, 'w') as f:
            for pair in pairs:
                print(','.join(pair), file=f)

