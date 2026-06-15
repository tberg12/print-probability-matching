"""
Preprocess old test sets for new larger unaligned data format
"""
from PIL import Image
from torchvision import transforms
from make_twin_dataset_from_splits import Binarizer, SquarePad
import argparse
from pathlib import Path
import numpy as np




def preprocess_image(args, input_path, output_path, transform):
    img = Image.open(input_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        Image.fromarray(np.array(transform(img))).save(output_path)
    except ValueError as e:
        print(f'{input_path.name}')
        return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set', type=str, choices=['leviathan', 'areopagitica', 'lockespinoza'], required=True)
    parser.add_argument('--unpadded_image_size', type=int, default=112, help='Size of output images')
    parser.add_argument('--padded_image_size', type=int, default=224, help='Size of output images')
    args = parser.parse_args()

    transform = transforms.Compose([
        SquarePad(),
        transforms.Resize((args.unpadded_image_size, args.unpadded_image_size)),
        transforms.Grayscale(num_output_channels=1),
        Binarizer(),
        transforms.Pad((args.padded_image_size - args.unpadded_image_size) // 2, fill=255, padding_mode='constant'),
    ])

    if args.test_set == 'leviathan':
        test_set = Path('/home/nvog/projects/git/anomaly-detection/matching/leviathan_matching_test_set')
        preproc_test_set = Path('/home/nvog/projects/git/anomaly-detection/matching/leviathan_matching_test_set_preprocessed')
        chars = 'ABCDEFGHKLMNPRTW'
        for c in chars:
            test_set_char = test_set/c/'test'
            preproc_test_set_char = preproc_test_set/c/'test'
            preproc_test_set_char.mkdir(parents=True, exist_ok=True)
            matches_unaligned = test_set_char/'matches_unaligned.csv'
            # deal with matches_unaligned.csv with contains multiple comma separated value file paths
            preproc_matches = preproc_test_set_char/'matches.csv'
            print('Preprocessing ground truth match sets.')
            with open(matches_unaligned) as f, open(preproc_matches, 'w') as g:
                for line in f:
                    img_paths = line.strip().split(',')
                    output_paths = []
                    for img_path in img_paths:
                        name = Path(img_path).name.replace('_aligned', '')
                        # find og_img_path
                        og_img_path = Path(f'/trunk2/nvog/char_images3_aligned/leviathanornaments_2022-05-15/leviathanornaments-{c}-0.001-out/char_{c}_uc')/name
                        assert og_img_path.exists(), f'{og_img_path} from {matches_unaligned} does not exist'
                        output_path = preproc_test_set_char/name
                        preprocess_image(args, og_img_path, output_path, transform)
                        output_paths.append(str(output_path))
                    print(','.join(output_paths), file=g)
            print('Done with ground truth match sets.')
            print('Now preprocessing background sets.')
            # deal with background sets
            bg_sets = [
                test_set_char/f'{setting}_negative_background_set.txt' 
                for setting in ['mix', 'strong', 'weak']
            ]
            preproc_bg_sets = [
                preproc_test_set_char/f'{setting}_negative_background_set.txt'
                for setting in ['mix', 'strong', 'weak']
            ]
            for i, bg_set in enumerate(bg_sets):
                with open(bg_set) as f, open(preproc_bg_sets[i], 'w') as g:
                    for line in f:
                        aligned_img_path = line.strip()
                        name = Path(aligned_img_path).name.replace('_aligned', '')
                        # find og_img_path
                        og_img_path = Path(f'/trunk2/nvog/char_images3_aligned/leviathanornaments_2022-05-15/leviathanornaments-{c}-0.001-out/char_{c}_uc')/name
                        assert og_img_path.exists(), f'{og_img_path} from {bg_set} does not exist'
                        output_path = preproc_test_set_char/'bg_imgs'/name
                        ret = preprocess_image(args, og_img_path, output_path, transform)
                        if ret:
                            print(str(output_path), file=g)
    
    elif args.test_set == 'lockespinoza':
        test_set = Path('/home/nvog/projects/git/anomaly-detection/matching/lockespinoza_matching_test_set')
        preproc_test_set = Path('/home/nvog/projects/git/anomaly-detection/matching/lockespinoza_matching_test_set_preprocessed')
        chars = 'BEFHKMNPQRW'
        for c in chars:
            test_set_char = test_set/c/'test'
            preproc_test_set_char = preproc_test_set/c/'test'
            preproc_test_set_char.mkdir(parents=True, exist_ok=True)
            matches_aligned = test_set_char/'matches.csv'
            # deal with matches_unaligned.csv with contains multiple comma separated value file paths
            preproc_matches = preproc_test_set_char/'matches.csv'
            print('Preprocessing ground truth match sets.')
            with open(matches_aligned) as f, open(preproc_matches, 'w') as g:
                for line in f:
                    img_paths = line.strip().split(',')
                    output_paths = []
                    for img_path in img_paths:
                        name = Path(img_path).name.replace('_aligned', '')
                        # find og_img_path
                        book_name = name.split('-')[0]
                        og_img_path = Path(f'/trunk2/print-probability/char_{c}_uc')/name
                        assert og_img_path.exists(), f'{og_img_path} from {matches_aligned} does not exist'
                        output_path = preproc_test_set_char/name
                        preprocess_image(args, og_img_path, output_path, transform)
                        output_paths.append(str(output_path))
                    print(','.join(output_paths), file=g)
            print('Done with ground truth match sets.')
            print('Now preprocessing background sets.')
            # deal with background sets
            bg_sets = [
                test_set_char/f'{setting}_negative_background_set.txt' 
                for setting in ['mix']
            ]
            preproc_bg_sets = [
                preproc_test_set_char/f'{setting}_negative_background_set.txt'
                for setting in ['mix']
            ]
            for i, bg_set in enumerate(bg_sets):
                with open(bg_set) as f, open(preproc_bg_sets[i], 'w') as g:
                    for line in f:
                        aligned_img_path = line.strip()
                        name = Path(aligned_img_path).name.replace('_aligned', '')
                        # find og_img_path
                        og_img_path = Path(f'/trunk2/print-probability/char_{c}_uc')/name
                        assert og_img_path.exists(), f'{og_img_path} from {bg_set} does not exist'
                        output_path = preproc_test_set_char/'bg_imgs'/name
                        ret = preprocess_image(args, og_img_path, output_path, transform)
                        if ret:
                            print(str(output_path), file=g)

    elif args.test_set == 'areopagitica':
        # NOTE: these are aligned
        test_set = '/home/nvog/projects/git/anomaly-detection/matching/areopagitica_matching_test_set'
        raise NotImplementedError