import os
import shutil
import csv
from matcher import get_transform
from PIL import Image
from torchvision.transforms import ToPILImage
import random

def process_matches_csv(csv_filepath, source_dir, output_base_dir, transform):
    with open(csv_filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        
        chars = []
        for row in reader:
            # Remove empty entries
            row = [filename for filename in row if filename.strip()]
            
            if not row:
                continue  # Skip empty rows
            
            # Extract character from the first filename
            char = row[0].split("_uc")[0][-1]
            chars.append(char)
            
            # Define output directory for the character
            char_test_dir = os.path.join(output_base_dir, char, "test")
            os.makedirs(char_test_dir, exist_ok=True)
            
            # Copy files to destination
            new_fps = []
            for filename in row:
                source_filepath = os.path.join(source_dir, f'char_{char}_uc', filename)
                
                if not os.path.exists(source_filepath):
                    print(f"Warning: {source_filepath} not found.")

                # preprocess the file using the transform
                img = Image.open(source_filepath)
                img = ToPILImage()(transform(img))
                print(f"Saving preprocessed {filename} to {char_test_dir}")
                new_fp = os.path.join(char_test_dir, filename)
                img.save(
                    new_fp
                )
                new_fps.append(new_fp)
            
            # Append the cleaned row to matches.csv in the new directory
            # matches_csv_path = os.path.join(char_test_dir, "matches.csv")
            # with open(matches_csv_path, "a", newline='') as outfile:
            #     writer = csv.writer(outfile)
            #     writer.writerow(new_fps)

        for char in set(chars):
            print(f"Creating mix negative background set for char {char}...")
            char_test_dir = os.path.join(output_base_dir, char, "test", "bg_imgs")
            os.makedirs(char_test_dir, exist_ok=True)
            char_dir = os.path.join(source_dir, f'char_{char}_uc')
            # grab 1000 random images from the char_dir, preprocess them, and save them to char_test_dir
            all_files = os.listdir(char_dir)
            random.seed(42)
            random.shuffle(all_files)
            new_fps = []
            for filename in all_files:
                if len(new_fps) >= 1000:
                    break
                source_filepath = os.path.join(char_dir, filename)
                try:
                    img = Image.open(source_filepath)
                except Exception as e:
                    print(f"Error opening {source_filepath}: {e}")
                    continue
                img = ToPILImage()(transform(img))
                new_fp = os.path.join(char_test_dir, filename)
                img.save(new_fp)
                new_fps.append(new_fp)
            # save new_fp list to mix_negative_background_set.txt
            mix_negative_background_set_path = os.path.join(char_test_dir, "mix_negative_background_set.txt")
            with open(mix_negative_background_set_path, "w") as outfile:
                for fp in new_fps:
                    outfile.write(fp + "\n")



if __name__ == "__main__":
    csv_filepath = "/graft2/code/nvog/git/matching/data/lockespinoza_matching_test_set/matches.csv"  # Path to the matches.csv file
    source_dir = "/graft2/code/nvog/git/matching/char_images3"  # Directory where images are stored
    output_base_dir = "/graft2/code/nvog/git/matching/data/lockespinoza_matching_test_set"  # Base output directory
    transform = get_transform()
    
    process_matches_csv(csv_filepath, source_dir, output_base_dir, transform)