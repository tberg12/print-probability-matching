import sys
from PIL import Image
from tqdm import tqdm

def main():
    lines = sys.stdin.readlines()
    for line in tqdm(lines, desc="Processing images"):
        filepath = line.strip()
        if not filepath:
            continue
        try:
            with Image.open(filepath) as img:
                img.verify()  # Verify that this is indeed an image
            print(filepath)
        except Exception:
            continue

if __name__ == "__main__":
    main()