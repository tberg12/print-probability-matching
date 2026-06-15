import polars as pl
from utils import load_and_prepare_sample

# Load data
all_sample = load_and_prepare_sample('LockeSpinozaMatches_processed.csv')

# Filter for votesofthehouseofcommons1690
votes_data = all_sample.filter(pl.col("book_name") == "votesofthehouseofcommons1690")

# Show sample filenames
print("Sample filenames from votesofthehouseofcommons1690:")
print("="*80)
sample_filenames = votes_data["filename"].unique().to_list()[:20]
for i, filename in enumerate(sample_filenames, 1):
    print(f"{i}. {filename}")

print(f"\nTotal unique filenames: {len(votes_data['filename'].unique())}")

# Also check root_image column
print("\n" + "="*80)
print("Sample root_image values:")
print("="*80)
sample_root_images = votes_data["root_image"].unique().to_list()[:20]
for i, root_image in enumerate(sample_root_images, 1):
    print(f"{i}. {root_image}")
