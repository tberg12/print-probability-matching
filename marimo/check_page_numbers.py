import polars as pl
from utils import load_and_prepare_sample

# Load data
all_sample = load_and_prepare_sample('LockeSpinozaMatches_processed.csv')

# Filter for votesofthehouseofcommons1690
votes_data = all_sample.filter(pl.col("book_name") == "votesofthehouseofcommons1690")

print("Sample data with page_number:")
print("="*80)
print(votes_data.select(["book_name", "filename", "page_number", "letter_with_class", "confidence_level"]).head(20))

print("\n" + "="*80)
print("Unique page_numbers:")
print("="*80)
page_nums = sorted([int(p) for p in votes_data["page_number"].unique().to_list() if p])
print(f"Range: {min(page_nums)} to {max(page_nums)}")
print(f"Count: {len(page_nums)}")
print(f"First 20: {page_nums[:20]}")
