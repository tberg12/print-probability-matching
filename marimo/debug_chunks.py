import polars as pl
from utils import load_and_prepare_sample

# Load with debug mode enabled
sample = load_and_prepare_sample('LockeSpinozaMatches_processed.csv', split_parts=True, debug=True)

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

# Show final distribution
print("\nFinal book_name distribution:")
print(sample.group_by("book_name").agg(pl.count()).sort("book_name"))

# Check specifically for fortysermons1685 (no part suffix)
no_part = sample.filter(pl.col("book_name") == "fortysermons1685")
if len(no_part) > 0:
    print(f"\nRecords with 'fortysermons1685' (no part): {len(no_part)}")
    print("Sample records:")
    print(no_part.select(["filename", "page_number", "book_name"]).head(10))
