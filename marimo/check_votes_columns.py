import polars as pl
from utils import load_and_prepare_sample

# Load data
all_sample = load_and_prepare_sample('LockeSpinozaMatches_processed.csv')

# Filter for votesofthehouseofcommons1690
votes_data = all_sample.filter(pl.col("book_name") == "votesofthehouseofcommons1690")

print("All columns in the dataset:")
print("="*80)
for col in all_sample.columns:
    print(f"  {col}")

print("\n" + "="*80)
print("Sample rows from votesofthehouseofcommons1690:")
print("="*80)
print(votes_data.select(["book_name", "filename", "root_image", "letter_with_class", "confidence_level"]).head(10))

# Check if there's an issue-related column
print("\n" + "="*80)
print("Looking for issue-related columns:")
print("="*80)
issue_cols = [col for col in all_sample.columns if 'issue' in col.lower()]
if issue_cols:
    print(f"Found: {issue_cols}")
    for col in issue_cols:
        print(f"\n{col} sample values:")
        print(votes_data[col].unique().head(20))
else:
    print("No columns with 'issue' in the name")
