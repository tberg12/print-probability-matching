import polars as pl
from utils import load_and_prepare_sample

# Load data
all_sample = load_and_prepare_sample('LockeSpinozaMatches_processed.csv')

# Extract printer name from root_image column
all_sample = all_sample.with_columns(
    printer_name=pl.col("root_image").str.split("_").list.get(0)
)

# Apply specific printer name rules based on keywords in root_image
all_sample = all_sample.with_columns(
    printer_name=pl.when(pl.col("root_image").str.contains("fortysermons1685"))
    .then(pl.lit("everingham"))
    .when(pl.col("root_image").str.contains("treatises"))
    .then(pl.lit("anon"))
    .when(pl.col("root_image").str.contains("sermons"))
    .then(pl.lit("everingham"))
    .when(pl.col("root_image").str.contains("theological"))
    .then(pl.lit("anon"))
    .otherwise(pl.col("printer_name"))
)

# Normalize printer names - treat "reveringham" and "everingham" as the same
all_sample = all_sample.with_columns(
    printer_name=pl.when(pl.col("printer_name").str.to_lowercase() == "reveringham")
    .then(pl.lit("everingham"))
    .otherwise(pl.col("printer_name"))
)

# Create issue map for votesofthehouseofcommons1690
# Map page numbers to sequential issue numbers
votes_data = all_sample.filter(pl.col("book_name") == "votesofthehouseofcommons1690")
unique_pages = sorted(votes_data["page_number"].unique().to_list())
page_to_issue = {page: idx + 1 for idx, page in enumerate(unique_pages)}

print("="*80)
print("PAGE TO ISSUE MAPPING for votesofthehouseofcommons1690")
print("="*80)
print(f"Total issues: {len(page_to_issue)}")
print("\nPage -> Issue:")
for page, issue in sorted(page_to_issue.items())[:20]:
    print(f"  Page {page:3d} -> Issue {issue:2d}")
if len(page_to_issue) > 20:
    print(f"  ... ({len(page_to_issue) - 20} more)")

# Add issue_number column
all_sample = all_sample.with_columns(
    issue_number=pl.col("page_number").map_elements(
        lambda page: page_to_issue.get(page, None) if page else None,
        return_dtype=pl.Int64
    )
)

# Filter for high confidence matches only
hi_matches = all_sample.filter(pl.col("confidence_level") == "hi")

# Filter for matches involving votesofthehouseofcommons1690
votes_matches = hi_matches.filter(pl.col("book_name") == "votesofthehouseofcommons1690")

print("\n" + "="*80)
print("HIGH CONFIDENCE MATCHES FOR votesofthehouseofcommons1690")
print("="*80)

# Books of interest - anon printed
anon_books_of_interest = [
    "twotreatisesofgov1690_partA",
    "spinozatheologicalpolitical1689_partA"
]

print("\n" + "="*80)
print("ANON BOOKS OF INTEREST")
print("="*80)

for book in anon_books_of_interest:
    # Get matches between votes and this book
    matches = hi_matches.filter(
        (pl.col("book_name") == "votesofthehouseofcommons1690") |
        (pl.col("book_name") == book)
    )

    # Get shared letters
    votes_letters = set(matches.filter(pl.col("book_name") == "votesofthehouseofcommons1690")["letter_with_class"].unique().to_list())
    book_letters = set(matches.filter(pl.col("book_name") == book)["letter_with_class"].unique().to_list())
    shared_letters = votes_letters.intersection(book_letters)

    print(f"\n{book} (printer: anon)")
    print(f"  Shared high confidence letters: {len(shared_letters)}")

    if shared_letters:
        print(f"  Letters: {', '.join(sorted(shared_letters))}")

        # For each shared letter, show which issues it appears in
        print(f"\n  Issue breakdown (page number in parentheses):")
        for letter in sorted(shared_letters):
            votes_with_letter = votes_matches.filter(pl.col("letter_with_class") == letter)
            issues = votes_with_letter["issue_number"].unique().to_list()
            pages = votes_with_letter["page_number"].unique().to_list()
            if issues:
                issue_page_pairs = sorted(zip(issues, pages))
                issue_str = ', '.join([f"{issue}(p{page})" for issue, page in issue_page_pairs])
                print(f"    {letter}: issues {issue_str}")

# Now get matches with tbraddyll and everingham books
print("\n" + "="*80)
print("MATCHES WITH TBRADDYLL BOOKS")
print("="*80)

# Get all tbraddyll books
tbraddyll_books = hi_matches.filter(pl.col("printer_name") == "tbraddyll")["book_name"].unique().to_list()

for book in sorted(tbraddyll_books):
    if book == "votesofthehouseofcommons1690":
        continue

    # Get matches between votes and this book
    matches = hi_matches.filter(
        (pl.col("book_name") == "votesofthehouseofcommons1690") |
        (pl.col("book_name") == book)
    )

    # Get shared letters
    votes_letters = set(matches.filter(pl.col("book_name") == "votesofthehouseofcommons1690")["letter_with_class"].unique().to_list())
    book_letters = set(matches.filter(pl.col("book_name") == book)["letter_with_class"].unique().to_list())
    shared_letters = votes_letters.intersection(book_letters)

    if not shared_letters:
        continue

    print(f"\n{book} (printer: tbraddyll)")
    print(f"  Shared high confidence letters: {len(shared_letters)}")
    print(f"  Letters: {', '.join(sorted(shared_letters))}")

    # For each shared letter, show which issues it appears in
    print(f"\n  Issue breakdown (page number in parentheses):")
    for letter in sorted(shared_letters):
        votes_with_letter = votes_matches.filter(pl.col("letter_with_class") == letter)
        issues = votes_with_letter["issue_number"].unique().to_list()
        pages = votes_with_letter["page_number"].unique().to_list()
        if issues:
            issue_page_pairs = sorted(zip(issues, pages))
            issue_str = ', '.join([f"{issue}(p{page})" for issue, page in issue_page_pairs])
            print(f"    {letter}: issues {issue_str}")

print("\n" + "="*80)
print("MATCHES WITH EVERINGHAM BOOKS")
print("="*80)

# Get all everingham books
everingham_books = hi_matches.filter(pl.col("printer_name") == "everingham")["book_name"].unique().to_list()

for book in sorted(everingham_books):
    if book == "votesofthehouseofcommons1690":
        continue

    # Get matches between votes and this book
    matches = hi_matches.filter(
        (pl.col("book_name") == "votesofthehouseofcommons1690") |
        (pl.col("book_name") == book)
    )

    # Get shared letters
    votes_letters = set(matches.filter(pl.col("book_name") == "votesofthehouseofcommons1690")["letter_with_class"].unique().to_list())
    book_letters = set(matches.filter(pl.col("book_name") == book)["letter_with_class"].unique().to_list())
    shared_letters = votes_letters.intersection(book_letters)

    if not shared_letters:
        continue

    print(f"\n{book} (printer: everingham)")
    print(f"  Shared high confidence letters: {len(shared_letters)}")
    print(f"  Letters: {', '.join(sorted(shared_letters))}")

    # For each shared letter, show which issues it appears in
    print(f"\n  Issue breakdown (page number in parentheses):")
    for letter in sorted(shared_letters):
        votes_with_letter = votes_matches.filter(pl.col("letter_with_class") == letter)
        issues = votes_with_letter["issue_number"].unique().to_list()
        pages = votes_with_letter["page_number"].unique().to_list()
        if issues:
            issue_page_pairs = sorted(zip(issues, pages))
            issue_str = ', '.join([f"{issue}(p{page})" for issue, page in issue_page_pairs])
            print(f"    {letter}: issues {issue_str}")

print("\n" + "="*80)
print("CROSS-REFERENCE SUMMARY")
print("="*80)
print("\nTo determine printer attribution:")
print("  - Look for letters shared between anon books and votes")
print("  - Check which issues those letters appear in")
print("  - Compare with letters/issues from known tbraddyll and everingham books")
print("  - Overlapping issues suggest the same printer")
print("\nNote: Issue numbers are sequential (1-N) based on sorted page numbers.")
print("      Page numbers are shown in parentheses for reference.")
