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

# Apply book-level printer overrides
def get_printer_for_book(book_name):
    if "fortysermons1685" in book_name:
        return "everingham"
    elif "twotreatises" in book_name.lower():
        return "anon"
    elif "religionandreason1688" in book_name:
        return "tbraddyll"
    return None

# Add a column for book-level printer (for display purposes)
all_sample = all_sample.with_columns(
    book_printer=pl.col("book_name").map_elements(
        lambda x: get_printer_for_book(x) or all_sample.filter(pl.col("book_name") == x)["printer_name"].first(),
        return_dtype=pl.Utf8
    )
)

# Load the issue map for votesofthehouseofcommons1690
issue_map = pl.read_csv("votes_map_nikolai.csv")
print("="*80)
print("Loaded issue map:")
print(issue_map.head(10))
print(f"Total mappings: {len(issue_map)}")

# Create a dictionary for page_num -> issue mapping
page_to_issue = dict(zip(issue_map["page_num"].to_list(), issue_map["issue"].to_list()))

# Add issue_number column based on page_number
all_sample = all_sample.with_columns(
    issue_number=pl.when(pl.col("book_name") == "votesofthehouseofcommons1690")
    .then(
        pl.col("page_number").map_elements(
            lambda page: page_to_issue.get(page, None) if page else None,
            return_dtype=pl.Int64
        )
    )
    .otherwise(None)
)

# Filter for high confidence matches only
hi_matches = all_sample.filter(pl.col("confidence_level") == "hi")

# Filter for matches involving votesofthehouseofcommons1690
votes_matches = hi_matches.filter(pl.col("book_name") == "votesofthehouseofcommons1690")

# Get unique books that match with votes
books_matching_votes = votes_matches["book_name"].unique().to_list()

print("="*80)
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
        print(f"\n  Issue breakdown (page in parentheses):")
        for letter in sorted(shared_letters):
            votes_with_letter = votes_matches.filter(pl.col("letter_with_class") == letter)
            issues = votes_with_letter["issue_number"].unique().to_list()
            pages = votes_with_letter["page_number"].unique().to_list()
            if issues and pages:
                issue_page_pairs = sorted([(i, p) for i, p in zip(issues, pages) if i is not None])
                issue_str = ', '.join([f"{issue}(p{page})" for issue, page in issue_page_pairs])
                print(f"    {letter}: issues {issue_str}")
            else:
                print(f"    {letter}: unknown issue")

# Now get matches with tbraddyll and everingham books
print("\n" + "="*80)
print("MATCHES WITH TBRADDYLL BOOKS")
print("="*80)

# Get all tbraddyll books
tbraddyll_books = hi_matches.filter(pl.col("printer_name") == "tbraddyll")["book_name"].unique().to_list()

for book in tbraddyll_books:
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
    print(f"\n  Issue breakdown (page in parentheses):")
    for letter in sorted(shared_letters):
        votes_with_letter = votes_matches.filter(pl.col("letter_with_class") == letter)
        issues = votes_with_letter["issue_number"].unique().to_list()
        pages = votes_with_letter["page_number"].unique().to_list()
        if issues and pages:
            issue_page_pairs = sorted([(i, p) for i, p in zip(issues, pages) if i is not None])
            issue_str = ', '.join([f"{issue}(p{page})" for issue, page in issue_page_pairs])
            print(f"    {letter}: issues {issue_str}")
        else:
            print(f"    {letter}: unknown issue")

print("\n" + "="*80)
print("MATCHES WITH EVERINGHAM BOOKS")
print("="*80)

# Get all everingham books
everingham_books = hi_matches.filter(pl.col("printer_name") == "everingham")["book_name"].unique().to_list()

for book in everingham_books:
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
    print(f"\n  Issue breakdown (page in parentheses):")
    for letter in sorted(shared_letters):
        votes_with_letter = votes_matches.filter(pl.col("letter_with_class") == letter)
        issues = votes_with_letter["issue_number"].unique().to_list()
        pages = votes_with_letter["page_number"].unique().to_list()
        if issues and pages:
            issue_page_pairs = sorted([(i, p) for i, p in zip(issues, pages) if i is not None])
            issue_str = ', '.join([f"{issue}(p{page})" for issue, page in issue_page_pairs])
            print(f"    {letter}: issues {issue_str}")
        else:
            print(f"    {letter}: unknown issue")

print("\n" + "="*80)
print("CROSS-REFERENCE SUMMARY")
print("="*80)
print("\nTo determine printer attribution:")
print("  - Look for letters shared between anon books and votes")
print("  - Check which issues those letters appear in")
print("  - Compare with letters/issues from known tbraddyll and everingham books")
print("  - Overlapping issues suggest the same printer")
print("\nNote: Issue numbers are from votes_map_nikolai.csv mapping.")
