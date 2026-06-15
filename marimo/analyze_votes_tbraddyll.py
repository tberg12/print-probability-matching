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

# Load the issue map for votesofthehouseofcommons1690
issue_map = pl.read_csv("votes_map_nikolai.csv")

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

print("="*80)
print("TBRADDYLL BOOKS ANALYSIS")
print("="*80)

# Books of interest - tbraddyll printed
tbraddyll_books_of_interest = [
    "religionandreason1688",
    "annalsofkingjamescharles1681"
]

for book in tbraddyll_books_of_interest:
    # Get matches between votes and this book
    matches = hi_matches.filter(
        (pl.col("book_name") == "votesofthehouseofcommons1690") |
        (pl.col("book_name") == book)
    )

    # Get shared letters
    votes_letters = set(matches.filter(pl.col("book_name") == "votesofthehouseofcommons1690")["letter_with_class"].unique().to_list())
    book_letters = set(matches.filter(pl.col("book_name") == book)["letter_with_class"].unique().to_list())
    shared_letters = votes_letters.intersection(book_letters)

    print(f"\n{book} (printer: tbraddyll)")
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

# Now let's see what other books share these same issues
print("\n" + "="*80)
print("CROSS-REFERENCE: What other books appear in these issues?")
print("="*80)

# Collect all issues from tbraddyll books
tbraddyll_issues = set()
for book in tbraddyll_books_of_interest:
    matches = hi_matches.filter(
        (pl.col("book_name") == "votesofthehouseofcommons1690") |
        (pl.col("book_name") == book)
    )
    votes_letters = set(matches.filter(pl.col("book_name") == "votesofthehouseofcommons1690")["letter_with_class"].unique().to_list())
    book_letters = set(matches.filter(pl.col("book_name") == book)["letter_with_class"].unique().to_list())
    shared_letters = votes_letters.intersection(book_letters)

    for letter in shared_letters:
        votes_with_letter = votes_matches.filter(pl.col("letter_with_class") == letter)
        issues = votes_with_letter["issue_number"].unique().to_list()
        tbraddyll_issues.update([i for i in issues if i is not None])

print(f"\nIssues with tbraddyll matches: {sorted(tbraddyll_issues)}")

# For each issue, find all books that have matches there
for issue in sorted(tbraddyll_issues):
    print(f"\n{'='*80}")
    print(f"ISSUE {issue}")
    print(f"{'='*80}")

    # Get all letters in this issue
    issue_letters = votes_matches.filter(pl.col("issue_number") == issue)["letter_with_class"].unique().to_list()

    # For each letter, find which books share it
    letter_book_map = {}
    for letter in issue_letters:
        # Find books that share this letter with votes
        all_with_letter = hi_matches.filter(pl.col("letter_with_class") == letter)
        books_with_letter = all_with_letter["book_name"].unique().to_list()
        # Remove votesofthehouseofcommons1690 itself
        books_with_letter = [b for b in books_with_letter if b != "votesofthehouseofcommons1690"]

        if books_with_letter:
            letter_book_map[letter] = books_with_letter

    # Print sorted by letter
    for letter in sorted(letter_book_map.keys()):
        books = letter_book_map[letter]
        print(f"  {letter}:")
        for book in sorted(books):
            # Get printer for this book
            book_printer = all_sample.filter(pl.col("book_name") == book)["printer_name"].unique().to_list()
            printer = book_printer[0] if book_printer else "unknown"
            print(f"    - {book} ({printer})")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nThis shows which issues of votesofthehouseofcommons1690 have matches to")
print("tbraddyll books, and what other books (with their printers) also appear")
print("in those same issues. Clustering of printers in specific issues suggests")
print("those issues were printed by that printer.")
