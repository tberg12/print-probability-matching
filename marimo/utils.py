import polars as pl

# hardcoded letter_with_class filter
FILTER_LETTER_WITH_CLASS = [
    "A44",
    "P43",
]

SPLIT_RANGES = {
            "spinozatheologicalpolitical1689": {
                'A': [(1, 274), (386, 485)],
                'B': [(274, 384)],
                'C': [(384, 386)],
            },
            "twotreatisesofgov1690": {
                'A': [(21, 255)],
                'D': [(255, 257)],
                'E': [(257, 464)],
            },
            "fortysermons1685": {
                'PRE': [(1, 12)],
                'B': [(12, 457)],
                'C': [(457, 585)],
                'A': [(585, 713)],
                'F': [(713, 731)],
            },
        }

# Create letter_with_class for all samples first and extract book name
def extract_book_name(filename):
    """Extract book name from filename"""
    try:
        return filename.split('_')[4].split('-')[0].replace('REDO', '').replace('.', '')
    except IndexError:
        parts = filename.split()
        if len(parts) >= 5:
            return '_'.join(parts[3:5]).lower().replace('.', '')
        else:
            return 'unknown'

def load_and_prepare_sample(filename="sample_review_updated_latex_bigletter_dedup.xlsx", split_parts=True, debug=False, only_hi_matches=False):
    """Load and prepare sample data

    Args:
        filename: Path to data file (.xlsx or .csv)
        split_parts: Whether to split books into parts based on SPLIT_RANGES
        debug: Enable debug output
        only_hi_matches: If True, filter to only include high confidence matches (default: False)
    """
    if filename.endswith('.xlsx'):
        sample = pl.read_excel(filename)
    elif filename.endswith('.csv'):
        sample = pl.read_csv(filename)
    else:
        raise ValueError("Unsupported file format. Please provide a .xlsx or .csv file.")

    sample = sample.with_columns(
        letter_with_class=pl.concat_str([pl.col("letter"), pl.col("letter_class")]),
        book_name=pl.col("filename").map_elements(extract_book_name, return_dtype=pl.Utf8)
    )

    # Filter to only high confidence matches if requested
    if only_hi_matches:
        if "confidence_level" in sample.columns:
            original_count = len(sample)
            sample = sample.filter(pl.col("confidence_level") == "hi")
            filtered_count = len(sample)
            if debug:
                print(f"DEBUG: Filtered to high confidence matches only: {original_count} -> {filtered_count} records")
        elif debug:
            print("WARNING: only_hi_matches=True but 'confidence_level' column not found in data")
    
    if debug:
        print("DEBUG: Initial book names extracted:")
        print(sample.select(["filename", "book_name", "page_number"]).head(10))
        print("\nDEBUG: Unique book names before standardization:")
        print(sorted(sample["book_name"].unique().to_list()))
    
    sample = sample.with_columns(
        # Standardize book names right after extraction
        book_name=pl
        .when(pl.col("book_name").str.to_lowercase() == "a_treatise")
        .then(pl.lit("spinozatheologicalpolitical1689"))
        .when(pl.col("book_name").str.to_lowercase() == "two_treatises")
        .then(pl.lit("twotreatisesofgov1690"))
        .when(pl.col("book_name").str.to_lowercase() == "forty_sermons")
        .then(pl.lit("fortysermons1685"))
        .otherwise(pl.col("book_name"))
    )
    
    if debug:
        print("\nDEBUG: Unique book names after standardization:")
        print(sorted(sample["book_name"].unique().to_list()))
        
        # Check fortysermons1685 specifically
        forty_sermons = sample.filter(pl.col("book_name") == "fortysermons1685")
        if len(forty_sermons) > 0:
            print("\nDEBUG: fortysermons1685 records:")
            print(f"  Total count: {len(forty_sermons)}")
            print(f"  Chunk files: {forty_sermons.filter(pl.col('filename').str.contains('chunk')).shape[0]}")
            print(f"  Non-chunk files: {forty_sermons.filter(~pl.col('filename').str.contains('chunk')).shape[0]}")
            print(f"  Page number range: {forty_sermons['page_number'].min()} - {forty_sermons['page_number'].max()}")
            print("\nDEBUG: Sample chunk filenames:")
            chunk_files = forty_sermons.filter(pl.col('filename').str.contains('chunk'))
            if len(chunk_files) > 0:
                print(chunk_files.select(["filename", "page_number"]).head(5))
    
    # apply hardcoded filter
    sample = sample.filter(~pl.col("letter_with_class").is_in(FILTER_LETTER_WITH_CLASS))
    
    # define split parts for each book name of interest
    if split_parts:
        # change the book names to include part info
        def assign_split_part(row):
            book = row['book_name']
            page = row['page_number']
            if book in SPLIT_RANGES:
                for part, ranges in SPLIT_RANGES[book].items():
                    for start, end in ranges:
                        if start <= page < end:
                            return f"{book}_part{part}"
            # If no part matched, return the original book name
            # This happens when page numbers fall outside defined ranges
            return book
        
        if debug:
            print("\nDEBUG: Applying split parts...")
            print(f"SPLIT_RANGES for fortysermons1685: {SPLIT_RANGES.get('fortysermons1685', 'Not found')}")
        
        sample = sample.with_columns(
            book_name=pl.struct(pl.col("*")).map_elements(assign_split_part, return_dtype=pl.Utf8)
        )
        
        if debug:
            print("\nDEBUG: Book names after split parts:")
            print(sorted(sample["book_name"].unique().to_list()))
            
            # Check what happened to chunk files
            chunk_files = sample.filter(pl.col('filename').str.contains('chunk'))
            if len(chunk_files) > 0:
                print("\nDEBUG: Chunk files book_name distribution:")
                print(chunk_files.group_by("book_name").agg(pl.len()).sort("book_name"))
                
            # Check for records that didn't get a part assigned
            for book_base in SPLIT_RANGES.keys():
                unassigned = sample.filter(pl.col("book_name") == book_base)
                if len(unassigned) > 0:
                    print(f"\nWARNING: {len(unassigned)} records in '{book_base}' didn't match any part range:")
                    print(f"  Page numbers: {sorted(unassigned['page_number'].unique().to_list())}")
                    print(f"  Defined ranges: {SPLIT_RANGES[book_base]}")
                    print(f"  Chunk files: {unassigned.filter(pl.col('filename').str.contains('chunk')).shape[0]}")
                    print(f"  Non-chunk files: {unassigned.filter(~pl.col('filename').str.contains('chunk')).shape[0]}")
    
    return sample
