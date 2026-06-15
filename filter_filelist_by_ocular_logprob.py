# Input: a two column csv with filename, logprob fields
# Functionality: sorts csv by logprob, takes the top_n_pct
#   fnames of each book in the csv, shuffles them, and 
#   outputs to stdout

import sys
from collections import defaultdict
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('top_n_pct', type=float)
parser.add_argument('--separate_books', action='store_true')
parser.add_argument('--no_shuffle', action='store_true')
args = parser.parse_args()

# 0.2
top_n_pct = args.top_n_pct  # take top (100*top_n_pct)% of chars from book

#print('\n'.join([tup[0] for tup in sorted([(row.strip().split(',')[0].replace('.tif', '_aligned.tif'), float(row.strip().split(',')[1])) for row in sys.stdin], key=lambda x: -x[1])]))

csv_lines = []
for row in sys.stdin:
   fname, logprob = row.strip().split(',') 
   fname = fname.replace('.tif', '_aligned.tif')
   logprob = float(logprob)
   book_prefix = fname.split('-')[0]
   csv_lines.append((fname, book_prefix, logprob))

sorted_csv_lines = list(sorted(csv_lines, key=lambda x: -x[-1]))


sorted_fname_by_book = defaultdict(list)
for fname, book, logprob in sorted_csv_lines:
    sorted_fname_by_book[book].append(fname)

output = []
for book, fnames in sorted_fname_by_book.items():
    shortlist = fnames[: int(len(fnames) * top_n_pct)]
    output.extend(shortlist)
    if args.separate_books:
        with open(f'filelist_{book}.txt', 'w') as f:
            for fname in shortlist:
                print(fname, file=f)

if not args.no_shuffle:
    random.shuffle(output)
print('\n'.join(output))

