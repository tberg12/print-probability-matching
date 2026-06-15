import sys
import argparse
import csv
import math
import re

#!/usr/bin/env python3

def parse_args():
    parser = argparse.ArgumentParser(description="Filter CSV rows by float values in second column")
    # parser.add_argument('input_file', type=argparse.FileType('r'), help="Input CSV file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--top', type=int, help="Output the top N rows overall")
    group.add_argument('--percent', type=float, help="Output the top N% rows overall (0 < N <= 100)")
    group.add_argument('--percent_by_char', type=float, help="Output the top N% rows for each char class (0 < N <= 100)")
    return parser.parse_args()

def main():
    args = parse_args()
    # reader = csv.reader(args.input_file)
    reader = csv.reader(sys.stdin)
    rows = []
    
    # Read all rows and parse the float from second column
    for row in reader:
        if len(row) < 2:
            continue  # skip rows with insufficient columns
        try:
            value = float(row[1])
        except ValueError:
            continue  # skip rows where conversion fails
        rows.append((row, value))
    
    writer = csv.writer(sys.stdout)
    
    if args.percent_by_char is not None:
        if args.percent_by_char <= 0 or args.percent_by_char > 100:
            sys.exit("Error: percent_by_char must be in the range (0, 100].")
        # Group rows by char class extracted from first column using regex:
        pattern = re.compile(r'([A-Za-z])_uc')
        groups = {}
        for row, value in rows:
            match = pattern.search(row[0])
            if not match:
                continue  # skip rows where extraction fails
            char_class = match.group(1)
            groups.setdefault(char_class, []).append((row, value))
        
        # For each group, sort by float value in descending order and output the top percent
        for char_class, group_rows in groups.items():
            group_rows.sort(key=lambda x: x[1], reverse=True)
            count = math.ceil(len(group_rows) * args.percent_by_char / 100)
            for row, _ in group_rows[:count]:
                writer.writerow(row)
    else:
        # Sort rows in descending order by the float value in the second column
        rows.sort(key=lambda x: x[1], reverse=True)
        
        # Determine how many rows to output
        if args.top is not None:
            count = args.top
        else:  # percentage option
            if args.percent <= 0 or args.percent > 100:
                sys.exit("Error: percent must be in the range (0, 100].")
            total = len(rows)
            count = math.ceil(total * args.percent / 100)
        
        # Output the selected rows as CSV to stdout
        for row, _ in rows[:count]:
            writer.writerow(row)

if __name__ == '__main__':
    main()