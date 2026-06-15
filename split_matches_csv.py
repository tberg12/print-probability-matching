"""
Loads `matches.csv' from args.root_path and creates a directory for args.char 
with valid/test splits from the matches.csv
"""
import os
from shutil import copy2
import sys
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('char', type=str)
parser.add_argument('--root_path', default='/home/kishore/data/anomaly-detection/gold-test-sets')
args = parser.parse_args()

lines = [line.strip() for line in open(os.path.join(args.root_path, args.char, 'all', 'matches.csv')) if not line.startswith('#')]
random.shuffle(lines)

val_path = os.path.join(args.root_path, args.char, 'valid', 'matches.csv')
test_path = os.path.join(args.root_path, args.char, 'test', 'matches.csv')
char_all = os.path.join(args.root_path, args.char, 'all')

with open(val_path, 'w') as val, open(test_path, 'w') as test:
    for i in range(len(lines)):
        if i % 2 == 0:
            print(lines[i], file=test)
            for filename in lines[i].split(','):
                try:
                    copy2(os.path.join(char_all, filename), os.path.join(args.root_path, args.char, 'test'))
                except FileNotFoundError as e:
                    pass
        else:
            print(lines[i], file=val)
            for filename in lines[i].split(','):
                try:
                    copy2(os.path.join(char_all, filename), os.path.join(args.root_path, args.char, 'valid'))
                except FileNotFoundError as e:
                    pass

