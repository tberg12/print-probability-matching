import csv
import tarfile
import os
import sys

tars = sys.argv[1]
csvs = sys.argv[2]
char = sys.argv[3]
outdir = sys.argv[4]
aligncsv = os.path.join(outdir, char + "_align.csv")
try:
    os.mkdir(outdir)
except:
    print("The output directory already exists")
    pass
outcsvfile = open(aligncsv, mode="w")
for infile in os.listdir(tars):
    if infile.endswith("_uc.tar"):
        print(infile)
        prefix = infile[:-4]
        intar = os.path.join(tars, prefix + ".tar")
        bincsv = os.path.join(csvs, prefix + ".csv")
        if not (os.path.isfile(bincsv)):
            print("Not untarring: {}".format(intar))
            continue
        t = tarfile.open(intar, "r")
        t.extractall(
            outdir,
            members=[t.getmember(e) for e in t.getnames() if char + "_uc.tif" in e],
        )

        with open(bincsv, newline="") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")
            csvwriter = csv.writer(
                outcsvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            for row in csvreader:
                infile = row[3]
                outfile = row[3][:-4] + "_aligned.csv"
                binthresh = row[5]
                if char + "_uc.tif" in infile:
                    csvwriter.writerow([infile, outfile, binthresh])
outcsvfile.close()
