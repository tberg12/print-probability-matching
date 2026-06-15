""" Splits /graft2/code/nvog/git/matching/damage_classifier_data/3_class_data.csv 
    into train and test sets.
    The split is done by stratifying the data based on the labels.
    The split is done in a 90/10 ratio.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from PIL import Image


def split_data(input_file, output_dir, test_size=0.1):
    # Read the data
    df = pd.read_csv(input_file)

    # try to load each path with pil, delete the row if it fails
    for i, row in df.iterrows():
        try:
            img = Image.open(row['path'])
            img.close()
        except Exception as e:
            print(f"Error loading image {row['path']}: {e}")
            df.drop(i, inplace=True)

    # Split the data into train and test sets
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=42)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the train and test sets to csv files
    train_df.to_csv(os.path.join(output_dir, '3_class_data_train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, '3_class_data_test.csv'), index=False)

if __name__ == "__main__":
    # Define the input file and output directory
    input_file = '/graft2/code/nvog/git/matching/damage_classifier_data/3_class_data.csv'
    output_dir = '/graft2/code/nvog/git/matching/damage_classifier_data'

    # Split the data
    split_data(input_file, output_dir)
