from PIL import Image
import base64
import io
import pandas as pd
import os

# expected columns in the CSV file:
# root_image,page_number,filename,image,is_root,letter,letter_class,confidence_level

def encode_image_to_base64(file_path):
    """
    Opens a TIFF image, converts it to PNG in memory, 
    and returns its Base64 encoded Data URI string.
    """
    # Monkeypatch Image.open to auto-resize images to max height 144 (proportional scale)
    MAX_HEIGHT = 88
    _original_image_open = Image.open

    def _open_and_maybe_resize(fp, *args, **kwargs):
        im = _original_image_open(fp, *args, **kwargs)
        try:
            if im.height > MAX_HEIGHT:
                scale = MAX_HEIGHT / im.height
                new_w = max(1, int(im.width * scale))
                im = im.resize((new_w, MAX_HEIGHT), Image.Resampling.LANCZOS)
            return im
        except Exception:
            im.close()
            raise

    Image.open = _open_and_maybe_resize
    try:
        # Open the TIFF image with Pillow
        with Image.open(file_path) as img:
            # Create an in-memory binary stream
            with io.BytesIO() as in_mem_file:
                # Save the image to the in-memory stream as a PNG
                img.save(in_mem_file, format="PNG")
                # Get the binary content of the PNG
                image_bytes = in_mem_file.getvalue()
        
        # Encode the PNG bytes to Base64
        base64_bytes = base64.b64encode(image_bytes)
        base64_string = base64_bytes.decode('utf-8')
        
        # Prepend the Data URI scheme for PNG images
        return f"data:image/png;base64,{base64_string}"

    except FileNotFoundError:
        print(f"⚠️ WARNING: File not found at '{file_path}'. Image column will be empty for this entry.")
        return None
    except Exception as e:
        print(f"❌ ERROR: Could not process file '{file_path}'. Reason: {e}")
        return None
    

def main():
    # Load the CSV file
    csv_path = "LockeSpinozaMatches.csv"
    images_dir = "workbench_images"
    
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Total rows: {len(df)}")
    
    # Count rows that need image encoding
    rows_needing_images = df['image'].isna() | (df['image'] == '')
    print(f"Rows with missing images: {rows_needing_images.sum()}")
    
    # Count rows that need filename filled
    rows_needing_filename = df['filename'].isna() | (df['filename'] == '')
    print(f"Rows with missing filename: {rows_needing_filename.sum()}")
    
    # Process each row
    images_encoded = 0
    filenames_filled = 0
    
    for idx, row in df.iterrows():
        # Fill filename from root_image if empty
        if pd.isna(row['filename']) or row['filename'] == '':
            df.at[idx, 'filename'] = row['root_image']
            filenames_filled += 1
            if filenames_filled <= 5:  # Show first 5
                print(f"  Filled filename for row {idx}: {row['root_image']}")
        
        # Encode image if missing
        if pd.isna(row['image']) or row['image'] == '':
            # Try to find the image in workbench_images directory
            # The root_image might have .tif extension, but workbench files are .jpg
            root_image = row['root_image']
            
            # Try different possible filenames
            possible_names = [
                root_image,
                root_image.replace('.tif', '.jpg'),
                root_image.replace('.tif', '.png'),
                os.path.splitext(root_image)[0] + '.jpg',
                os.path.splitext(root_image)[0] + '.png',
            ]
            
            image_found = False
            for possible_name in possible_names:
                image_path = os.path.join(images_dir, possible_name)
                if os.path.exists(image_path):
                    if images_encoded < 5:  # Show first 5
                        print(f"  Encoding image for row {idx}: {possible_name}")
                    
                    encoded = encode_image_to_base64(image_path)
                    if encoded:
                        df.at[idx, 'image'] = encoded
                        images_encoded += 1
                        image_found = True
                        break
            
            if not image_found and images_encoded < 10:  # Show first 10 failures
                print(f"  ⚠️  Could not find image file for: {root_image}")
    
    print(f"\nSummary:")
    print(f"  - Filled {filenames_filled} filename fields")
    print(f"  - Encoded {images_encoded} images")
    
    # Save the updated CSV
    output_path = "LockeSpinozaMatches_processed.csv"
    print(f"\nSaving to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
