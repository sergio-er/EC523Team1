from PIL import Image
import os
from pathlib import Path
import glob

def downsample_images(input_folder, output_folder, num_images=5000, target_size=(16, 16)):
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    print(f"Looking for images in: {input_folder}")
    
    # Get all image files and sort them numerically
    image_files = []
    for ext in ('*.png', '*.jpg', '*.jpeg'):
        found_files = glob.glob(os.path.join(input_folder, ext))
        print(f"Found {len(found_files)} files with extension {ext}")
        image_files.extend(found_files)
    
    if not image_files:
        print("No image files found! Please check the input directory and file extensions.")
        return
        
    print(f"Total images found: {len(image_files)}")
    
    # Sort files numerically (assuming filenames are numbers)
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    
    # Process only the first num_images
    for i, image_path in enumerate(image_files[:num_images]):
        try:
            # Open and resize image
            with Image.open(image_path) as img:
                # Verify input size
                if img.size != (64, 64):
                    print(f"Warning: {image_path} is not 64x64. Skipping...")
                    continue
                
                # Resize image
                resized_img = img.resize(target_size, Image.Resampling.BICUBIC)
                
                # Create output filename with same extension as input
                output_filename = os.path.basename(image_path)
                output_path = os.path.join(output_folder, output_filename)
                
                # Save resized image
                resized_img.save(output_path)
                
                print(f"Processed {i+1}/{num_images}: {output_filename}")
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
    # Set your input and output folders here
    input_folder = "nina/Inference/Originals/CelebA_5k"
    output_folder = "nina/Inference/Bicubic/LR"
    
    # Debug: Check if input directory exists
    if not os.path.exists(input_folder):
        print(f"Error: Input directory '{input_folder}' does not exist!")
        exit(1)
        
    downsample_images(input_folder, output_folder) 