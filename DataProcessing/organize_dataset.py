from pathlib import Path
import shutil
import os
import sys

def get_image_files(directory, extensions=('*.png', '*.jpg', '*.jpeg')):
    """Get all image files in a directory"""
    files = []
    for ext in extensions:
        files.extend(Path(directory).glob(ext))
    return set(file.name for file in files)

def organize_dataset():
    # Define paths
    training_path = Path("nina/CycleGAN/datasets/H2L_50k/trainB")
    validation_path = Path("nina/CycleGAN/datasets/H2L_50k/testB")
    total_path = Path("Bulat_datasets/LR")
    output_path = Path("nina/Inference/Originals/testB")

    # Check if directories exist
    for path, name in [(training_path, "Training"), (validation_path, "Validation"), (total_path, "Total")]:
        if not path.exists():
            print(f"Error: {name} directory not found at {path}")
            sys.exit(1)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Get filenames from each directory
    print("Reading directories...")
    training_files = get_image_files(training_path)
    validation_files = get_image_files(validation_path)
    total_files = get_image_files(total_path)

    # Check for empty directories
    if len(training_files) == 0:
        print(f"Error: No image files found in training directory: {training_path}")
        sys.exit(1)
    if len(validation_files) == 0:
        print(f"Error: No image files found in validation directory: {validation_path}")
        sys.exit(1)
    if len(total_files) == 0:
        print(f"Error: No image files found in total directory: {total_path}")
        sys.exit(1)

    print(f"Found {len(training_files)} training files")
    print(f"Found {len(validation_files)} validation files")
    print(f"Found {len(total_files)} total files")

    # Find files that are in total but not in training or validation
    testing_files = total_files - (training_files | validation_files)
    print(f"Found {len(testing_files)} testing files")

    if len(testing_files) == 0:
        print("Warning: No files found for testing set (all files are in training or validation)")
        sys.exit(1)

    # Copy files to testing directory
    print("Copying files to testing directory...")
    for filename in testing_files:
        source = total_path / filename
        destination = output_path / filename
        shutil.copy2(source, destination)
        
    print(f"Successfully copied {len(testing_files)} files to {output_path}")

    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Training images: {len(training_files)}")
    print(f"Validation images: {len(validation_files)}")
    print(f"Testing images: {len(testing_files)}")
    print(f"Total images: {len(total_files)}")
    
    # Verify no overlap
    overlap_train_val = len(training_files & validation_files)
    overlap_train_test = len(training_files & testing_files)
    overlap_val_test = len(validation_files & testing_files)
    
    print("\nOverlap Check:")
    print(f"Training-Validation overlap: {overlap_train_val}")
    print(f"Training-Testing overlap: {overlap_train_test}")
    print(f"Validation-Testing overlap: {overlap_val_test}")

if __name__ == "__main__":
    organize_dataset() 