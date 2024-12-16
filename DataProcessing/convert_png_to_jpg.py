import os
from pathlib import Path
from PIL import Image


def convert_png_to_jpg(input_dir, output_dir, quality=95, limit=None):
    """
    Convert .png images in a directory to .jpg format, sorted by filename, and save them to another directory.
    
    Args:
        input_dir (str): Path to the input directory containing .png images.
        output_dir (str): Path to the output directory where .jpg images will be saved.
        quality (int): Quality of the saved .jpg images (1-100).
        limit (int): Maximum number of files to convert (default: None, converts all).
    """
    # Ensure input directory exists
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Create the output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all .png files, sorted by filename
    png_files = sorted(input_path.glob("*.png"))

    # Apply the limit, if specified
    if limit is not None:
        png_files = png_files[:limit]

    # Process each .png file
    for file in png_files:
        try:
            # Load the PNG image
            img = Image.open(file)

            # Ensure it's in RGB mode (JPG doesn't support alpha channel)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Generate output filename
            output_file = output_path / f"{file.stem}.jpg"

            # Save the image as JPG
            img.save(output_file, "JPEG", quality=quality)
            print(f"Converted: {file.name} -> {output_file.name}")
        except Exception as e:
            print(f"Failed to convert {file.name}: {e}")


if __name__ == "__main__":
    import argparse

    # Argument parser for command-line usage
    parser = argparse.ArgumentParser(description="Convert PNG images to JPG.")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Path to the input directory containing .png images"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to the output directory for .jpg images"
    )
    parser.add_argument(
        "--quality", type=int, default=95, help="Quality of the saved .jpg images (1-100)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum number of files to convert (default: all)"
    )
    args = parser.parse_args()

    # Convert PNG to JPG
    convert_png_to_jpg(args.input_dir, args.output_dir, args.quality, args.limit)
