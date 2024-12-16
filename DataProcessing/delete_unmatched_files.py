from pathlib import Path

def delete_unmatched_files(source_folder, reference_folder):
    """
    Delete files from source_folder whose names don't appear in reference_folder.
    """
    source_path = Path(source_folder)
    reference_path = Path(reference_folder)
    
    reference_names = {f.stem for f in reference_path.glob('*')}
    deleted_count = 0
    
    for file_path in source_path.glob('*'):
        if file_path.stem not in reference_names:
            file_path.unlink()
            deleted_count += 1
    
    print(f"Deleted {deleted_count} files")
    print(f"Remaining files: {len(list(source_path.glob('*')))}")

if __name__ == "__main__":
    source_folder = "/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/nina/ResShift/data/inference/originals"     # Replace with your source folder path
    reference_folder = "/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/nina/Inference/Originals/CelebA_5k"  # Replace with your reference folder path
    delete_unmatched_files(source_folder, reference_folder) 