import os
import pickle

# Paths to your ground truth and super-resolution image directories
gt_dir = '/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/nina/Inference/Originals/CelebA_50k'
sr_dir = '/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/nina/Inference/Baseline_ResShift/HR_jpg'

# Ensure paths are valid
if not os.path.exists(gt_dir) or not os.path.exists(sr_dir):
    raise FileNotFoundError("Ensure both 'gt_dir' and 'sr_dir' directories exist.")

# Get lists of filenames (without paths) for ground truth and SR directories
gt_files = set(f for f in os.listdir(gt_dir) if os.path.isfile(os.path.join(gt_dir, f)))
sr_files = set(f for f in os.listdir(sr_dir) if os.path.isfile(os.path.join(sr_dir, f)))

# Find common files by name in both directories
common_files = list(gt_files & sr_files)
common_files = common_files[:5000]

if not common_files:
    raise ValueError("No matching files found between the ground truth and super-resolution directories.")

# Organize into a single mask type
mask_split = {
    "common_images": common_files
}

# Create output directory for mask_split.pkl
info_dir = os.path.join(os.path.dirname(gt_dir), 'infos')
os.makedirs(info_dir, exist_ok=True)

# Save mask_split.pkl
output_path = os.path.join(info_dir, 'mask_split.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(mask_split, f)

print(f"mask_split.pkl has been created at: {output_path}")
print(f"Number of common files: {len(common_files)}")
