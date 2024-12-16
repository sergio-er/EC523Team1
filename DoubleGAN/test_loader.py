from dataset import get_loader

# Path to your dataset
dataset_path = "/projectnb/ec523kb/projects/teams_Fall_2024/Team_1/Sergio/BulatDoubleGan/unpaired_face_sr/Dataset/Dataset/HIGH/wider_lnew"

# Create a DataLoader
test_loader = get_loader(dataset_path, batch_size=1)

# Test the DataLoader
for idx, data_dict in enumerate(test_loader):
    print(f"Processing image: {data_dict['imgpath']}")
    print(f"Image 16x16 shape: {data_dict['img16'].shape}")
    print(f"Image 64x64 shape: {data_dict['img64'].shape}")
    if idx >= 2:  # Limit output for testing
        break
