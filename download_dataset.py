import kagglehub
import os
import shutil

# Download the latest version of PlantVillage dataset
path = kagglehub.dataset_download("emmarex/plantdisease")

print(f"Dataset downloaded to: {path}")

# The downloaded path contains folders like 'PlantVillage' etc.
# Let's copy the images into our data/raw/plantvillage/ folder
import os
import shutil

dest = "data/raw/plantvillage"
os.makedirs(dest, exist_ok=True)

# The exact structure may vary; list contents to see
print("Contents of downloaded dataset:")
for root, dirs, files in os.walk(path):
    print(root)

# Typically, the dataset has subfolders like 'train', 'valid', 'test'
# We'll copy everything into data/raw/plantvillage/
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(dest, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print("Dataset copied to data/raw/plantvillage/")
