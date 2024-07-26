# This script is used to process the dataset and split it into training, validation, and test sets.
    # The dataset is a tar file containing images of digits from 0 to 9. The images are named as follows:
    # 0_1.jpg, 0_2.jpg, ..., 0_9.jpg, 1_1.jpg, 1_2.jpg, ..., 1_9.jpg, ..., 9_1.jpg, 9_2.jpg, ..., 9_9.jpg
    # The first digit in the filename is the label of the image. The images are stored in a folder named "images".
# The script will unpack the tar file, process the images and labels, and split the dataset into training, validation, and test sets.
# The labels will be saved to JSON files.


import os
import json
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# Unpack tar file
import tarfile
tar = tarfile.open("./datasets/datasets.tar")
os.makedirs("images", exist_ok=True)
tar.extractall('images')
tar.close()

# Process images and labels
imgfiles = []
labels = []
for root, dirs, files in os.walk('images'):
    for file in files:
        if file.endswith(".jpg") and not file.startswith("10_") and not file.startswith("N"):
            imgfiles.append(os.path.join(root, file))
            base = os.path.basename(file)
            if base[1] == ".":
                target = base[0:3]  # e.g., 1.2
            else:
                target = base[0:1]  # e.g., 1
            labels.append({"image": os.path.join(root, file), "label": target})

# Split dataset
train_val_files, test_files, train_val_labels, test_labels = train_test_split(
    imgfiles, labels, test_size=0.1, random_state=42)

train_files, val_files, train_labels, val_labels = train_test_split(
    train_val_files, train_val_labels, test_size=0.1, random_state=42)

# Save labels to JSON files
with open('train_labels.json', 'w') as f:
    json.dump(train_labels, f)

with open('val_labels.json', 'w') as f:
    json.dump(val_labels, f)

with open('test_labels.json', 'w') as f:
    json.dump(test_labels, f)
