# Training vision model on meter digits
# This code is used to train a model to recognize digits on a meter

    # To test the code below, install the following packages
    # pip install matplotlib
        # Used to plot the images
    # pip install numpy
        # Used to work with arrays
    # pip install tensorflow
        # Used to build the model
    # pip install pillow
        # Used to work with images, resize, etc

    # To install all at once run the following command
    # pip install matplotlib numpy tensorflow pillow


# CELL 1

# Unpack tar file
import  tarfile
import os

tar = tarfile.open("./datasets/datasets.tar") # This line will open the tar file
os.makedirs("images", exist_ok=True) # This line will create a directory called images in the current directory
tar.extractall('images') # This line will extract all files in the tar file to the current directory
print("Imported and Extracted successfully")

# Expected output
# Imported and Extracted successfully

# CELL 2

import os
import shutil # Used to move files
from PIL import Image 
from tensorflow import keras
import numpy as np

imgfiles = []
for root, dirs, files in os.walk('images'):
    for file in files:
        if (file.endswith(".jpg") and not file.startswith("10_") and not file.startswith("N")):
            imgfiles.append(root + "/" + file)

y_data = np.empty((len(imgfiles)))
y_file = np.empty((len(imgfiles)), dtype="S100")
x_data = np.empty((len(imgfiles),32,20,3))

for i, aktfile in enumerate(imgfiles):
    base = os.path.basename(aktfile)

    # get label from filename (1.2_ new or 1_ old),
    if (base[1]=="."):
        target = base[0:3]
    else:
        target = base[0:1]

    category = float(target)

    test_image = Image.open(aktfile).resize((20, 32))
    test_image = np.array(test_image, dtype="float32")
    y_file[i] =  base
    x_data[i] =  test_image
    y_data[i] =  category
print("digit data count: ", len(y_data))   

# EXPECTED TERMINAL OUTPUT: 
# digit data count:  12319


# CELL 3
# Now to look at the images

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(18, 10))
columns = 10
rows = 5

# Iterate over the images
for i in range(1, columns*rows +1):
    if (i>len(x_data)):
        break
    fig.add_subplot(rows, columns, i)
    plt.title(y_data[i-1])  # set title
    plt.xticks([0.2, 0.4, 0.6, 0.8])
    plt.imshow((x_data[i-1]).astype(np.uint8), aspect='1.6', extent=[0, 1, 0, 1])
    
    # Add yellow lines to separate the digits
    for y in np.arange(0.2, 0.8, 0.2):
        plt.axhline(y=y,color='yellow')
    
    ax=plt.gca()
    ax.get_xaxis().set_visible(False) 
    plt.tight_layout()

plt.show()

# CELL 4
# Data distribution

# Calculate the count of each digit class
_, inverse = np.unique(y_data, return_inverse=True)
data_bincount = np.bincount(inverse)

# Plot the data distribution
fig = plt.figure(figsize=(40, 10))
fig.suptitle("Data distribution")
plt.bar(np.arange (0, 10, 0.1), data_bincount, width=0.09, align='center')
plt.ylabel('count')
plt.xlabel('digit class')
plt.xticks(np.arange(0, 10, 0.1))
plt.show()

print("Distribution successful")

# Expected output
# Distribution successful
# A bar chart showing the distribution of the data


# CELL 5
