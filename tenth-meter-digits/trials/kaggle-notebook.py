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
        target = base[0:3] # 1.2
    else:
        target = base[0:1] # 1

    category = float(target) # this means that the category is the first digit of the filename

    test_image = Image.open(aktfile).resize((20, 32)) # Resize the image to 20x32
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
        plt.axhline(y=y,color='yellow') # Add yellow lines to separate the digits
    
    ax=plt.gca() # Get the current axes
    ax.get_xaxis().set_visible(False) # Hide the x-axis
    plt.tight_layout() # Adjust the subplots to fit into the figure area

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
# Create a model to train the data
from tensorflow import keras
from keras import Sequential
from keras.layers import *
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPool2D, Dropout, Flatten, Dense
import tensorflow as tf

model = Sequential() # Create a sequential model
model.add(BatchNormalization(input_shape=(32,20,3)))
model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(100, activation = None))

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
              optimizer="adam", metrics = ["accuracy"])

y = tf.keras.utils.to_categorical(y_data, num_classes=100, dtype='float32')



# CELL 6
# Train the model


history = model.fit(x_data, y,
                validation_split=0.2, 
                batch_size=32, 
                epochs = 40,
                verbose=1)
