# To train the model on the dataset, run the following code:

# CELL 3

import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

num_samples = 100  # Replace with the actual number of samples

x_data = np.zeros((num_samples, 32, 20, 3))
y_data = np.zeros((num_samples,))
y_file = np.zeros((num_samples,))

test_image = # Replace with the actual image data
category = # Replace with the actual category
base = # Replace with the actual base value

for i in range(num_samples):
    x_data[i] = np.array(test_image)
    y_data[i] = category
    y_file[i] = base

# Split the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 20, 3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# Save the model
model.save('meter_digits_model')

# Load the model
new_model = keras.models.load_model('meter_digits_model')

# Make predictions
probability_model = tf.keras.Sequential([new_model,
                                        tf.keras.layers.Softmax()])

predictions = probability_model.predict(x_test)

# Plot the first 10 test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
fig = plt.figure(figsize=(20, 20))
for i in range(10):
    fig.add_subplot(5, 5, i+1)
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    predicted_label = np.argmax(predictions[i])
    true_label = y_test[i]
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(predicted_label, true_label), color=color)

plt.show()

# Expected output
# A plot showing the first 10 test images, their predicted label, and the true label

# CELL 13

# Save the model as a .h5 file
new_model.save('meter_digits_model.h5')

# Expected output
# INFO:tensorflow:Assets written to: meter_digits_model.h5/assets
