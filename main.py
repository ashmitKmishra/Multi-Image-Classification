# This code hasbeen coded in Lnux Operating Software.
import os
import tensorflow as tf
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

home_directory = os.path.expanduser("~")

# Define the other components of the path
downloads_directory = "Downloads"
data_directory = "data"
data_dir = os.path.join(home_directory, downloads_directory, data_directory)

image_exts = ['jpeg', 'jpg', 'bmp', 'png']
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list{}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('IMage with image {}'.format(image_path))
# =============================================================================
# img = cv2.imread(os.path.join(data_dir, 'cats', 'images5.jpg'))
# print(img.shape)
# plt.imshow(img)
# plt.show()
# =============================================================================
data = tf.keras.utils.image_dataset_from_directory(data_dir)
data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()
# print(batch[1])
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])


# =============================================================================
    
# 0 = cat
# 1 = dog
# 2 = flower
# 3 = tree
# =============================================================================
# 2). Preprocessing Data
data = data.map(lambda x, y: (x/255, y))
data.as_numpy_iterator().next()
# print(len(data))

train_size = int(0.8 * len(data))
val_size = int(0.1 * len(data))
test_size = int(0.1 * len(data))

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)
# =============================================================================
# 3). Building Network


# Define the CNN model
model = models.Sequential()

# Convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256,256, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten layer
model.add(layers.Flatten())

# Fully connected layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))

# Output layer for multi-class classification
model.add(layers.Dense(4, activation='softmax'))  # 4 classes: cats, dogs, flowers, trees

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use sparse categorical crossentropy for integer-encoded labels
              metrics=['accuracy'])

# ...
model.summary()

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
# hist = model.fit(train, epochs=30, validation_data=val, callbacks=[tensorboard_callback])


# Load an image (replace 'your_image_path.jpg' with the actual path)
image_path = '/home/workstation1/Downloads/data/flowers/AUGUSTpink50x70.jpg'
img = cv2.imread(image_path)
img = cv2.resize(img, (256, 256))  # Resize the image to match the input shape
img = img / 255.0  # Normalize the pixel values
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Predict the class probabilities
predictions = model.predict(img)

# Display the predicted probabilities for each class
print(f'Predicted probabilities for each class: {predictions}')

# Get the predicted class
predicted_class = np.argmax(predictions)

# Define class names
class_names = sorted(os.listdir(data_dir))

print("Class names:", class_names)
# Print the predicted class
print(f'Predicted class: {class_names[predicted_class]}')








