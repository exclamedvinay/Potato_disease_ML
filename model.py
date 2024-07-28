import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt

# Define the image dimensions
img_height, img_width = 256, 256

# Get the list of images in each folder
late_blight_dir = 'C:/Users/mayan/Downloads/archive/Late Blight'
healthy_leaves_dir = 'C:/Users/mayan/Downloads/archive/Healthy'

late_blight_images = os.listdir(late_blight_dir)
healthy_leaves_images = os.listdir(healthy_leaves_dir)

# Print the number of images in each class
print(f'Number of late blight images: {len(late_blight_images)}')
print(f'Number of healthy leaves images: {len(healthy_leaves_images)}')

# Display a few images from each class
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].imshow(plt.imread(os.path.join(late_blight_dir, late_blight_images[0])))
ax[0, 1].imshow(plt.imread(os.path.join(late_blight_dir, late_blight_images[1])))
ax[1, 0].imshow(plt.imread(os.path.join(healthy_leaves_dir, healthy_leaves_images[0])))
ax[1, 1].imshow(plt.imread(os.path.join(healthy_leaves_dir, healthy_leaves_images[1])))
plt.show()

# Define the data generator for training and validation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    'C:/Users/mayan/Downloads/archive',
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    'C:/Users/mayan/Downloads/archive',
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

test_generator = datagen.flow_from_directory(
    'C:/Users/mayan/Downloads/archive',
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Define the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Evaluate the model
loss, accuracy = model.evaluate(val_generator)
print(f'Validation accuracy: {accuracy:.2f}')

# Make predictions on the test data
test_data, test_labels = next(test_generator)
predictions = model.predict(test_data)

# Convert predictions to class labels
predicted_labels = tf.argmax(predictions, axis=1)
test_labels = tf.argmax(test_labels, axis=1)

# Calculate the classification metrics
precision = precision_score(test_labels, predicted_labels, average='macro')
recall = recall_score(test_labels, predicted_labels, average='macro')
f1 = f1_score(test_labels, predicted_labels, average='macro')

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')

# Save the model
model.save('plant_disease_model.h5')
print('Model saved to plant_disease_model.h5')