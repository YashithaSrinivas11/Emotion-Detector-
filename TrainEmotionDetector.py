import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam  # Updated optimizer import
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1.0 / 255)
validation_data_gen = ImageDataGenerator(rescale=1.0 / 255)

# Preprocess all train images
train_generator = train_data_gen.flow_from_directory(
    r'C:\Users\Admin\Downloads\archive\train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
)

# Preprocess all validation images
validation_generator = validation_data_gen.flow_from_directory(
    r'C:\Users\Admin\Downloads\archive\test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
)

# Create model structure
emotion_model = Sequential()

# Convolutional and pooling layers
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

# Flatten and dense layers
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# Disable OpenCL in OpenCV
cv2.ocl.setUseOpenCL(False)

# Compile the model
# Updated Adam optimizer without deprecated arguments
emotion_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),  # Use `learning_rate` instead of `lr`
    metrics=['accuracy'],
)

# Train the model
emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=28709 // 64,  # Ensure the number matches your dataset
    epochs=20,
    validation_data=validation_generator,
    validation_steps=7178 // 64,
)

# Save model structure in a JSON file
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# Save trained model weights in an H5 file
emotion_model.save_weights('emotion_model.weights.h5')

