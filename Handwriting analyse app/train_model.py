import pickle
import numpy as np
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define constants
IMAGE_SIZE = (64, 64)  # Resize images to 64x64 pixels
BATCH_SIZE = 32
train_data_dir = r'Downloads\archive\dataset\training_set'
test_data_dir = r'Downloads\archive\dataset\test_set'

# Ensure models directory exists
os.makedirs('trained_model', exist_ok=True)


# Function to extract image data and labels
def load_data(data_dir):
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    data = []
    labels = []
    for _ in range(len(generator)):
        batch_data, batch_labels = next(generator)
        flattened_data = batch_data.reshape(batch_data.shape[0], -1)
        data.append(flattened_data)
        labels.append(batch_labels)

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    return data, labels


# Train an SVM model for behavior
def train_behavior_model():
    print("Training behavior model...")
    X_train, y_train = load_data(train_data_dir)
    X_test, y_test = load_data(test_data_dir)

    # Create and train an SVM model
    model = SVC(kernel='linear', C=1.0, random_state=42)
    model.fit(X_train, np.argmax(y_train, axis=1))

    # Calculate and print accuracy
    accuracy = model.score(X_test, np.argmax(y_test, axis=1))
    print(f"Behavior model accuracy: {accuracy * 100:.2f}%")

    # Save the model
    with open('trained_model/model_behavior', 'wb') as f:
        pickle.dump(model, f)
    print("Behavior model saved as model_behavior")


# Train an SVM model for personality
def train_personality_model():
    print("Training personality model...")
    X_train, y_train = load_data(train_data_dir)
    X_test, y_test = load_data(test_data_dir)

    # Create and train an SVM model
    model = SVC(kernel='linear', C=1.0, random_state=42)
    model.fit(X_train, np.argmax(y_train, axis=1))

    # Calculate and print accuracy
    accuracy = model.score(X_test, np.argmax(y_test, axis=1))
    print(f"Personality model accuracy: {accuracy * 100:.2f}%")

    # Save the model
    with open('trained_model/model_personality', 'wb') as f:
        pickle.dump(model, f)
    print("Personality model saved as model_personality")


# Main script to train both models
if __name__ == "__main__":
    train_behavior_model()
    train_personality_model()