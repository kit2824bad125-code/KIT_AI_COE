# PROJECT 2: DOG-CAT IMAGE CLASSIFICATION USING CNN
# Google Colab continuous code | Exactly 3 Conv2D layers

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from google.colab import files

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Download and extract public dataset
dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
zip_path = tf.keras.utils.get_file(
    "cats_and_dogs_filtered.zip", origin=dataset_url, extract=True
)
data_dir = pathlib.Path(zip_path).parent / "cats_and_dogs_filtered"
train_dir = data_dir / "train"
validation_dir = data_dir / "validation"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", shuffle=True, seed=SEED
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    validation_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", shuffle=False
)
class_names = train_ds.class_names
print("Class order:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Display sample images
plt.figure(figsize=(9, 7))
for images, labels in train_ds.take(1):
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[int(labels[i].numpy().ravel()[0])])
        plt.axis("off")
plt.tight_layout()
plt.show()

# CNN with exactly three convolutional layers
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1.0 / 255),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# Train
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
history = model.fit(
    train_ds, validation_data=val_ds, epochs=20,
    callbacks=[early_stopping], verbose=1
)

# Evaluate
val_loss, val_accuracy = model.evaluate(val_ds, verbose=1)
print(f"Validation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy * 100:.2f}%")

# Plot accuracy and loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Classification report and confusion matrix
y_true = np.concatenate([y.numpy().ravel() for _, y in val_ds]).astype(int)
y_prob = model.predict(val_ds, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print(classification_report(
    y_true, y_pred, labels=[0, 1],
    target_names=class_names, zero_division=0
))
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
print("Confusion matrix:\n", cm)

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()
ticks = np.arange(len(class_names))
plt.xticks(ticks, class_names, rotation=30)
plt.yticks(ticks, class_names)
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
threshold = cm.max() / 2 if cm.size else 0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > threshold else "black")
plt.tight_layout()
plt.show()

# Save model
model_path = "/content/cnn_cats_dogs.keras"
model.save(model_path)
print("Saved model:", model_path)

# Upload an image and predict its class
print("Upload a cat or dog image to test the model.")
uploaded = files.upload()
for filename in uploaded.keys():
    try:
        img = tf.keras.utils.load_img(filename, target_size=IMG_SIZE)
        arr = tf.keras.utils.img_to_array(img)
        prob = float(model.predict(np.expand_dims(arr, axis=0), verbose=0)[0][0])
        index = int(prob >= 0.5)
        confidence = prob if index == 1 else 1.0 - prob
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.title(f"Prediction: {class_names[index]}\nConfidence: {confidence * 100:.2f}%")
        plt.axis("off")
        plt.show()
        print("Predicted class:", class_names[index])
        print(f"Confidence: {confidence * 100:.2f}%")
    except Exception as error:
        print(f"Could not process {filename}: {error}")

# Download the trained model
files.download(model_path)
print("Project completed!")
