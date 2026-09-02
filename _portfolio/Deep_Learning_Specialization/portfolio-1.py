# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: tensorflow
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Apple and Orange Classifier

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/Deep_Learning_Specialization/portfolio-1.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a basic Convolutional Neural Network (CNN) for apple and orange classification.

# %% [markdown]
# A [Convolutional Neural Network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network) is a type of Deep Neural Network that is specifically designed for image and video processing tasks. It uses convolutional layers to extract features from input data, and is particularly useful for tasks such as image classification, object detection, and image segmentation. CNNs are designed to take advantage of the spatial hierarchies and local feature extraction in images, and are often used in computer vision applications.

# %% [markdown]
# ## Import libraries

# %%
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from keras import Sequential, Input, layers, callbacks
import tensorflow as tf
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# %% [markdown]
# ## Download the dataset

# %% [markdown]
# Download the dataset from [Kaggle](https://www.kaggle.com/datasets/balraj98/apple2orange-dataset).

# %%
kagglehub.dataset_download("balraj98/apple2orange-dataset", output_dir="/tmp/apple2orange-dataset")


# %% [markdown]
# ## Load the dataset

# %%
def get_file_paths(path):
    file_paths = []

    for file in os.listdir(path):
        if file.endswith(".jpg"):
            file_paths.append(os.path.join(path, file))
    
    return file_paths

def load_image(file_path, image_size):
    image = tf.io.read_file(file_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    
    return image


# %%
apple_file_paths = get_file_paths("/tmp/apple2orange-dataset/trainA")[:500]
orange_file_paths = get_file_paths("/tmp/apple2orange-dataset/trainB")[:500]

X = np.array(list(map(lambda x: load_image(x, (128, 128)), apple_file_paths + orange_file_paths)))
y = np.array([0] * len(apple_file_paths) + [1] * len(orange_file_paths))

X_train, X_right, y_train, y_right = train_test_split(X, y, test_size=0.2, random_state=0)
X_test, X_validation, y_test, y_validation = train_test_split(X_right, y_right, test_size=0.5, random_state=0)

ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
ds_validation = tf.data.Dataset.from_tensor_slices((X_validation, y_validation)).batch(32)
classes = ["apple", "orange"]

# %%
print("Number of training samples:", X_train.shape[0])
print("Number of validation samples:", X_validation.shape[0])
print("Number of test samples:", X_test.shape[0])
print("Each image has a shape of", X_train.shape[1:])

# %% [markdown]
# ## Visualize the dataset

# %%
indexes = np.random.choice(range(0, X_train.shape[0]), size=16, replace=False)
samples = zip(X_train[indexes], y_train[indexes])

fig, axs = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle('Random samples')

for ax, sample in zip(axs.flatten(), samples):
    ax.imshow(sample[0] / 255)
    ax.set_title(classes[sample[1]])
    ax.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualize the class distribution

# %%
labels, sizes = np.unique(y_train, return_counts=True)

fig, ax = plt.subplots()
ax.pie(sizes, textprops={'color': "w", 'fontsize': '12'}, autopct=lambda pct: "{:.2f}%\n({:d})".format(pct, round(pct/100 * sum(sizes))))
ax.legend([f"{label} ({classes[label]})" for label in labels])
ax.set_title("Class")
plt.show()


# %% [markdown]
# ## Define a custom callback

# %%
class CustomVerbose(callbacks.Callback):
    def __init__(self, epochs_to_show):
        self.epochs_to_show = epochs_to_show

    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.epochs_to_show:
            self.epoch_start_time = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.epochs_to_show:
            self.epoch_stop_time = datetime.now()
            print(f"Epoch {epoch + 1}/{self.epochs_to_show[-1] + 1}")
            print(f"\telapsed time: {(self.epoch_stop_time - self.epoch_start_time).total_seconds():.3f}s - accuracy: {logs['binary_accuracy']:.4f} - loss: {logs['loss']:.4f} - val_accuracy: {logs['val_binary_accuracy']:.4f} - val_loss: {logs['val_loss']:.4f}")


# %% [markdown]
# ## Build a CNN

# %%
model = Sequential([Input(shape=(128, 128, 3)),
                    # Rescale
                    layers.Rescaling(scale=1 / 255),
                    # Data augmentation
                    layers.RandomFlip(mode="horizontal"),
                    layers.RandomTranslation(height_factor=0.2, width_factor=0.2, fill_mode="nearest"),
                    layers.RandomRotation(factor=0.2, fill_mode="nearest"),
                    layers.RandomZoom(height_factor=0.2, width_factor=0.2, fill_mode="nearest"),
                    # Convolutional layers
                    layers.Conv2D(16, 3, padding='same', activation='relu'),
                    layers.MaxPooling2D(),
                    layers.Conv2D(32, 3, padding='same', activation='relu'),
                    layers.MaxPooling2D(),
                    layers.Conv2D(64, 3, padding='same', activation='relu'),
                    layers.MaxPooling2D(),
                    # Deep layers
                    layers.Flatten(),
                    layers.Dense(128, activation="relu"),
                    layers.Dense(1, activation="sigmoid")])

model.summary()

# %% [markdown]
# ## Compile and train the CNN

# %%
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])

epochs = 200
patience = int(epochs / 10)
epochs_to_show = [0] + [i for i in range(patience - 1, epochs, patience)]
custom_verbose = CustomVerbose(epochs_to_show)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
history = model.fit(ds_train, epochs=epochs, verbose=0, validation_data=ds_validation, callbacks=[custom_verbose, early_stopping])

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history.history['binary_accuracy'])
ax1.plot(history.history['val_binary_accuracy'])
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.legend(["Training", "Validation"])

ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.legend(["Training", "Validation"])

plt.show()

# %% [markdown]
# ## Evaluate the CNN

# %%
indexes = np.random.choice(range(0, X_test.shape[0]), size=16, replace=False)

fig, axs = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle('Random samples')

for image, ax in zip(X_test[indexes], axs.flatten()):
    prediction_proba = model.predict(np.expand_dims(image, axis=0), verbose=0)
    ax.imshow(image / 255)
    ax.set_title("Prediction: " + classes[int(prediction_proba.squeeze().round())])
    ax.axis("off")

plt.tight_layout()
plt.show()

# %%
y_pred = model.predict(X_test, verbose=0).squeeze().round()
print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["0 (apple)", "1 (orange)"])
plt.grid(False)
plt.show()
