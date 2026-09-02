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
# # Alpaca Recognition

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/Deep_Learning_Specialization/portfolio-3.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Use transfer learning to build an alpaca classifier.

# %% [markdown]
# [Transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) is a machine learning technique where knowledge gained from one task is applied to improve performance on a related task. This method reuses a pre-trained model as the starting point for a new task, reducing the need for large amounts of labeled data and computational resources.

# %% [markdown]
# ## Import libraries

# %%
import tensorflow as tf
from datetime import datetime
from keras import applications, Sequential, layers, Input, Model, callbacks
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# %% [markdown]
# ## Download the dataset

# %% [markdown]
# Download the dataset from [Kaggle](https://www.kaggle.com/datasets/sid4sal/alpaca-dataset-small).

# %%
kagglehub.dataset_download("sid4sal/alpaca-dataset-small", output_dir="/tmp/alpaca-dataset-small")


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
alpaca_file_paths = get_file_paths("/tmp/alpaca-dataset-small/dataset/alpaca")
nonalpaca_file_paths = get_file_paths("/tmp/alpaca-dataset-small/dataset/not alpaca")

X = np.array(list(map(lambda x: load_image(x, (160, 160)), alpaca_file_paths + nonalpaca_file_paths)))
y = np.array([1] * len(alpaca_file_paths) + [0] * len(nonalpaca_file_paths))

X_train, X_right, y_train, y_right = train_test_split(X, y, test_size=0.2, random_state=0)
X_test, X_validation, y_test, y_validation = train_test_split(X_right, y_right, test_size=0.5, random_state=0)

ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
ds_validation = tf.data.Dataset.from_tensor_slices((X_validation, y_validation)).batch(32)
classes = ["non-alpaca", "alpaca"]

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
ax.legend(["1 (alpaca)" if label else "0 (non-alpaca)" for label in labels])
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
# ## Build a CNN using the MobileNetV2 architecture as a base

# %%
MobileNetV2 = applications.MobileNetV2(include_top=False, input_shape=(160, 160, 3))
MobileNetV2.trainable = False

data_augmentation = Sequential([layers.RandomFlip(mode="horizontal"),
                                layers.RandomTranslation(height_factor=0.2, width_factor=0.2, fill_mode="nearest"),
                                layers.RandomRotation(factor=0.2, fill_mode="nearest"),
                                layers.RandomZoom(height_factor=0.2, width_factor=0.2, fill_mode="nearest")])

inputs = Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = applications.mobilenet_v2.preprocess_input(x)
x = MobileNetV2(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = Model(inputs, outputs)
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

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["0 (non-alpaca)", "1 (alpaca)"])
plt.grid(False)
plt.show()
