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
#     display_name: portfolio
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Finger Signs

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/Deep_Learning_Specialization/portfolio-2.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a Convolutional Neural Network (CNN) and a Residual Neural Network (ResNet) to classify a collection of six finger signs representing numbers from 0 to 5.

# %% [markdown]
# A [Residual Network (ResNet)](https://en.wikipedia.org/wiki/Residual_neural_network) is a deep learning architecture that addresses the problem of training very deep neural networks. The key innovation of ResNet is the introduction of residual connections, also known as skip connections, which allow the network to learn residual functions with reference to the layer inputs. This enables the network to bypass the vanishing gradient problem, where gradients become smaller as they propagate through deep layers, making it difficult to train very deep networks.

# %% [markdown]
# ## Import libraries

# %%
import numpy as np
import h5py
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from keras import Sequential, Input, layers, applications, Model, optimizers, callbacks
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# %% [markdown]
# ## Download the dataset

# %% language="bash"
#
# wget -nc --progress=bar:force:noscroll https://raw.githubusercontent.com/LuisAngelMendozaVelasco/Deep_Learning_Specialization/main/Convolutional_Neural_Networks/Week1/Labs/datasets/test_signs.h5 -P /tmp
# wget -nc --progress=bar:force:noscroll https://raw.githubusercontent.com/LuisAngelMendozaVelasco/Deep_Learning_Specialization/main/Convolutional_Neural_Networks/Week1/Labs/datasets/train_signs.h5.gz -P /tmp
# gunzip -kf /tmp/train_signs.h5.gz

# %% [markdown]
# ## Load the dataset

# %%
dataset = h5py.File('/tmp/train_signs.h5', "r")
X = dataset["train_set_x"][:]
y = dataset["train_set_y"][:]
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.1, random_state=0)

dataset_test = h5py.File('/tmp/test_signs.h5', "r")
X_test = dataset_test["test_set_x"][:]
y_test = dataset_test["test_set_y"][:]

ds_train = tf.data.Dataset.from_tensor_slices((X_train, tf.one_hot(y_train, 6))).batch(32)
ds_validation = tf.data.Dataset.from_tensor_slices((X_validation, tf.one_hot(y_validation, 6))).batch(32)
classes = dataset_test["list_classes"][:].astype(str)

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
    ax.imshow(sample[0])
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
ax.legend(labels)
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
            print(f"\telapsed time: {(self.epoch_stop_time - self.epoch_start_time).total_seconds():.3f}s - accuracy: {logs['categorical_accuracy']:.4f} - loss: {logs['loss']:.4f} - val_accuracy: {logs['val_categorical_accuracy']:.4f} - val_loss: {logs['val_loss']:.4f}")


# %% [markdown]
# ## Build a CNN

# %%
model_CNN = Sequential([Input(shape=(64, 64, 3)),
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
                        layers.Dense(6, activation="softmax")])

model_CNN.summary()

# %% [markdown]
# ## Compile and train the CNN

# %%
model_CNN.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['categorical_accuracy'])

epochs = 500
patience = int(epochs / 10)
epochs_to_show = [0] + [i for i in range(patience - 1, epochs, patience)]
custom_verbose = CustomVerbose(epochs_to_show)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
history_CNN = model_CNN.fit(ds_train, epochs=epochs, verbose=0, validation_data=ds_validation, callbacks=[custom_verbose, early_stopping])

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history_CNN.history['categorical_accuracy'])
ax1.plot(history_CNN.history['val_categorical_accuracy'])
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.legend(["Training", "Validation"])

ax2.plot(history_CNN.history['loss'])
ax2.plot(history_CNN.history['val_loss'])
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.legend(["Training", "Validation"])

plt.show()

# %% [markdown]
# ## Evaluate the model

# %%
indexes = np.random.choice(range(0, X_test.shape[0]), size=16, replace=False)

fig, axs = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle('Random samples')

for image, ax in zip(X_test[indexes], axs.flatten()):
    prediction_proba = model_CNN.predict(np.expand_dims(image, axis=0), verbose=0)
    ax.imshow(image)
    ax.set_title("Prediction: " + classes[np.argmax(prediction_proba.squeeze())])
    ax.axis("off")

plt.tight_layout()
plt.show()

# %%
y_pred = np.argmax(model_CNN.predict(X_test, verbose=0).squeeze(), axis=1)
print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.grid(False)
plt.show()

# %% [markdown]
# ## Build a ResNet using the ResNET50V2 architecture as a base

# %%
ResNet50V2 = applications.ResNet50V2(include_top=False, input_shape=(64, 64, 3))
data_augmentation = Sequential([layers.RandomFlip(mode="horizontal"),
                                layers.RandomTranslation(height_factor=0.2, width_factor=0.2, fill_mode="nearest"),
                                layers.RandomRotation(factor=0.2, fill_mode="nearest"),
                                layers.RandomZoom(height_factor=0.2, width_factor=0.2, fill_mode="nearest")])

inputs = Input(shape=(64, 64, 3))
x = data_augmentation(inputs)
x = applications.resnet_v2.preprocess_input(x)
x = ResNet50V2(x)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(6, activation="softmax")(x)

model_ResNet = Model(inputs, outputs)
model_ResNet.summary()

# %% [markdown]
# ## Compile and train the ResNet

# %%
model_ResNet.compile(optimizer=optimizers.Adam(5e-5), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

epochs = 100
patience = int(epochs / 10)
epochs_to_show = [0] + [i for i in range(patience - 1, epochs, patience)]
custom_verbose = CustomVerbose(epochs_to_show)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
history_ResNet = model_ResNet.fit(ds_train, epochs=epochs, verbose=0, validation_data=ds_validation, callbacks=[custom_verbose, early_stopping])

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history_ResNet.history['categorical_accuracy'])
ax1.plot(history_ResNet.history['val_categorical_accuracy'])
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.legend(["Training", "Validation"])

ax2.plot(history_ResNet.history['loss'])
ax2.plot(history_ResNet.history['val_loss'])
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.legend(["Training", "Validation"])

plt.show()

# %% [markdown]
# ## Evaluate the model

# %%
indexes = np.random.choice(range(0, X_test.shape[0]), size=16, replace=False)

fig, axs = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle('Random samples')

for image, ax in zip(X_test[indexes], axs.flatten()):
    prediction_proba = model_ResNet.predict(np.expand_dims(image, axis=0), verbose=0)
    ax.imshow(image)
    ax.set_title("Prediction: " + classes[np.argmax(prediction_proba.squeeze())])
    ax.axis("off")

plt.tight_layout()
plt.show()

# %%
y_pred = np.argmax(model_ResNet.predict(X_test, verbose=0).squeeze(), axis=1)
print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.grid(False)
plt.show()
