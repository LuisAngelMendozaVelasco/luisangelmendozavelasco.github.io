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
# # MNIST database

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/IBM_AI_Engineering/portfolio-13.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png"/>Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a basic Convolutional Neural Network (CNN) to classify handwritten digits.

# %% [markdown]
# The [MNIST database](https://en.wikipedia.org/wiki/MNIST_database) is a large dataset of handwritten digits that is commonly used for training image processing systems. It is also widely used for training machine learning models.
#
# It contains 60,000 training images and 10,000 testing images of digits written by high school students and employees of the United States Census Bureau.

# %% [markdown]
# ## Import libraries

# %%
import tensorflow as tf
from keras import Sequential, Input, layers, datasets, utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# %% [markdown]
# ## Load the dataset

# %%
(X_train, y_train), (X_right, y_right) = datasets.mnist.load_data()
X_validation, X_test, y_validation, y_test = train_test_split(X_right, y_right, train_size=0.5, random_state=0)

ds_train = tf.data.Dataset.from_tensor_slices((X_train[..., np.newaxis], utils.to_categorical(y_train))).batch(32)
ds_validation = tf.data.Dataset.from_tensor_slices((X_validation[..., np.newaxis], utils.to_categorical(y_validation))).batch(32)

print("X_train shape:", X_train.shape)
print("X_validation shape:", X_validation.shape)
print("X_test shape:", X_test.shape)

# %% [markdown]
# ## Visualize the dataset

# %%
indexes = np.random.choice(range(0, X_train.shape[0]), size=16, replace=False)
samples = zip(X_train[indexes], y_train[indexes])

fig, axs = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle('Random samples')

for ax, sample in zip(axs.flatten(), samples):
    ax.imshow(sample[0], cmap="gray")
    ax.set_title(sample[1])
    ax.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualize the class distribution

# %%
plt.figure()
sns.countplot(x=y_train, hue=y_train, palette="tab10", stat="percent", legend=False)
plt.title("Class")
plt.show()

# %% [markdown]
# ## Build a CNN

# %%
number_classes = np.unique(y_train).size

model = Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add(layers.Rescaling(scale=1 / 255))
model.add(layers.Conv2D(16, kernel_size=(5, 5), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Conv2D(8, kernel_size=(2, 2), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(number_classes, activation="softmax"))

model.summary()

# %% [markdown]
# ## Compile and train the CNN

# %%
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['categorical_accuracy'])

epochs = 10
history_CNN = model.fit(ds_train, epochs=epochs, validation_data=ds_validation)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history_CNN.history['categorical_accuracy'])
ax1.plot(history_CNN.history['val_categorical_accuracy'])
ax1.set_xticks(range(epochs))
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.legend(["Training", "Validation"])

ax2.plot(history_CNN.history['loss'])
ax2.plot(history_CNN.history['val_loss'])
ax2.set_xticks(range(epochs))
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
    ax.imshow(image, cmap="gray")
    ax.set_title("Prediction: " + str(np.argmax(prediction_proba)))
    ax.axis("off")

plt.tight_layout()
plt.show()

# %%
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.grid(False)
plt.show()
