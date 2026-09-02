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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MNIST Digits Classification with TensorFlow Serving

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/TensorFlow-Data_and_Deployment/portfolio-7.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective:** Train a Neural Network to classify images of handwritten digits from the MNIST dataset and serve it using TensorFlow Serving.

# %% [markdown]
# [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) is a flexible, high-performance serving system designed for deploying machine learning models in production environments. It handles the inference aspect of ML by managing model lifetimes and providing clients with versioned access via a high-performance, reference-counted lookup table. 

# %% [markdown]
# **Warning: This notebook is designed to be run in a Google Colab only. It installs packages on the system and requires root access.**

# %% [markdown]
# ## Import libraries

# %%
import keras
from keras import Input, layers, Sequential
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import json
import requests
import os
sns.set_style("whitegrid")

# %% [markdown]
# ## Load the dataset

# %% [markdown]
# The [MNIST digits classification dataset](https://keras.io/api/datasets/mnist/) dataset contains 70,000 grayscale images of the digits 0 through 9. The images show individual digits at a low resolution (28 by 28 pixels).

# %%
mnist = keras.datasets.mnist
(X_train, y_train), (X_right, y_right) = mnist.load_data()
X_validation, X_test, y_validation, y_test = train_test_split(X_right, y_right, train_size=0.5, random_state=0)


# %% [markdown]
# ## Data preprocessing

# %% [markdown]
# Scale the pixel values to be between 0.0 and 1.0 and reshape the arrays to be 4-dimensional.

# %%
def preprocess_data(X):
    # Normalize the pixel values to be between 0 and 1
    X = X / 255.0
    # Reshape the data to add a channel dimension (for grayscale images)
    X = np.expand_dims(X, axis=-1)

    return X


# %%
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)
X_validation = preprocess_data(X_validation)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"X_validation shape: {X_validation.shape}, y_validation shape: {y_validation.shape}")

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
with sns.axes_style("whitegrid"):
    plt.figure()
    sns.countplot(x=y_train, hue=y_train, palette="tab10", stat="percent", legend=False)
    plt.title("Class")
    plt.show()

# %% [markdown]
# ## Build the model

# %% [markdown]
# Use the simplest possible Convolutional Neural Network (CNN) architecture to classify the MNIST digits dataset.

# %%
number_classes = np.unique(y_train).size

model = Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add(layers.Conv2D(8, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(number_classes, activation="softmax"))

model.summary()

# %% [markdown]
# ## Compile and train the model

# %%
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

epochs = 5
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_validation, y_validation))

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_xticks(range(epochs))
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.legend(["Training", "Validation"])

ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_xticks(range(epochs))
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

# %% [markdown]
# ## Export the trained model

# %%
MODEL_DIR = "/tmp/digits_model"
version = 1
export_path = os.path.join(MODEL_DIR, str(version))
os.makedirs(export_path, exist_ok=True)
model.export(export_path)

# %% [markdown]
# ## Install TensorFlow Serving

# %% language="bash"
# echo "deb http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && \
# curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
#
# apt update && apt install tensorflow-model-server

# %% [markdown]
# ## Run the TensorFlow Model Server

# %%
os.environ["MODEL_DIR"] = MODEL_DIR

# %% magic_args="--bg" language="bash"
# nohup tensorflow_model_server \
#   --rest_api_port=8501 \
#   --model_name=digits_model \
#   --model_base_path="${MODEL_DIR}" >server.log 2>&1

# %% [markdown]
# ## Make prediction requests to the TensorFlow Model Server

# %%
indexes = np.random.choice(range(0, X_test.shape[0]), size=16, replace=False)
data = json.dumps({"signature_name": "serving_default", "instances": X_test[indexes].tolist()})
headers = {"content-type": "application/json"}
json_response = requests.post('http://127.0.0.1:8501/v1/models/digits_model:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions']

# %% [markdown]
# ## Visualize the predictions

# %%
fig, axs = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle('Random samples')

for i, (image, ax) in enumerate(zip(X_test[indexes], axs.flatten())):
    ax.imshow(image, cmap="gray")
    ax.set_title("Prediction: " + str(np.argmax(predictions[i])))
    ax.axis("off")

plt.tight_layout()
plt.show()
