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
# # Concrete Crack Detector

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/IBM_AI_Engineering/portfolio-19.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png"/>Run in Google Colab</a>

# %% [markdown]
# **Objective**: Use transfer learning to build an image classifier that detects concrete with and without cracks.

# %% [markdown]
# Pre-trained models such as [VGG-16](https://arxiv.org/abs/1409.1556), [Inception](http://arxiv.org/abs/1512.00567), and [ResNet](https://arxiv.org/abs/1512.03385) can be used for image classification. These models are trained on large datasets like [ImageNet](https://www.image-net.org/), which contains over a million images and can classify images into 1000 object categories. By leveraging these pre-trained models, we can fine-tune them for specific tasks with smaller datasets, improving accuracy and reducing the need for extensive training data.

# %% [markdown]
# ## Import libraries

# %%
import os
import tensorflow as tf
from keras import utils, applications, Input, layers, Model
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# %% [markdown]
# ## Download the dataset

# %% language="bash"
#
# wget -nc --progress=bar:force:noscroll https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0321EN-SkillsNetwork/concrete_data_week3.zip -P /tmp
# unzip -qn /tmp/concrete_data_week3.zip -d /tmp

# %% [markdown]
# ## Load the dataset

# %%
image_size = (64, 64) # Reduce image size to speed up training and reduce memory consumption
batch_size = 100

ds_train = utils.image_dataset_from_directory(directory='/tmp/concrete_data_week3/train',
                                              image_size=image_size,
                                              batch_size=batch_size)

ds_validation, ds_test = utils.image_dataset_from_directory(directory='/tmp/concrete_data_week3/valid',
                                                            image_size=image_size,
                                                            batch_size=batch_size,
                                                            seed=0,
                                                            subset="both",
                                                            validation_split=0.5)

classes = ds_train.class_names

# %% [markdown]
# ## Visualize the dataset

# %% [markdown]
# The **negative** class represents concrete images with no cracks and the **positive** class represents concrete images with cracks.

# %%
images, labels = next(iter(ds_train))

fig, axs = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle('Random samples')

for i, ax in enumerate(axs.flatten()):
    ax.imshow(images[i] / 255)
    ax.set_title(classes[labels[i].numpy()])
    ax.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualize the class distribution

# %%
_, _, negative_files = next(os.walk("/tmp/concrete_data_week3/train/negative"))
_, _, positive_files = next(os.walk("/tmp/concrete_data_week3/train/positive"))
sizes = [len(negative_files), len(positive_files)]

fig, ax = plt.subplots()
ax.pie(sizes, textprops={'color': "w", 'fontsize': '12'}, autopct=lambda pct: "{:.2f}%\n({:d})".format(pct, round(pct/100 * sum(sizes))))
ax.legend(["0 (negative)", "1 (positive)"])
ax.set_title("Class")
plt.show()

# %% [markdown]
# ## Build a CNN using the ResNet50 architecture as a base

# %%
input_shape = (64, 64, 3)
ResNet50 = applications.ResNet50(include_top=False, input_shape=input_shape, pooling="avg")
ResNet50.trainable = False

inputs = Input(shape=input_shape)
x = applications.resnet.preprocess_input(inputs) # Preprocess our images the same way the images used to train ResNet50 model were processed
x = ResNet50(x, training=False)
outputs = layers.Dense(1, activation="sigmoid")(x)

model_resnet = Model(inputs, outputs)
model_resnet.summary()

# %% [markdown]
# ## Compile and train the ResNet50 model

# %%
model_resnet.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])

epochs = 5
history_resnet = model_resnet.fit(ds_train, epochs=epochs, validation_data=ds_validation)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history_resnet.history['binary_accuracy'])
ax1.plot(history_resnet.history['val_binary_accuracy'])
ax1.set_xticks(range(epochs))
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.legend(["Training", "Validation"])

ax2.plot(history_resnet.history['loss'])
ax2.plot(history_resnet.history['val_loss'])
ax2.set_xticks(range(epochs))
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.legend(["Training", "Validation"])

plt.show()

# %% [markdown]
# ## Evaluate the ResNet50 model

# %%
images, labels = next(iter(ds_test))

fig, axs = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle('Random samples')

for i, ax in enumerate(axs.flatten()):
    prediction_proba = model_resnet.predict(tf.expand_dims(images[i], axis=0), verbose=0)
    ax.imshow(images[i] / 255)
    ax.set_title("Prediction: " + classes[int(prediction_proba.squeeze().round())])
    ax.axis("off")

plt.tight_layout()
plt.show()

# %%
_, y_test = zip(*ds_test.unbatch())
y_test = [y.numpy() for y in y_test]

prediction_proba = model_resnet.predict(ds_test, verbose=0)
y_pred = prediction_proba.squeeze().round()

print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["0 (negative)", "1 (positive)"])
plt.grid(False)
plt.show()

# %% [markdown]
# ## Build a CNN using the VGG16 architecture as a base

# %%
input_shape = (64, 64, 3)
VGG16 = applications.VGG16(include_top=False, input_shape=input_shape, pooling="avg")
VGG16.trainable = False

inputs = Input(shape=input_shape)
x = applications.vgg16.preprocess_input(inputs) # Preprocess our images the same way the images used to train VGG16 model were processed
x = VGG16(x, training=False)
outputs = layers.Dense(1, activation="sigmoid")(x)

model_vgg = Model(inputs, outputs)
model_vgg.summary()

# %% [markdown]
# ## Compile and train the VGG16 model

# %%
model_vgg.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])

epochs = 5
history_vgg = model_vgg.fit(ds_train, epochs=epochs, validation_data=ds_validation)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history_vgg.history['binary_accuracy'])
ax1.plot(history_vgg.history['val_binary_accuracy'])
ax1.set_xticks(range(epochs))
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.legend(["Training", "Validation"])

ax2.plot(history_vgg.history['loss'])
ax2.plot(history_vgg.history['val_loss'])
ax2.set_xticks(range(epochs))
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.legend(["Training", "Validation"])

plt.show()

# %% [markdown]
# ## Evaluate the VGG16 model

# %%
images, labels = next(iter(ds_test))

fig, axs = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle('Random samples')

for i, ax in enumerate(axs.flatten()):
    prediction_proba = model_vgg.predict(tf.expand_dims(images[i], axis=0), verbose=0)
    ax.imshow(images[i] / 255)
    ax.set_title("Prediction: " + classes[int(prediction_proba.squeeze().round())])
    ax.axis("off")

plt.tight_layout()
plt.show()

# %%
_, y_test = zip(*ds_test.unbatch())
y_test = [y.numpy() for y in y_test]

prediction_proba = model_vgg.predict(ds_test, verbose=0)
y_pred = prediction_proba.squeeze().round()

print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["0 (negative)", "1 (positive)"])
plt.grid(False)
plt.show()
