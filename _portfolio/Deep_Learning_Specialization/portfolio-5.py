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
#     display_name: deep-learning
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Image Segmentation

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/Deep_Learning_Specialization/portfolio-5.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a U-Net to predict a label for every pixel in an image from an autonomous driving dataset.

# %% [markdown]
# The [U-Net](https://en.wikipedia.org/wiki/U-Net) is convolutional network architecture for fast and precise segmentation of images. This type of image classification is similar to object detection in that both ask the question, "What objects are in this image and where in the image are those objects located?", but where object detection systems with bounding boxes include pixels that aren't part of the object, semantic image segmentation can predict an accurate mask for each object in an image by labeling each pixel with its corresponding class.

# %% [markdown]
# ## Import libraries

# %%
import tensorflow as tf
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from keras import layers, Input, Model, losses, callbacks
from IPython.display import HTML
import kagglehub
import seaborn as sns
sns.set_style("whitegrid")

# %% [markdown]
# ## Download the dataset

# %% [markdown]
# Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mojtabanafez/self-driving-car-dataset-ai-course).

# %%
kagglehub.dataset_download("mojtabanafez/self-driving-car-dataset-ai-course", output_dir="/tmp/self-driving-car-dataset-ai-course")


# %% [markdown]
# ## Load the dataset

# %%
def load_image_mask(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, (144, 192), method="nearest")
    image = tf.image.convert_image_dtype(image, "float32")

    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    mask = tf.image.resize(mask, (144, 192), method="nearest")
    mask = tf.image.convert_image_dtype(mask, "uint8")

    return image, mask


# %%
image_directory = "/tmp/self-driving-car-dataset-ai-course/Files/data/CameraRGB"
mask_directory = "/tmp/self-driving-car-dataset-ai-course/Files/data/CameraMask"
image_paths = [os.path.join(image_directory, file) for file in sorted(os.listdir(image_directory)) if file.endswith(".png")]
mask_paths = [os.path.join(mask_directory, file) for file in sorted(os.listdir(mask_directory)) if file.endswith(".png")]
image_paths_train, image_paths_validation, image_paths_test = image_paths[:800], image_paths[800:900], image_paths[900:]
mask_paths_train, mask_paths_validation, mask_paths_test = mask_paths[:800], mask_paths[800:900], mask_paths[900:]

ds_train = tf.data.Dataset.from_tensor_slices((image_paths_train, mask_paths_train)).map(load_image_mask).batch(32)
ds_validation = tf.data.Dataset.from_tensor_slices((image_paths_validation, mask_paths_validation)).map(load_image_mask).batch(32)

# %%
print("Number of training samples:", len(image_paths_train))
print("Number of validation samples:", len(image_paths_validation))
print("Number of test samples:", len(image_paths_test))
print("Each image has a shape of", tuple(load_image_mask(image_paths_train[0], mask_paths_train[0])[0].shape))

# %% [markdown]
# ## Visualize the dataset

# %%
# if not os.path.exists("/tmp/portfolio-5.mp4"):
#     frames = []

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

#     for image_path, mask_path in zip(image_paths_train, mask_paths_train):
#         image, mask = load_image_mask(image_path, mask_path)

#         ax1.set_title("Image")
#         ax1.axis("off")
#         frame1 = ax1.imshow(image, animated=True)

#         ax2.set_title("Mask")
#         ax2.axis("off")
#         frame2 = ax2.imshow(mask, animated=True, cmap="Paired")

#         plt.tight_layout()
#         frames.append([frame1, frame2])

#     anim = animation.ArtistAnimation(fig, frames, interval=25, blit=True, repeat_delay=1000)
#     plt.close(fig)

#     anim.save('/tmp/portfolio-5.mp4', writer=animation.FFMpegWriter(fps=40))

# %% language="html"
#
# <iframe src="https://drive.google.com/file/d/1YOj8Bwf98uith5JL5IudSvKP9cbp5ZOb/preview" width="640" height="480"></iframe>

# %% [markdown]
# ## Build a U-Net

# %% [markdown]
# U-Net uses a matching number of convolutions to downsample the input image to a feature map, and transposed convolutions to upsample those maps to the size of the original input image. It also adds skip connections to retain information that would otherwise become lost during encoding. Skip connections send information to each upsampling layer in the decoder from the corresponding downsampling layer in the encoder, capturing finer information while keeping a low computation level. The latter helps to prevent information loss, as well as model overfitting.
#
# ![U-Net Architecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

# %%
#########
# Input #
#########
inputs = Input(shape=(144, 192, 3))

################
# Downsampling #
################

# Block 1
x = layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
block_1 = x
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

# Block 2
x = layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
block_2 = x
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

# Block 3
x = layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
block_3 = x
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

# Block 4
x = layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
x = layers.Dropout(0.3)(x)
block_4 = x
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

# Block 5
x = layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
x = layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
x = layers.Dropout(0.3)(x)

##############
# Upsampling #
##############

# Block 1
x = layers.Conv2DTranspose(filters=256, kernel_size=3, strides=(2, 2), padding='same')(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(layers.concatenate([x, block_4]))
x = layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(x)

# Block 2
x = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=(2, 2), padding='same')(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(layers.concatenate([x, block_3]))
x = layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(x)

# Block 3
x = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='same')(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(layers.concatenate([x, block_2]))
x = layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(x)

# Block 4
x = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding='same')(x)
x = layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(layers.concatenate([x, block_1]))
x = layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(x)

##########
# Output #
##########
number_classes = 23
x = layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
outputs = layers.Conv2D(filters=number_classes, kernel_size=1, padding='same')(x)
model = Model(inputs=inputs, outputs=outputs)

model.summary()


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
            print(f"Epoch {epoch+1}/{self.epochs_to_show[-1] + 1}")
            print(f"\telapsed time: {(self.epoch_stop_time - self.epoch_start_time).total_seconds():.3f}s - accuracy: {logs['accuracy']:.4f} - loss: {logs['loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f} - val_loss: {logs['val_loss']:.4f}")


# %% [markdown]
# ## Compile and train the U-Net

# %%
model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

epochs = 50
epochs_to_show = [0] + [i for i in range(int(epochs/10)-1, epochs, int(epochs/10))]
custom_verbose = CustomVerbose(epochs_to_show)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=int(epochs/10), verbose=1)
history = model.fit(ds_train, epochs=epochs, verbose=0, validation_data=ds_validation, callbacks=[custom_verbose, early_stopping])

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
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
# ## Evaluate the U-Net

# %%
def predict_mask(image, model):
    mask_pred = model.predict(image[tf.newaxis, ...], verbose=0)
    mask_pred = tf.argmax(mask_pred, axis=-1)
    mask_pred = mask_pred[..., tf.newaxis]

    return mask_pred[0]


# %%
frames = []

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

for image_path, mask_path in zip(image_paths_test, mask_paths_test):
    image, mask = load_image_mask(image_path, mask_path)

    ax1.set_title("Image")
    ax1.axis("off")
    frame1 = ax1.imshow(image, animated=True)

    ax2.set_title("True mask")
    ax2.axis("off")
    frame2 = ax2.imshow(mask, animated=True, cmap="Paired")

    ax3.set_title("Predicted mask")
    ax3.axis("off")
    frame3 = ax3.imshow(predict_mask(image, model), animated=True, cmap="Paired")

    plt.tight_layout()
    frames.append([frame1, frame2, frame3])

anim = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
plt.close(fig)

HTML(anim.to_html5_video())
