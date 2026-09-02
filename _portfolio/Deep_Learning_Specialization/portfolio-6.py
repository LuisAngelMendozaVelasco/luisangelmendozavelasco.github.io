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
# # Face Recognition

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/Deep_Learning_Specialization/portfolio-6.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Build a facial recognition system using a pretrained FaceNet model.

# %% [markdown]
# [FaceNet](https://arxiv.org/pdf/1503.03832.pdf) is a neural network that encodes the image of a face image into a vector of 128 numbers. By comparing two of these vectors, we can then determine if two pictures are of the same person.

# %% [markdown]
# ## Download the dataset and FaceNet

# %% language="bash"
#
# if [ -e "/tmp/keras_facenet.zip" ]; then
#     echo "keras_facenet.zip already exists!"
# else
#     gdown 1pbwUyttn5MWkc4kciW52Xlpf5esUjJUo -O /tmp/
# fi
#
# unzip -qn /tmp/keras_facenet.zip -d /tmp

# %% [markdown]
# ## Import libraries

# %%
import sys
sys.path.insert(0, '/tmp')
from keras_facenet.inception_resnet_v1 import InceptionResNetV1
from keras import utils
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

# %% [markdown]
# ## Visualize the dataset

# %% [markdown]
# This is a small dataset containing photos of some famous celebrities.

# %%
path = "/tmp/keras_facenet/Celebrity_Faces_Dataset"

fig, axs = plt.subplots(5, 5, figsize=(10, 10))

for i, celebrity in enumerate(os.listdir(path)):
    for j, file in enumerate(os.listdir(os.path.join(path, celebrity))):
        image = Image.open(os.path.join(path, celebrity, file)).resize((160, 160))
        axs[i, j].imshow(image)
        axs[i, j].set_title(celebrity.replace("_", " "))
        axs[i, j].axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Load the pretrained FaceNet

# %% [markdown]
# This network follows the [Inception-ResNet V1 architecture](https://arxiv.org/pdf/1409.4842) and uses 160x160 dimensional RGB images as input. The output is a matrix that encodes each input image into a 128-dimensional vector.

# %%
model = InceptionResNetV1(weights_path="/tmp/keras_facenet/weights.h5")


# %% [markdown]
# ## Evaluate the FaceNet

# %% [markdown]
# By calculating the distance between two encodings and setting a threshold, we can determine whether two pictures represent the same person. Therefore, an encoding is good if:
#
# - The encodings of two images of the same person are quite similar to each other.
# - The encodings of two images of different people are very different.

# %%
def encode_image(image_path, model):
    image = utils.load_img(image_path, target_size=(160, 160))
    image = utils.img_to_array(image) / 255
    image = np.array([image])
    encoding = model.predict_on_batch(image)
    encoding = encoding / np.linalg.norm(encoding, ord=2)
    
    return encoding

def who_is_it(image_path, database, model):
    """
    Implements face recognition by finding who is the person on the image_path image.
    
    Arguments:
        image_path -- Path to an image
        database -- Database containing image encodings along with the name of the person on the image
        model -- Inception model instance in Keras
    
    Returns:
        minimum_distance -- The minimum distance between image_path encoding and the encodings from the database
        identity -- String, the name prediction for the person on image_path
    """

    ## Step 1: Compute the target "encoding" for the image.
    encoding = encode_image(image_path, model)
    
    ## Step 2: Find the closest encoding.
    # Initialize "minimum_distance" to a large value.
    minimum_distance = 100
    
    # Loop over the database dictionary's names and encodings.
    for (name, database_encoding) in database.items():
        # Compute L2 distance between the target "encoding" and the current database_encoding from the database.
        distance = np.linalg.norm(encoding - database_encoding)

        # If this distance is less than the minimum_distance, then set minimum_distance to distance, and identity to name.
        if distance < minimum_distance:
            minimum_distance = distance
            identity = name
        
    return minimum_distance, identity


# %% [markdown]
# Build a database containing one encoding vector for each celebrity.

# %%
path = "/tmp/keras_facenet/Celebrity_Faces_Dataset"
database = {}

for celebrity in os.listdir(path):
    for file in os.listdir(os.path.join(path, celebrity)):
        if file.endswith('1.jpg'):
            database[celebrity] = encode_image(os.path.join(path, celebrity, file), model)

# %%
path = "/tmp/keras_facenet/Celebrity_Faces_Dataset"

fig, axs = plt.subplots(5, 5, figsize=(10, 10))

for i, celebrity in enumerate(os.listdir(path)):
    for j, file in enumerate(os.listdir(os.path.join(path, celebrity))):
        image = Image.open(os.path.join(path, celebrity, file)).resize((160, 160))
        minimum_distance, identity = who_is_it(os.path.join(path, celebrity, file), database, model)
        axs[i, j].imshow(image)
        axs[i, j].set_title("Prediction:\n" + identity.replace("_", " ") + "\nDistance = " + str(round(minimum_distance, 2)))
        axs[i, j].axis("off")

plt.tight_layout()
plt.show()
