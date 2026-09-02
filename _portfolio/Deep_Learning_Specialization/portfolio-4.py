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
# # Autonomous Driving

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/Deep_Learning_Specialization/portfolio-4.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Use the "You Only Look Once" (YOLO) algorithm to build a car detection system.

# %% [markdown]
# [You Only Look Once (YOLO)](https://arxiv.org/pdf/1506.02640) is a popular real-time object detection system that achieves high accuracy by requiring only one forward propagation through the network to make predictions. It recognizes objects along with their corresponding bounding boxes.

# %% [markdown]
# ## Download the "Yet Another Darknet 2 Keras" (YAD2K) repository

# %% [markdown]
# [Yet Another Darknet 2 Keras (YAD2K)](https://github.com/allanzelener/YAD2K) is a repository that converts the [Darknet YOLOv2 model](https://pjreddie.com/darknet/yolo/) to a Keras model. The YAD2K code was modified to work with [Keras 3](https://keras.io/keras_3/).

# %% language="bash"
#
# if [ -e "/tmp/YAD2K.zip" ]; then
#     echo "YAD2K.zip already exists!"
# else
#     gdown 1MA4g7d-pJEChYey2PotscZXq9EJr0K2H -O /tmp/
# fi
#
# unzip -qn /tmp/YAD2K.zip -d /tmp

# %% [markdown]
# ## Import libraries

# %%
import sys
sys.path.insert(0, '/tmp/YAD2K')
from yad2k.utils.utils import preprocess_image, read_classes, read_anchors, draw_boxes
from yad2k.models.keras_yolo import space_to_depth_x2, space_to_depth_x2_output_shape, yolo_head, yolo_eval
import os
from keras import models
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.animation as animation
from IPython.display import HTML

# %% [markdown]
# ## Visualize the dataset

# %%
path = "/tmp/YAD2K/images"
files = [os.path.join(path, file) for file in sorted(os.listdir(path)) if file.endswith(".jpg") and file.startswith("0")]
frames = []

fig, ax = plt.subplots()

for file in files:
    image = Image.open(file)
    plt.tight_layout()
    ax.set_title("Frames")
    ax.axis("off")
    frame = ax.imshow(image, animated=True)
    frames.append([frame])
    
anim = animation.ArtistAnimation(fig, frames, blit=True, repeat_delay=1000)
plt.close(fig)

HTML(anim.to_html5_video())

# %% [markdown]
# ## Download the Darknet YOLOv2 model and convert it to a Keras model

# %% language="bash"
#
# wget -nc --progress=bar:force:noscroll https://pjreddie.com/media/files/yolov2.weights -P /tmp/YAD2K/model_data
# wget -nc --progress=bar:force:noscroll https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2.cfg -P /tmp/YAD2K/model_data
#
# if [ -e "/tmp/YAD2K/model_data/yolov2.keras" ]; then
#     echo "yolov2.keras already exists!"
# else
#     /tmp/YAD2K/yad2k.py /tmp/YAD2K/model_data/yolov2.cfg /tmp/YAD2K/model_data/yolov2.weights /tmp/YAD2K/model_data/yolov2.keras
# fi

# %% [markdown]
# ## Load the Keras model, classes, and anchor boxes

# %%
classes = read_classes("/tmp/YAD2K/model_data/coco_classes.txt")
anchors = read_anchors("/tmp/YAD2K/model_data/yolov2_anchors.txt")
model = models.load_model("/tmp/YAD2K/model_data/yolov2.keras", 
                          custom_objects={'space_to_depth_x2': space_to_depth_x2, 'space_to_depth_x2_output_shape': space_to_depth_x2_output_shape})
model.summary()

# %% [markdown]
# ## Evaluate the YOLOv2 model

# %%
file = "/tmp/YAD2K/images/test.jpg"
image, image_data = preprocess_image(file, model_image_size=(608, 608))
model_outputs = model(image_data)
outputs = yolo_head(model_outputs, anchors, len(classes))
out_boxes, out_scores, out_classes = yolo_eval(outputs, [image.size[1], image.size[0]], score_threshold=0.3, iou_threshold=0.5)
output_image = draw_boxes(image.copy(), out_boxes, out_classes, classes, out_scores)

fig, axs = plt.subplots(1, 2, figsize=(15, 5))

axs[0].imshow(image)
axs[0].axis("off")
axs[0].set_title("Input image")

axs[1].imshow(output_image)
axs[1].axis("off")
axs[1].set_title("Output image")

plt.show()

# %%
frames = []

fig, ax = plt.subplots()

for file in files:
    image, image_data = preprocess_image(file, model_image_size=(608, 608))
    model_outputs = model(image_data)
    outputs = yolo_head(model_outputs, anchors, len(classes))
    out_boxes, out_scores, out_classes = yolo_eval(outputs, [image.size[1], image.size[0]], score_threshold=0.3, iou_threshold=0.5)
    output_image = draw_boxes(image.copy(), out_boxes, out_classes, classes, out_scores)
    
    plt.tight_layout()
    ax.set_title("Output frames")
    ax.axis("off")
    frame = ax.imshow(output_image, animated=True)
    frames.append([frame])
    
anim = animation.ArtistAnimation(fig, frames, blit=True, repeat_delay=1000)
plt.close(fig)

HTML(anim.to_html5_video())
