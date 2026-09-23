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
#     display_name: pytorch
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Real-Time Object Detection

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/IBM_AI_Engineering/portfolio-15.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png"/>Run in Google Colab</a>

# %% [markdown]
# **Objective**: Build a pedestrian detection system using a pretrained Faster R-CNN.

# %% [markdown]
# [Faster Region-based Convolutional Neural Networks (R-CNN)](https://arxiv.org/pdf/1506.01497) is an advanced object detection model that improves upon its predecessor, [Fast R-CNN](https://arxiv.org/pdf/1504.08083), by incorporating a Region Proposal Network (RPN) which shares full-image convolutional features with the detection network, enabling nearly cost-free region proposals. This RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position, trained end-to-end to generate high-quality region proposals. The Faster R-CNN architecture consists of two main modules: a deep fully convolutional network that proposes regions and a Fast R-CNN detector that uses these proposed regions for detection. Despite its name, Faster R-CNN is known for being more accurate but slower than some other models like [YOLOv3](https://arxiv.org/pdf/1804.02767) or [MobileNet](https://en.wikipedia.org/wiki/MobileNet) during inference.

# %% [markdown]
# ## Import libraries

# %%
import torch
from torchvision.transforms import Resize, functional
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.io import decode_image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from PIL import ImageDraw, ImageFont
import os

# %% [markdown]
# ## Download the dataset

# %% language="bash"
#
# wget -nc --progress=bar:force:noscroll https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip -P /tmp
# wget -nc --progress=bar:force:noscroll https://assets.ubuntu.com/v1/0cef8205-ubuntu-font-family-0.83.zip -P /tmp
# unzip -qn /tmp/PennFudanPed.zip -d /tmp
# unzip -qn /tmp/0cef8205-ubuntu-font-family-0.83.zip -d /tmp

# %% [markdown]
# ## Visualize the dataset

# %% [markdown]
# The [Penn-Fudan Database](https://www.cis.upenn.edu/~jshi/ped_html/) is a dataset used for pedestrian detection and segmentation. It contains 170 images with 345 instances of pedestrians, taken from scenes around the University of Pennsylvania and Fudan University. This dataset is often used in research and training of object detection and instance segmentation models.

# %%
path = "/tmp/PennFudanPed/PNGImages"
files = [os.path.join(path, file) for file in sorted(os.listdir(path)) if file.endswith(".png")]
resize = Resize(size=(450, 450))
frames = []

fig, ax = plt.subplots()

for file in files:
    image = decode_image(file)
    image = resize(image)
    plt.tight_layout()
    ax.set_title("Frames")
    ax.axis("off")
    frame = ax.imshow(image.permute(1, 2, 0), animated=True)
    frames.append([frame])
    
anim = animation.ArtistAnimation(fig, frames, blit=True, repeat_delay=1000)
plt.close(fig)

HTML(anim.to_html5_video())

# %% [markdown]
# ## Load the pretrained Faster R-CNN

# %% [markdown]
# The model was trained using the [COCO dataset](https://cocodataset.org/).

# %%
# Run on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using {device} device!\n")

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights).to(device)
model.eval()
print(model)


# %% [markdown]
# ## Evaluate the Faster R-CNN

# %%
def draw_boxes(image, prediction, score_threshold=0.9, font_size=14, box_width=5, colors=["white", "black"]):
    categories = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]
    indexes = [prediction['scores'] > score_threshold]
    boxes = prediction["boxes"][indexes].to(torch.int64).tolist()
    labels = [categories[i] for i in prediction["labels"][indexes]]
    scores = (prediction['scores'][indexes] * 100).to(torch.int64).tolist()
    labels_scores = [label + " " + str(score) + "%" for label, score in zip(labels, scores)]
    pil_image = functional.to_pil_image(image)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(font="/tmp/ubuntu-font-family-0.83/Ubuntu-B.ttf", size=font_size)

    for box, label in zip(boxes, labels_scores):
        draw.rectangle(box, width=box_width, outline=colors[0])
        text_box = draw.textbbox((box[0], box[1] - font_size), label, font=font)
        draw.rectangle(text_box, fill=colors[0])
        draw.text((box[0], box[1] - font_size), label, fill=colors[1], font=font)

    return pil_image


# %%
transforms = weights.transforms()
file = files[0]
image = decode_image(file)
image = resize(image)

with torch.no_grad():
    X = transforms(image).to(device)
    prediction = model([X])[0]

output_image = draw_boxes(image, prediction)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(image.permute(1, 2, 0))
axs[0].set_title("Input image")
axs[0].axis("off")

axs[1].imshow(output_image)
axs[1].set_title("Output image")
axs[1].axis("off")

plt.show()

# %%
frames = []

fig, ax = plt.subplots()

for file in files:
    image = decode_image(file)
    image = resize(image)

    with torch.no_grad():
        X = transforms(image).to(device)
        prediction = model([X])[0]

    output_image = draw_boxes(image, prediction)

    plt.tight_layout()
    ax.set_title("Output frames")
    ax.axis("off")
    frame = ax.imshow(output_image, animated=True)
    frames.append([frame])
    
anim = animation.ArtistAnimation(fig, frames, blit=True, repeat_delay=1000)
plt.close(fig)

HTML(anim.to_html5_video())
