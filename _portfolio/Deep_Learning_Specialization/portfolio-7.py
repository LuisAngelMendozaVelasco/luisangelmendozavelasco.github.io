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
# # Art Generator

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/Deep_Learning_Specialization/portfolio-7.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Generate a novel artistic image using Neural Style Transfer (NST).

# %% [markdown]
# [Neural Style Transfer (NST)](https://en.wikipedia.org/wiki/Neural_style_transfer) is one of the most fun and interesting techniques in deep learning. It merges two images, a "content" image (C-image) and a "style" image (S-image), to create a "generated" image (G-image). The G-image combines the "content" of the C-image with the "style" of the S-image.

# %% [markdown]
# ## Import libraries

# %%
from keras import applications, optimizers, Model, utils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import numpy as np
from IPython.display import HTML
from tqdm import tqdm


# %% [markdown]
# ## Download and load the images

# %%
def load_image(file_path, image_size):
    image = tf.io.read_file(file_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32) # Normalize to [0, 1]
    image = tf.image.resize(image, image_size)
    image = tf.expand_dims(image, axis=0) # Add batch dimension
    
    return image


# %%
content_path = utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style_path = utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

content_image = load_image(content_path, (500, 500))
style_image = load_image(style_path, (500, 500))

# %% [markdown]
# ## Visualize the images 

# %%
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(content_image[0])
axs[0].axis("off")
axs[0].set_title("Content image")

axs[1].imshow(style_image[0])
axs[1].axis("off")
axs[1].set_title("Style image")

plt.show()

# %% [markdown]
# ## Load a pretrained VGG19 model

# %% [markdown]
# We'll load a pretrained Convolutional Neural Network (CNN). We'll use the [Visual Geometry Group (VGG)](https://arxiv.org/abs/1508.06576) network, specifically the VGG19, which is a 19-layer version of the VGG network. This model has already been trained on the extensive [ImageNet database](https://image-net.org/) and has learned to recognize a variety of low-level features, such as edges and simple textures (in the shallower layers), and high-level features, such as more complex textures and object classes (in the deeper layers).

# %%
vgg19 = applications.VGG19(include_top=False, input_shape=(500, 500, 3))
vgg19.trainable = False
vgg19.summary()

# %% [markdown]
# ## Build the model

# %% [markdown]
# We'll build a VGG19 model that returns a list of intermediate layer outputs. These intermediate layers are necessary to define the representation of content and style from the images. For an input image, we'll try to match the corresponding style and content target representations at these intermediate layers.

# %%
content_layers = ['block5_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

outputs = [vgg19.get_layer(layer).output for layer in content_layers + style_layers]
model = Model([vgg19.input], outputs)


# %% [markdown]
# ## Extract style and content

# %% [markdown]
# The style of an image can be described by the means and correlations across the different feature maps. We calculate a [Gram matrix](https://en.wikipedia.org/wiki/Gram_matrix) that includes this information by taking the outer product of the feature vector with itself at each location, and averaging that outer product over all locations.

# %%
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)

    return result/(num_locations)


# %% [markdown]
# Then, we'll build a model that returns the style and content tensors.

# %%
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, model, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = model
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0, 1]"
        inputs = inputs * 255.0
        preprocessed_input = applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)

        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


# %%
extractor = StyleContentModel(model, style_layers, content_layers)
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']


# %% [markdown]
# ## Define the total loss

# %%
def style_content_loss(outputs, style_weight=1e-2, content_weight=1e4):
    """Calculate the mean square error for the image output relative to each target, then take the weighted sum of these losses."""
    style_outputs = outputs['style']
    content_outputs = outputs['content']

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2) for name in style_outputs.keys()])
    style_loss *= style_weight / len(style_layers)

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2) for name in content_outputs.keys()])
    content_loss *= content_weight / len(content_layers)

    loss = style_loss + content_loss

    return loss


# %% [markdown]
# ## Define the training step

# %%
image = tf.Variable(content_image) # Image to optimze, initialized with the content image
optimizer = optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))


# %% [markdown]
# ## Run the style transfer model

# %%
epochs = 1000
frames = []

fig, ax = plt.subplots()

for i in tqdm(range(1, epochs + 1)):
    train_step(image)

    if i % (epochs / 10) == 0 or i == 1:
        plt.tight_layout()
        ax.axis("off")
        text = ax.text(0, -10, "Epoch {}".format(i), fontdict={"fontsize": "large"}, animated=True)
        frame = ax.imshow(image[0], animated=True)
        frames.append([frame, text])
    
anim = animation.ArtistAnimation(fig, frames, blit=True, repeat_delay=1000)
plt.close(fig)

HTML(anim.to_html5_video())
