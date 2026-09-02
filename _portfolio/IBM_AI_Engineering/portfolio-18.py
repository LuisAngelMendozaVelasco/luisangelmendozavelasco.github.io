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
# # Autoencoder

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/IBM_AI_Engineering/portfolio-18.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png"/>Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a basic autoencoder for image reconstruction.

# %% [markdown]
# An [autoencoder](https://en.wikipedia.org/wiki/Autoencoder) is an artificial neural network employed to recreate the given input. It takes a set of unlabeled inputs, encodes them and then tries to extract the most valuable information from them. They are used for feature extraction, learning generative models of data, dimensionality reduction and compression. Autoencoders, based on [Restricted Boltzmann Machine (RBM)](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine), are employed in some of the largest deep learning applications. They are the building blocks of [Deep Belief Networks (DBN)](https://en.wikipedia.org/wiki/Deep_belief_network).
#
# An autoencoder can be divided in two parts, the **encoder** and the **decoder**. The **encoder** compress the representation of an input by running the data through its layers. The **decoder** works like the **encoder** network in reverse. It works to recreate the input as closely as possible. The training procedure produces at the center of the network a compressed, low dimensional representation that can be decoded to obtain the higher dimensional representation with minimal loss of information between the input and the output.

# %% [markdown]
# ## Import libraries

# %%
import tensorflow as tf
from keras import datasets, layers, losses, optimizers, Model
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# %% [markdown]
# ## Load the dataset

# %%
(X_train, _), (X_test, _) = datasets.mnist.load_data()
X_train, X_test = X_train.astype('float32') / 255, X_test.astype('float32') / 255

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# %% [markdown]
# ## Visualize the dataset

# %%
indexes = np.random.choice(range(0, X_train.shape[0]), size=16, replace=False)
samples = X_train[indexes]

fig, axs = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle('Random samples')

for ax, sample in zip(axs.flatten(), samples):
    ax.imshow(sample, cmap="gray")
    ax.axis("off")

plt.tight_layout()
plt.show()


# %% [markdown]
# ## Build an Autoencoder

# %%
class AutoEncoder(Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.n_hidden_1 = 256 # 1st layer num features
        self.n_hidden_2 = 128 # 2nd layer num features
        self.encoding_layer = 32
        self.n_input = 784 # MNIST data input (img shape: 28*28)

        self.flatten_layer = layers.Flatten()
        self.encoding_1 = layers.Dense(self.n_hidden_1, activation=tf.nn.sigmoid)
        self.encoding_2 = layers.Dense(self.n_hidden_2, activation=tf.nn.sigmoid)
        self.encoding_final = layers.Dense(self.encoding_layer, activation=tf.nn.relu)
        self.decoding_1 = layers.Dense(self.n_hidden_2, activation=tf.nn.sigmoid)
        self.decoding_2 = layers.Dense(self.n_hidden_1, activation=tf.nn.sigmoid)
        self.decoding_final = layers.Dense(self.n_input)

    # Build the encoder
    def encoder(self, x):
        x = self.flatten_layer(x)
        layer_1 = self.encoding_1(x)
        layer_2 = self.encoding_2(layer_1)
        code = self.encoding_final(layer_2)

        return code

    # Build the decoder
    def decoder(self, x):
        layer_1 = self.decoding_1(x)
        layer_2 = self.decoding_2(layer_1)
        decode = self.decoding_final(layer_2)

        return decode

    def call(self, x):
        encoder_op = self.encoder(x)
        # Reconstructed Images
        y_pred = self.decoder(encoder_op)

        return y_pred
        
def cost(y_true, y_pred):
    loss = losses.mean_squared_error(y_true, y_pred)
    cost = tf.reduce_mean(loss)
    
    return cost

def grad(model, inputs, targets):
    targets = layers.Flatten()(targets)

    with tf.GradientTape() as tape:    
        reconstruction = model(inputs)
        loss_value = cost(targets, reconstruction)

    return loss_value, tape.gradient(loss_value, model.trainable_variables), reconstruction


# %% [markdown]
# ## Train the Autoencoder

# %%
learning_rate = 0.01
epochs = 20
batch_size = 256
total_batch = int(len(X_train) / batch_size)
epochs_to_show = [0] + [i for i in range(int(epochs / 10) - 1, epochs, int(epochs / 10))]

model = AutoEncoder()
optimizer = optimizers.RMSprop(learning_rate)
loss_values = []

for epoch in range(epochs):
    start_time = time.time()

    for i in range(total_batch):
        X_input = X_train[i * batch_size : i * batch_size + batch_size]
        loss_value, grads, reconstruction = grad(model, X_input, X_input)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    loss_values.append(loss_value)

    # Display logs
    if epoch in epochs_to_show:
        print(f"Epoch {epoch + 1}/{epochs}:")
        print(f"\telapsed time: {(time.time() - start_time):.3f}s - cost: {loss_value:.4f}")

# %%
plt.figure()
plt.plot(loss_values)
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.xticks(range(0, epochs, 2))
plt.show()

# %% [markdown]
# ## Evaluate the Autoencoder

# %% [markdown]
# It can be seen that some noise were added to the images.

# %%
# Applying encode and decode over test set
indexes = np.random.choice(range(0, X_test.shape[0]), size=10, replace=False)
X_input = layers.Flatten()(X_test[indexes])
encode_decode = model(X_input)

# Compare original images with their reconstructions
fig, axs = plt.subplots(2, 10, figsize=(10, 2))
plt.suptitle("Originals (top) vs Reconstructions (bottom)")

for i, index in enumerate(indexes):
    axs[0][i].imshow(np.reshape(X_test[index], (28, 28)), cmap="gray")
    axs[0][i].axis("off")
    
    axs[1][i].imshow(np.reshape(encode_decode[i], (28, 28)), cmap="gray")
    axs[1][i].axis("off")

plt.show()
