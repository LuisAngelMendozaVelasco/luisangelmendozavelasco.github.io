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
# # Fashion MNIST database

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/IBM_AI_Engineering/portfolio-14.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png"/>Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a basic Convolutional Neural Network (CNN) to classify fashion products.

# %% [markdown]
# The [Fashion MNIST database](https://en.wikipedia.org/wiki/Fashion_MNIST) is a dataset of 28x28 grayscale images of fashion products, designed to serve as a direct drop-in replacement for the original [MNIST database](https://en.wikipedia.org/wiki/MNIST_database) for benchmarking machine learning algorithms. It consists of 70,000 images, with 60,000 images in the training set and 10,000 images in the test set. Each image is labeled with one of 10 categories:
#
# - T-shirt/top: 0
# - Trouser: 1
# - Pullover: 2
# - Dress: 3
# - Coat: 4
# - Sandal: 5
# - Shirt: 6
# - Sneaker: 7
# - Bag: 8
# - Ankle boot: 9
#
# The Fashion MNIST database was created to provide a more challenging classification task than the simple MNIST database. It is freely available and commonly used in machine learning libraries.

# %% [markdown]
# ## Import libraries

# %%
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# %% [markdown]
# ## Load the dataset

# %%
# Download training and test data from open datasets
dataset_train = datasets.FashionMNIST(root='/tmp', download=True, transform=ToTensor())
dataset_right = datasets.FashionMNIST(root='/tmp', train=False, download=True, transform=ToTensor())
dataset_validation, dataset_test = random_split(dataset_right, [int(len(dataset_right) / 2)] * 2)
classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Create data loaders
dataloader_train = DataLoader(dataset_train, batch_size=32)
dataloader_validation = DataLoader(dataset_validation, batch_size=32)
dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test))

print("Number of training samples:", len(dataset_train))
print("Number of validation samples:", len(dataset_validation))
print("Number of test samples:", len(dataset_test))
print("Image size:", tuple(dataset_train[0][0].shape))

# %% [markdown]
# ## Visualize the dataset

# %%
indexes = np.random.choice(range(0, len(dataset_train)), size=16, replace=False)
samples = [(dataset_train[index][0], dataset_train[index][1]) for index in indexes]

fig, axs = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle('Random samples')

for ax, sample in zip(axs.flatten(), samples):
    ax.imshow(sample[0][0], cmap="gray")
    ax.set_title(classes[sample[1]])
    ax.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualize the class distribution

# %%
_, y_train = zip(*dataset_train)

plt.figure()
sns.countplot(x=y_train, hue=y_train, palette="tab10", stat="percent", legend=False)
plt.title("Class")
plt.show()

# %% [markdown]
# ## Build a CNN

# %%
# Get cpu, gpu or mps device for training
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} device!\n")

# Define model
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, number_classes):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            # Channel width after this layer: 28
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            # Channel width after this layer: 14
            nn.MaxPool2d(kernel_size=2),
            # Channel width after this layer: 14
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            # Channel width after this layer: 7
            nn.MaxPool2d(kernel_size=2),
            # Flatten the output of the convolutional layers
            nn.Flatten(),
            # In total we have 32 channels which are each 7 * 7 in size
            nn.Linear(32 * 7 * 7, number_classes))

    def forward(self, x):
        logits = self.linear_relu_stack(x)

        return logits

number_classes = np.unique(y_train).size

model = ConvolutionalNeuralNetwork(number_classes).to(device)
print(model)


# %% [markdown]
# ## Compile and train the CNN

# %%
def train(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    number_batches = len(dataloader)
    model.train()
    train_loss, train_accuracy = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        y_pred = model(X)
        loss = loss_function(y_pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        train_accuracy += (y_pred.argmax(1) == y).type(torch.float).sum().item()
    
    train_loss /= number_batches
    train_accuracy /= size

    return train_accuracy, train_loss

def validation(dataloader, model, loss_function):
    size = len(dataloader.dataset)
    number_batches = len(dataloader)
    model.eval()
    validation_loss, validation_accuracy = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_function(y_pred, y)
            validation_loss += loss.item()
            validation_accuracy += (y_pred.argmax(1) == y).type(torch.float).sum().item()

    validation_loss /= number_batches
    validation_accuracy /= size

    return validation_accuracy, validation_loss


# %%
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

epochs = 10
history = {"accuracy": [], "loss": [], "val_accuracy": [], "val_loss": []}

for epoch in range(epochs):
    start_time = datetime.now()
    accuracy, loss = train(dataloader_train, model, loss_function, optimizer)
    val_accuracy, val_loss = validation(dataloader_validation, model, loss_function)
    stop_time = datetime.now()

    history["accuracy"].append(accuracy)
    history["loss"].append(loss)
    history["val_accuracy"].append(val_accuracy)
    history["val_loss"].append(val_loss)

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"\telapsed time: {(stop_time - start_time).total_seconds():.3f}s - accuracy: {accuracy:.4f} - loss: {loss:.4f} - val_accuracy: {val_accuracy:.4f} - val_loss: {val_loss:.4f}")

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history['accuracy'])
ax1.plot(history['val_accuracy'])
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.legend(["Training", "Validation"])

ax2.plot(history['loss'])
ax2.plot(history['val_loss'])
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.legend(["Training", "Validation"])

plt.show()

# %% [markdown]
# ## Evaluate the CNN

# %%
indexes = np.random.choice(range(0, len(dataset_test)), size=16, replace=False)
images = [dataset_test[index][0] for index in indexes]

fig, axs = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle('Random samples')

with torch.no_grad():
    for image, ax in zip(images, axs.flatten()):
        X = torch.unsqueeze(image, 0).to(device)
        prediction_logits = model(X)
        ax.imshow(image[0], cmap="gray")
        ax.set_title("Prediction: " + classes[prediction_logits.argmax(1).item()])
        ax.axis("off")

plt.tight_layout()
plt.show()

# %%
with torch.no_grad():
    for X_test, y_test in dataloader_test:
        X_test = X_test.to(device)
        y_pred = model(X_test).argmax(1).cpu().numpy()

print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.grid(False)
plt.show()
