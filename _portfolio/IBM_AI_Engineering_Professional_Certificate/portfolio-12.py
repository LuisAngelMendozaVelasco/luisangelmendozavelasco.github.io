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
# # Compressive Strength of Concrete

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/IBM_AI_Engineering/portfolio-12.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png"/>Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a basic Neural Network (NN) for regression to predict the compressive strength of concrete.

# %% [markdown]
# ## Import libraries

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import Sequential, Input, layers, callbacks
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# %% [markdown]
# ## Load the dataset

# %%
file_url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv'
df = pd.read_csv(file_url)
df.head()

# %% [markdown]
# ## Understand the dataset

# %% [markdown]
# The dataset is about the compressive strength of different samples of concrete based on the volumes of the different ingredients that were used to make them. Ingredients include:
#
# 1. Cement
# 2. Blast Furnace Slag
# 3. Fly Ash
# 4. Water
# 5. Superplasticizer
# 6. Coarse Aggregate
# 7. Fine Aggregate
#
# So the first concrete sample has 540 m³ of cement, 0 m³ of blast furnace slag, 0 m³ of fly ash, 162 m³ of water, 2.5 m³ of superplaticizer, 1040 m³ of coarse aggregate, and 676 m³ of fine aggregate. Such a concrete mix which is 28 days old, has a compressive strength of 79.99 MPa.

# %%
df.info()

# %% [markdown]
# ## Visualize the dataset

# %%
number_features = len(df.columns) - 1
grid_rows = int(np.ceil(number_features / 3))
fig, axs = plt.subplots(grid_rows, 3, figsize=(15, 5 * grid_rows))

for ax, feature in zip(axs.flatten(), df.columns[:-1]):
    if len(df[feature].unique()) <= 10:
        labels, sizes = np.unique(df[feature], return_counts=True)
        sns.barplot(x=labels, y=sizes, hue=labels, ax=ax, palette="tab10", legend=False)
        ax.set_xlabel("")
        ax.set_title(feature)
    else:
        sns.histplot(data=df, x=feature, ax=ax)
        ax.set_xlabel("")
        ax.set_title(feature)

for ax in axs.flatten()[number_features:]:
    ax.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualize the target feature

# %%
plt.figure()
sns.histplot(data=df, x="Strength", kde=True)
plt.title("Histogram")
plt.show()

# %% [markdown]
# ## Preprocess the dataset

# %%
X = df.drop("Strength", axis=1)
y = df["Strength"]

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=df.columns[:-1])
X.head()

# %% [markdown]
# ## Split the dataset into train, validation and test subsets

# %%
X_train, X_right, y_train, y_right = train_test_split(X, y, train_size=0.8, random_state=0)
X_validation, X_test, y_validation, y_test = train_test_split(X_right, y_right, train_size=0.5, random_state=0)

print("X_train shape:", X_train.shape)
print("X_validation shape:", X_validation.shape)
print("X_test shape:", X_validation.shape)


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
            print(f"Epoch {epoch + 1}/{self.epochs_to_show[-1] + 1}")
            print(f"\telapsed time: {(self.epoch_stop_time - self.epoch_start_time).total_seconds():.3f}s - r2_score: {logs['r2_score']:.4f} - loss: {logs['loss']:.4f} - val_r2_score: {logs['val_r2_score']:.4f} - val_loss: {logs['val_loss']:.4f}")


# %% [markdown]
# ## Build a NN

# %%
model = Sequential()
model.add(Input(shape=(X.shape[1],)))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(1))

model.summary()

# %% [markdown]
# ## Compile and train the NN

# %%
model.compile(optimizer="adam", loss='mean_squared_error', metrics=['r2_score'])

epochs = 500
patience = int(epochs / 10)
epochs_to_show = [0] + [i for i in range(patience - 1, epochs, patience)]
custom_verbose = CustomVerbose(epochs_to_show)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
history = model.fit(x=X_train, y=y_train, epochs=epochs, verbose=0, validation_data=(X_validation, y_validation), callbacks=[custom_verbose, early_stopping])

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history.history['r2_score'])
ax1.plot(history.history['val_r2_score'])
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Score")
ax1.legend(["Training", "Validation"])
ax1.set_ylim(0, 1)

ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.legend(["Training", "Validation"])
ax2.set_ylim(0, 200)

plt.show()

# %% [markdown]
# ## Evaluate the NN

# %%
y_pred = model.predict(X_test, verbose=0)

print(f"MSE = {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² = {r2_score(y_test, y_pred):.2f}")
