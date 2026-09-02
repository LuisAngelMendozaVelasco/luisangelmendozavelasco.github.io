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
# # Dimensionality Reduction

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/IBM_Introduction_to_Machine_Learning_Specialization/portfolio-7.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Use Principal Component Analysis (PCA) to reduce the dimensionality of a given dataset while retaining most of the information.

# %% [markdown]
# [Principal Component Analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) is a statistical technique used to reduce the dimensionality of complex datasets by extracting the principal components that contain the most information while rejecting noise or less important data. It transforms an extensive collection of variables into a smaller group that retains nearly all the data in the larger set. The principal components are orthogonal, meaning there is no relationship within a pair of variables, and they are ranked by the amount of variance they explain, with the first principal component explaining the most variance.

# %% [markdown]
# ## Import libraries

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import MinMaxScaler

sns.set_style("whitegrid")

# %% [markdown]
# ## Load the dataset

# %%
file_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/Wholesale_Customers_Data.csv"
df = pd.read_csv(file_url)
df.head()

# %% [markdown]
# ## Understand the dataset

# %% [markdown]
# We will be using customer data from a [Portuguese wholesale distributor](https://archive.ics.uci.edu/dataset/292/wholesale+customers) for clustering.
#
# It contains the following features:
#
# - **Fresh**: annual spending (m.u.) on fresh products
# - **Milk**: annual spending (m.u.) on milk products
# - **Grocery**: annual spending (m.u.) on grocery products
# - **Frozen**: annual spending (m.u.) on frozen products
# - **Detergents_Paper**: annual spending (m.u.) on detergents and paper products
# - **Delicatessen**: annual spending (m.u.) on delicatessen products
# - **Channel**: customer channel (1: hotel/restaurant/cafe or 2: retail)
# - **Region**: customer region (1: Lisbon, 2: Porto, 3: Other)
#
# In this data, the values for all spending are given in an arbitrary unit (m.u. = monetary unit).

# %%
df.info()

# %% [markdown]
# ## Visualize the dataset

# %%
number_features = df.shape[1] - 2
grid_rows = int(np.ceil(number_features / 3))
fig, axs = plt.subplots(grid_rows, 3, figsize=(15, 5 * grid_rows))

for ax, feature in zip(axs.flatten(), df.drop(columns=["Region", "Channel"]).columns):
    sns.histplot(data=df, x=feature, ax=ax)
    ax.set_xlabel("")
    ax.set_title(feature)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Preprocess the dataset

# %%
preprocessed_df = df.drop(columns=['Channel', 'Region'])
skew_values = preprocessed_df.skew()
scaler = MinMaxScaler()

for column in preprocessed_df.columns:
    if abs(skew_values[column]) > 0.75:
        preprocessed_df[column] = np.log1p(preprocessed_df[column])
    
    preprocessed_df[column] = scaler.fit_transform(preprocessed_df[[column]])

preprocessed_df.head()

# %% [markdown]
# ## Visualize the dataset after preprocessing

# %%
number_features = preprocessed_df.shape[1]
grid_rows = int(np.ceil(number_features / 3))
fig, axs = plt.subplots(grid_rows, 3, figsize=(15, 5 * grid_rows))

for ax, feature in zip(axs.flatten(), preprocessed_df.columns):
    sns.histplot(data=preprocessed_df, x=feature, ax=ax)
    ax.set_xlabel("")
    ax.set_title(feature)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Perform PCA

# %%
pca_list = list()
feature_weight_list = list()

# Fit a range of PCA models
for n in range(1, preprocessed_df.shape[1]):
    # Create and fit the model
    pca = PCA(n_components=n)
    pca.fit(preprocessed_df)
    
    # Store the model and variance
    pca_list.append(pd.Series({'n': n, 'model': pca, 'var': pca.explained_variance_ratio_.sum()}))
    
    # Calculate and store feature importances
    abs_feature_values = np.abs(pca.components_).sum(axis=0)
    feature_weight_list.append(pd.DataFrame({'n': n, 'features': preprocessed_df.columns, 'values': abs_feature_values / abs_feature_values.sum()}))
    
pca_df = pd.concat(pca_list, axis=1).T.set_index('n')
features_df = pd.concat(feature_weight_list).pivot(index='n', columns='features', values='values')

# %% [markdown]
# ## Evaluate the PCA results

# %% [markdown]
# Because it retains more than **75%** of variance, we can reduce the dataset to **3** dimensions.

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

sns.barplot(data=pca_df, x=pca_df.index, y='var', hue=pca_df.index, palette='tab10', legend=False, ax=axs[0])
axs[0].axhline(y=0.75, color="black", linestyle="--")
axs[0].set_xlabel('Number of components')
axs[0].set_ylabel('Percentage of variance explained')

features_df.plot(kind='bar', ax=axs[1])
axs[1].legend().set_title("")
axs[1].set_xlabel('Number of components')
axs[1].set_ylabel('Relative importance')

plt.show()

# %%
pca = PCA(n_components=3)
df_pca = pca.fit_transform(preprocessed_df)
df_pca = pd.DataFrame(df_pca, columns=[f'PC{i+1}' for i in range(df_pca.shape[1])])

number_features = df_pca.shape[1]
grid_rows = int(np.ceil(number_features / 3))
fig, axs = plt.subplots(grid_rows, 3, figsize=(15, 5 * grid_rows))

for ax, feature in zip(axs.flatten(), df_pca.columns):
    sns.histplot(data=df_pca, x=feature, ax=ax)
    ax.set_xlabel("")
    ax.set_title(feature)

plt.tight_layout()
plt.show()
