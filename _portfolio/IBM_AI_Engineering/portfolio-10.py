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
# # Customer Segmentation

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/IBM_AI_Engineering/portfolio-10.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png"/>Run in Google Colab</a>

# %% [markdown]
# **Objective**: Use *k*-Means Clustering algorithm for customer segmentation.

# %% [markdown]
# [*k*-Means Clustering](https://en.wikipedia.org/wiki/K-means_clustering) is an unsupervised machine learning algorithm used for partitioning data into *k* clusters based on their similarities. It’s a centroid-based algorithm, meaning that each cluster is represented by a central point, called the centroid or mean.

# %% [markdown]
# ## Import libraries

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# %% [markdown]
# ## Load the dataset

# %%
file_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv"
df = pd.read_csv(file_url)
df.head()

# %% [markdown]
# ## Understand the dataset

# %% [markdown]
# [Customer segmentation](https://en.wikipedia.org/wiki/Market_segmentation) is the practice of dividing a customer base into groups of individuals that have similar characteristics. It is a significant strategy because a business can target these specific customer groups and effectively allocate marketing resources. For example, one group might contain high-profit and low-risk customers, those who are more likely to purchase products or subscribe to a service. A business task is to retain those customers.

# %%
df.info()

# %% [markdown]
# ## Visualize the dataset

# %%
number_features = len(df.columns) - 2
grid_rows = int(np.ceil(number_features / 3))
fig, axs = plt.subplots(grid_rows, 3, figsize=(15, 5 * grid_rows))

for ax, feature in zip(axs.flatten(), df.drop(["Customer Id", "Address"], axis=1).columns):
    if len(df[feature].unique()) <= 10:
        sns.countplot(data=df, x=feature, hue=feature, ax=ax, palette="tab10", legend=False)
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
# ## Preprocess the dataset

# %%
df = df.drop(["Customer Id", "Address"], axis=1)
df = df.fillna(0)

# %% [markdown]
# ## Compute *k*-Means Clustering

# %%
kmeans = KMeans(n_clusters=3)
kmeans.fit(df)

# %% [markdown]
# ## Visualize the clusters

# %%
df["Label"] = kmeans.labels_
df.groupby("Label").mean()

# %% [markdown]
# The customers in each cluster are similar to each other demographically. Now we can create a profile for each group, considering the common characteristics of each cluster. 
#
# For example, the 3 clusters can be:
#
# - AFFLUENT, EDUCATED AND OLD AGED
# - MIDDLE AGED AND MIDDLE INCOME
# - YOUNG AND LOW INCOME

# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(df["Edu"], df["Age"], df["Income"], c=df["Label"], cmap="brg")
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')
plt.tight_layout()
plt.show()
