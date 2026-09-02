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
# # Breast Cancer Diagnosis

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/Applied_Data_Science_with_Python_Specialization/portfolio-1.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png"/>Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a k-Nearest Neighbors (k-NN) classifier to predict whether a tumor is malignant or benign.

# %% [markdown]
# ## Import libraries

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# %% [markdown]
# ## Load the dataset

# %%
data = load_breast_cancer(as_frame=True)
df = data.frame
df.head()

# %% [markdown]
# ## Understand the dataset

# %% [markdown]
# The [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) dataset is a classic binary classification dataset.

# %%
print(data.DESCR[:1000], end="...")

# %%
df.info()

# %% [markdown]
# ## Visualize some features of the dataset

# %%
number_features = 10
grid_rows = int(np.ceil(number_features / 3))
fig, axs = plt.subplots(grid_rows, 3, figsize=(15, 5 * grid_rows))

for ax, feature in zip(axs.flatten(), df.columns[:number_features]):
    sns.histplot(data=df, x=feature, ax=ax, hue="target")
    ax.set_xlabel("")
    ax.set_title(feature)

for ax in axs.flatten()[number_features:]:
    ax.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualize the class distribution

# %%
target_feature = df.columns[-1]
class_names = ["malignant", "benign"]
labels, sizes = np.unique(df[target_feature], return_counts=True)

fig, ax = plt.subplots()
ax.pie(sizes, textprops={'color': "w", 'fontsize': '12'}, autopct=lambda pct: "{:.2f}%\n({:d})".format(pct, round(pct/100 * sum(sizes))))
ax.legend([str(i) + " (" + class_names[i] + ")" for i in labels])
ax.set_title(target_feature)
plt.show()

# %% [markdown]
# ## Split the dataset into train and test subsets

# %%
X = df.drop(target_feature, axis=1)
y = df[target_feature]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# %% [markdown]
# ## Find the optimal number of neighbors for a k-NN classifier

# %% [markdown]
# A [k-NN classifier](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) predicts the class of a new data point by finding the k nearest data points in the training set and assigning the majority class among these neighbors to the new point. The algorithm assumes that similar things exist in close proximity, making it intuitive and easy to understand.

# %%
n_neighbors = range(1, 21)
parameters = {'n_neighbors': n_neighbors}

knn_classifier = KNeighborsClassifier()
clf = GridSearchCV(knn_classifier, parameters, scoring='recall')
clf.fit(X_train, y_train)

# %%
plt.figure()
plt.plot(n_neighbors, clf.cv_results_['mean_test_score'])
plt.scatter(clf.best_params_['n_neighbors'], clf.best_score_, color='red')
plt.xlabel("Number of neighbors")
plt.ylabel("Mean recall score")
plt.xticks(n_neighbors[::-2])
plt.show()

# %% [markdown]
# ## Evaluate the best k-NN classifier

# %%
y_pred = clf.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=[str(i) + " (" + class_names[i] + ")" for i in labels])
plt.grid(False)
plt.show()
