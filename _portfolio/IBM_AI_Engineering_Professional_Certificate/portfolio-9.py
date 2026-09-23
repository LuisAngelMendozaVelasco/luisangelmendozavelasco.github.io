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
# # Iris Flower Species

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/IBM_AI_Engineering/portfolio-9.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png"/>Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a Logistic Regression classifier to predict Iris flower species.

# %% [markdown]
# [Multiclass classification](https://en.wikipedia.org/wiki/Multiclass_classification) or multinomial classification is the problem of classifying instances into multiple classes.

# %% [markdown]
# ## Import libraries

# %%
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# %% [markdown]
# ## Load the dataset

# %%
X, y = datasets.load_iris(return_X_y=True, as_frame=True)
df = pd.concat([X, y.to_frame()], axis=1)
df.head()

# %% [markdown]
# ## Understand the dataset

# %% [markdown]
# The well-known [Iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) consists of 50 samples of each of three Iris species (Iris setosa, Iris virginica and Iris versicolor). Four features were measured for each sample: the length and the width of the sepals and petals, in centimeters.

# %%
df.info()

# %% [markdown]
# ## Visualize the dataset

# %%
sns.pairplot(data=df, hue="target", diag_kind="hist", palette="dark")
plt.show()

# %% [markdown]
# ## Visualize the class distribution

# %%
target_feature = df.columns[-1]
class_names = ["setosa", "versicolor", "virginica"]
labels, sizes = np.unique(df[target_feature], return_counts=True)

fig, ax = plt.subplots()
ax.pie(sizes, textprops={'color': "w", 'fontsize': '12'}, autopct=lambda pct: "{:.2f}%\n({:d})".format(pct, round(pct/100 * sum(sizes))))
ax.legend([str(i) + " (" + class_names[i] + ")" for i in labels])
ax.set_title(target_feature)
plt.show()

# %% [markdown]
# ## Split the dataset into train and test subsets

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# %% [markdown]
# ## Train a Logistic Regression classifier

# %%
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# %% [markdown]
# ## Evaluate the Logistic Regression classifier

# %%
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=[str(i) + " (" + class_names[i] + ")" for i in labels])
plt.grid(False)
plt.show()
