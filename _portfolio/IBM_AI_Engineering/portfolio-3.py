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
# # Medical Treatment

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/IBM_AI_Engineering/portfolio-3.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png"/>Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a Decision Tree classifier to predict the appropriate drug for a patient based on their medical condition.

# %% [markdown]
#
# A [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree_learning) classifier is a type of [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) algorithm used for classification tasks in machine learning. It is a non-parametric method that creates a tree-like model of decisions, where each internal node represents a feature or attribute, and each leaf node represents a class label or predicted outcome.

# %% [markdown]
# ## Import libaries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
sns.set_style("whitegrid")

# %% [markdown]
# ## Load the dataset

# %%
file_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
df = pd.read_csv(file_url)
df.head()

# %% [markdown]
# ## Understand the dataset

# %% [markdown]
# A medical researcher has collected data about a set of patients, all of whom suffered from the same illness. During their course of treatment, each patient responded to one of five medications, Drug **A**, **B**, **C**, **X** and **Y**. The features of this dataset are **Age**, **Sex**, **Blood Pressure**, and **Cholesterol** of the patients. The target is the drug that each patient responded to.

# %%
df.info()

# %% [markdown]
# ## Visualize the dataset

# %%
number_features = df.shape[1] - 1
grid_rows = int(np.ceil(number_features / 3))
fig, axs = plt.subplots(grid_rows, 3, figsize=(15, 5 * grid_rows))

for ax, feature in zip(axs.flatten(), df.columns[:number_features]):
    if df[feature].dtype == 'O':
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
# ## Visualize the class distribution

# %%
target_feature = df.columns[-1]
labels, sizes = np.unique(df[target_feature], return_counts=True)

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(sizes, textprops={'color': "w", 'fontsize': '12'}, autopct=lambda pct: "{:.2f}%\n({:d})".format(pct, round(pct/100 * sum(sizes))))
ax.legend(labels)
ax.set_title(target_feature)
plt.show()

# %% [markdown]
# ## Preprocess the dataset

# %% [markdown]
# Some features in this dataset are categorical. [Sklearn Decision Tree classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) does not handle categorical variables, so we need to convert these features to numerical values.

# %%
X = df.drop(target_feature, axis=1)
y = df[target_feature]

for feature in X.columns:
    if X[feature].dtype == 'O':
        encoder = LabelEncoder()
        X[feature] = encoder.fit_transform(X[feature])

X.head()

# %% [markdown]
# ## Split the dataset into training and test sets

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# %% [markdown]
# ## Train a Decision Tree classifier

# %%
classifier = DecisionTreeClassifier(criterion="entropy", max_depth=4)
classifier.fit(X_train, y_train)

# %% [markdown]
# ## Evaluate the Decision Tree classifier

# %%
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.grid(False)
plt.show()

# %% [markdown]
# ## Visualize the Decision Tree classifier

# %%
plt.figure(figsize=(9, 6))
plot_tree(classifier, filled=True, feature_names=X.columns)
plt.show()
