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
# # Tumor Classifier

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/IBM_AI_Engineering/portfolio-8.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png"/>Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a Support Vector Machine (SVM) model to classify human cell samples as benign or malignant.

# %% [markdown]
# [Support Vector Machine (SMV)](https://en.wikipedia.org/wiki/Support_vector_machine) is a type of supervised learning algorithm used for classification and regression tasks. It’s a powerful and widely used machine learning technique, particularly effective in handling high-dimensional spaces and non-linear relationships between features.

# %% [markdown]
# ## Import libraries

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# %% [markdown]
# ## Load the dataset

# %%
file_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/cell_samples.csv"
df = pd.read_csv(file_url)
df.head()

# %% [markdown]
# ## Understand the dataset

# %% [markdown]
# The dataset consists of several hundred records of human cell samples, each containing the values of a set of cellular features. The `Class` field contains the diagnosis, the samples are benign (value = 2) or malignant (value = 4).
#
# |  Field name |         Description         |
# |:-----------:|:---------------------------:|
# | ID          | Patient identifier          |
# | Clump       | Clump thickness             |
# | UnifSize    | Uniformity of cell size     |
# | UnifShape   | Uniformity of cell shape    |
# | MargAdh     | Marginal adhesion           |
# | SingEpiSize | Single epithelial cell size |
# | BareNuc     | Bare nuclei                 |
# | BlandChrom  | Bland chromatin             |
# | NormNucl    | Normal nucleoli             |
# | Mit         | Mitoses                     |
# | Class       | Benign or malignant         |

# %%
df.info()

# %% [markdown]
# ## Preprocess the dataset

# %% [markdown]
# Drop rows that contain **NaN** values and the `ID` field:

# %%
original_size = df.size
df = df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()].drop("ID", axis=1)
df['BareNuc'] = df['BareNuc'].astype('int')
print("The dataset was reduced by {:.2f}%".format((1 - df.size / original_size) * 100))

# %% [markdown]
# ## Visualize the dataset

# %%
number_features = len(df.columns) - 1
grid_rows = int(np.ceil(number_features / 3))
fig, axs = plt.subplots(grid_rows, 3, figsize=(15, 5 * grid_rows))

for ax, feature in zip(axs.flatten(), df.columns[:-1]):
    if len(df[feature].unique()) <= 10:
        sns.countplot(data=df, x=feature, hue="Class", ax=ax, palette="tab10")
        ax.set_xlabel("")
        ax.set_title(feature)
    else:
        sns.histplot(data=df, x=feature, hue="Class", ax=ax)
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
class_names = ["benign", "malignant"]
labels, sizes = np.unique(df[target_feature], return_counts=True)

fig, ax = plt.subplots()
ax.pie(sizes, textprops={'color': "w", 'fontsize': '12'}, autopct=lambda pct: "{:.2f}%\n({:d})".format(pct, round(pct/100 * sum(sizes))))
ax.legend([str(j) + " (" + class_names[i] + ")" for i, j in enumerate(labels)])
ax.set_title(target_feature)
plt.show()

# %% [markdown]
# ## Split the dataset into train and test subsets

# %%
X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# %% [markdown]
# ## Train a SVM model

# %%
classifier = SVC()
classifier.fit(X_train, y_train) 

# %% [markdown]
# ## Evaluate the model

# %%
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=[str(j) + " (" + class_names[i] + ")" for i, j in enumerate(labels)])
plt.grid(False)
plt.show()
