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
# # Wine Color

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/IBM_Introduction_to_Machine_Learning_Specialization/portfolio-3.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a Support Vector Machine (SVM) classifier to predict the color of wine based on its chemical properties.

# %% [markdown]
# ## Import libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import GridSearchCV

sns.set_style("whitegrid")

# %% [markdown]
# ## Load the dataset

# %%
file_url = "https://raw.githubusercontent.com/LuisAngelMendozaVelasco/IBM_Introduction_to_Machine_Learning_Specialization/refs/heads/main/Supervised_Machine_Learning-Classification/Week2/Labs/data/Wine_Quality_Data.csv"
df = pd.read_csv(file_url)
df.head()

# %% [markdown]
# ## Understand the dataset

# %% [markdown]
# We will be using the [wine quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality). This data set contains various chemical properties of wine, such as acidity, sugar, pH, and alcohol. It also contains a quality metric (3-9, with highest being better) and a color (red or white). 

# %%
df.info()

# %% [markdown]
# ## Visualize the dataset

# %%
number_features = df.shape[1] - 1
grid_rows = int(np.ceil(number_features / 3))
fig, axs = plt.subplots(grid_rows, 3, figsize=(15, 5 * grid_rows))

for ax, feature in zip(axs.flatten(), df.columns[:-1]):
    if len(df[feature].unique()) <= 10:
        sns.countplot(data=df, x=feature, ax=ax, hue="color")
        ax.set_xlabel("")
        ax.set_title(feature)
    else:
        sns.histplot(data=df, x=feature, ax=ax, hue="color")
        ax.set_xlabel("")
        ax.set_title(feature)

for ax in axs.flatten()[number_features:]:
    ax.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualize the class distribution

# %%
target_feature = "color"
labels, sizes = np.unique(df[target_feature], return_counts=True)

fig, ax = plt.subplots()
ax.pie(sizes, textprops={'color': "w", 'fontsize': '12'}, autopct=lambda pct: "{:.2f}%\n({:d})".format(pct, round(pct/100 * sum(sizes))))
ax.legend([i for i in labels])
ax.set_title(target_feature)
plt.show()

# %% [markdown]
# ## Preprocess the dataset

# %%
le = LabelEncoder()
df[target_feature] = le.fit_transform(df[target_feature])
df.head()

# %% [markdown]
# ## Split the dataset into train and test subsets

# %%
X = df.drop(columns=[target_feature])
y = df[target_feature]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# %% [markdown]
# ## Train a SVM classifier

# %%
C = [10**i for i in range(6)]
parameters = {'C': C}

svm_classifier = SVC()
clf = GridSearchCV(svm_classifier, parameters, scoring='recall')
clf.fit(X_train, y_train)

# %%
plt.figure()
plt.plot(C, clf.cv_results_['mean_test_score'])
plt.scatter(clf.best_params_['C'], clf.best_score_, color='red')
plt.xlabel("Regularization parameter (C)")
plt.ylabel("Mean recall score")
plt.xscale('log')
plt.show()

# %% [markdown]
# ## Evaluate the SVM classifier

# %%
y_pred = clf.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=le.classes_)
plt.grid(False)
plt.show()
