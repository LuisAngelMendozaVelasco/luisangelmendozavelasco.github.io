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
# # Customer Churn

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/IBM_Introduction_to_Machine_Learning_Specialization/portfolio-2.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a k-Nearest Neighbors (k-NN) classifier to predict customer churn.

# %% [markdown]
# ## Import libraries

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

sns.set_style(style="whitegrid")

# %% [markdown]
# ## Load the dataset

# %%
file_url = "https://raw.githubusercontent.com/LuisAngelMendozaVelasco/IBM_Introduction_to_Machine_Learning_Specialization/refs/heads/main/Supervised_Machine_Learning-Classification/Week2/Labs/data/churndata.pkl"
df = pd.read_pickle(file_url)
df.head()

# %% [markdown]
# ## Understand the dataset

# %% [markdown]
# We use a customer churn dataset from a fictional telecom firm which includes customer data, usage of long-distance, data usage, monthly revenue, type of offerings, and other services purchased by customers. The data include a mix of numeric, categorical, and ordinal variables that will be used to predict customer churn rates.

# %%
df.info()

# %% [markdown]
# ## Visualize the dataset

# %%
number_features = df.shape[1] - 2
grid_rows = int(np.ceil(number_features / 3))
fig, axs = plt.subplots(grid_rows, 3, figsize=(15, 5 * grid_rows))

for ax, feature in zip(axs.flatten(), df.drop(columns=["id", "churn_value"]).columns):
    if df[feature].dtype == 'O':
        sns.countplot(data=df, x=feature, hue="churn_value", ax=ax)
        ax.set_xlabel("")
        ax.set_title(feature)
    elif len(df[feature].unique()) <= 10:
        sns.countplot(data=df, x=feature, hue="churn_value", ax=ax)
        ax.set_xlabel("")
        ax.set_title(feature)
    else:
        sns.histplot(data=df, x=feature, hue="churn_value", ax=ax)
        ax.set_xlabel("")
        ax.set_title(feature)

for ax in axs.flatten()[number_features:]:
    ax.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualize the class distribution

# %%
target_feature = "churn_value"
labels, sizes = np.unique(df[target_feature], return_counts=True)

fig, ax = plt.subplots()
ax.pie(sizes, textprops={'color': "w", 'fontsize': '12'}, autopct=lambda pct: "{:.2f}%\n({:d})".format(pct, round(pct/100 * sum(sizes))))
ax.legend([i for i in labels])
ax.set_title(target_feature)
plt.show()

# %% [markdown]
# ## Preprocess the dataset

# %%
preprocessed_df = df.drop(columns=["id"])
scaler = StandardScaler()

for feature in preprocessed_df.drop(columns=["churn_value", "satisfaction"]).columns:
    if preprocessed_df[feature].dtype in ["int64", "float64"]:
        preprocessed_df[feature] = scaler.fit_transform(preprocessed_df[feature].to_numpy().reshape(-1, 1))

preprocessed_df = pd.get_dummies(preprocessed_df, drop_first=True)
preprocessed_df.head()

# %% [markdown]
# ## Split the dataset into train and test subsets

# %%
X = preprocessed_df.drop(columns=[target_feature])
y = preprocessed_df[target_feature]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# %% [markdown]
# ## Train a k-NN classifier

# %%
n_neighbors = range(2, 52, 2)
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
plt.show()

# %% [markdown]
# ## Evaluate the k-NN classifier

# %%
y_pred = clf.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.grid(False)
plt.show()
