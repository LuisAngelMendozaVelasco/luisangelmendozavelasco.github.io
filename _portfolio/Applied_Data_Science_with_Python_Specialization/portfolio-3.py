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
# # Credit Card Fraud Detection

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/Applied_Data_Science_with_Python_Specialization/portfolio-3.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png"/>Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train multiple classifiers and evaluate their effectiveness in predicting credit card fraud on an imbalanced dataset.

# %% [markdown]
# ## Import libraries

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, precision_recall_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# %% [markdown]
# ## Load the dataset

# %%
file_url = "https://raw.githubusercontent.com/LuisAngelMendozaVelasco/Applied_Data_Science_with_Python_Specialization/main/Applied_Machine_Learning_in_Python/Week3/Labs/data/fraud_data.csv.gz"
df = pd.read_csv(file_url)
df.head()

# %% [markdown]
# ## Understand the dataset

# %% [markdown]
#
# Each row in the dataframe corresponds to a credit card transaction. Features include confidential variables `V1` through `V28` as well as `Amount` which is the amount of the transaction. The target is stored in the `Class` column, where a value of 1 corresponds to an instance of fraud and 0 corresponds to an instance of not fraud.

# %%
df.info()

# %% [markdown]
# ## Analyze the `Amount` feature

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

sns.histplot(data=df, x="Amount", hue="Class", bins="doane", ax=axs[0])
axs[0].set_yscale('log')
axs[0].set_title("Histogram")

sns.boxplot(data=df, x="Class", y="Amount", hue="Class", ax=axs[1], legend=False)
axs[1].set_yscale('log')
axs[1].set_title("Box plot")

plt.show()

# %% [markdown]
# ## Visualize the class distribution

# %%
target_feature = df.columns[-1]
labels, sizes = np.unique(df[target_feature], return_counts=True)
labels = ["1 (fraud)" if i else "0 (non-fraud)" for i in labels]

plt.figure()
g = sns.barplot(x=labels, y=sizes)
g.bar_label(g.containers[0], [str(round(100 * size / sum(sizes), 2)) + "%" + " (" + str(size) + ")" for size in sizes])
plt.yscale("log")
plt.title(target_feature)
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
# ## Find the optimal regularization parameter value for a SVM model

# %%
C = np.logspace(0, 6, 3)
parameters = {'C': C}

svc = SVC()
clf_svc = GridSearchCV(svc, parameters, scoring='recall')
clf_svc.fit(X_train, y_train)

# %%
plt.figure()
plt.plot(C, clf_svc.cv_results_['mean_test_score'])
plt.scatter(clf_svc.best_params_['C'], clf_svc.best_score_, color='red')
plt.xlabel("C")
plt.ylabel("Mean recall score")
plt.xscale("log")
plt.xticks(C, C)
plt.show()

# %% [markdown]
# ## Evaluate the best SVM model

# %%
y_pred = clf_svc.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["0 (non-fraud)", "1 (fraud)"])
plt.grid(False)
plt.show()

# %% [markdown]
# ## Find the optimal regularization parameter for a Logistic Regression classifier

# %% [markdown]
# A [Logistic Regression classifier](https://en.wikipedia.org/wiki/Logistic_regression) uses a logistic (or sigmoid) function to transform a linear combination of input features into a probability value ranging between 0 and 1. This probability indicates the likelihood that a given input corresponds to one of two predefined categories.

# %%
C = np.logspace(-3, 3, 3)
parameters = {'C': C}

logistic_regression_classifier = LogisticRegression(max_iter=1000)
clf_lr = GridSearchCV(logistic_regression_classifier, parameters, scoring='recall')
clf_lr.fit(X_train, y_train)

# %%
plt.figure()
plt.plot(C, clf_lr.cv_results_['mean_test_score'])
plt.scatter(clf_lr.best_params_['C'], clf_lr.best_score_, color='red')
plt.xlabel("C")
plt.ylabel("Mean recall score")
plt.xscale("log")
plt.xticks(C, C)
plt.show()

# %% [markdown]
# ## Evaluate the best Logistic Regression classifier

# %%
y_pred = clf_lr.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["0 (non-fraud)", "1 (fraud)"])
plt.grid(False)
plt.show()

# %% [markdown]
# Generally, the Precision-Recall Curve is used when there is a moderate to large class imbalance.

# %%
y_proba = clf_lr.best_estimator_.predict_proba(X_test)[:, -1]
precision, recall, _ = precision_recall_curve(y_test, y_proba)

plt.figure()
plt.plot(precision, recall)
plt.title('Precision - Recall Curve')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.show()
