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
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/IBM_AI_Engineering/portfolio-4.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png"/>Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a Decision Tree classifier to predict whether a credit card transaction is fraudulent on an imbalanced dataset.

# %% [markdown]
# ## Import libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
sns.set_style("whitegrid")

# %% [markdown]
# ## Load the dataset

# %%
file_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"
df = pd.read_csv(file_url)
df.head()

# %% [markdown]
# ## Understand the dataset

# %% [markdown]
# We have access to transaction data from a financial institution that wants to know if a transaction is fraudulent or not. Most transactions are usually legitimate and only a small fraction are not, so the dataset is highly unbalanced.
#
# Each row in the dataset represents a credit card transaction. For confidentiality reasons, the original names of most features are anonymized **V1, V2 .. V28**. The values of these features are the result of a [Principal Component Analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) transformation and are numerical. The feature **Class** is the target variable and it takes two values: "1" in case of fraud and "0" otherwise.

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

# %% [markdown]
# As can be seen the dataset is highly unbalanced, so the target variable classes are not equally represented. This case requires special attention when training and evaluating the quality of a model.

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
# ## Preprocess the dataset

# %% [markdown]
# One way of handling this case at train time is to apply an over-sampling method to pay more attention to the samples in the minority class. [SMOTE (Synthetic Minority Over-sampling TEchnique)](https://arxiv.org/pdf/1106.1813) is a widely used oversampling technique in machine learning to address class imbalance issues in binary classification problems. It generates synthetic minority class samples to balance the class distribution, thereby reducing the impact of class imbalance on model performance. Data preprocessing such as scaling/normalization is typically useful for linear models to accelerate the training convergence.

# %%
def preprocess_data(df, smote=False):
    X = df.drop(["Time", "Class"], axis=1)
    y = df["Class"]
    
    if smote:
        sm = SMOTE(random_state=0)
        X, y = sm.fit_resample(X, y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# %% [markdown]
# ## Train a Decision Tree classifier without SMOTE

# %%
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# %%
classifier = DecisionTreeClassifier(max_depth=6, random_state=0)
classifier.fit(X_train, y_train)

# %% [markdown]
# ## Evaluate the Decision Tree classifier without SMOTE

# %%
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["0 (not fraud)", "1 (fraud)"])
plt.grid(False)
plt.show()

# %% [markdown]
# ## Train a Decision Tree classifier with SMOTE

# %%
X, y = preprocess_data(df, smote=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# %%
class_names = ["non-fraud", "fraud"]
labels, sizes = np.unique(y, return_counts=True)

fig, ax = plt.subplots()
ax.pie(sizes, textprops={'color': "w", 'fontsize': '12'}, autopct=lambda pct: "{:.2f}%\n({:d})".format(pct, round(pct/100 * sum(sizes))))
ax.legend([str(i) + " (" + class_names[i] + ")" for i in labels])
ax.set_title(target_feature)
plt.show()

# %%
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# %% [markdown]
# ## Evaluate the Decision Tree classifier with SMOTE

# %%
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["0 (not fraud)", "1 (fraud)"])
plt.grid(False)
plt.show()
