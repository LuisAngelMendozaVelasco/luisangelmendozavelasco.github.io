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
# # Human Activity Recognition

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/IBM_Introduction_to_Machine_Learning_Specialization/portfolio-1.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a logistic regression model to classify human activities based on smartphone sensor data.

# %% [markdown]
# ## Import libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

sns.set_style(style="whitegrid")

# %% [markdown]
# ## Load the dataset

# %%
file_url = "https://raw.githubusercontent.com/LuisAngelMendozaVelasco/IBM_Introduction_to_Machine_Learning_Specialization/refs/heads/main/Supervised_Machine_Learning-Classification/Week1/Labs/data/Human_Activity_Recognition_Using_Smartphones_Data.csv"
df = pd.read_csv(file_url)
df.head()

# %% [markdown]
# ## Understand the dataset

# %% [markdown]
# The [Human Activity Recognition Using Smartphones database](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) was created from recordings of study participants performing [activities of daily living (ADLs)](https://en.wikipedia.org/wiki/Activities_of_daily_living) while carrying a smartphone with built-in inertial sensors. The objective is to classify activities into one of the six activities performed (walking, walking upstairs, walking downstairs, sitting, standing, and laying).
#
# For each record in the dataset the following is provided:
#
# - Triaxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration.
# - Triaxial angular velocity from the gyroscope.
# - A 561-feature vector with time and frequency domain variables.
# - Its activity label.

# %%
df.info()

# %% [markdown]
# ## Visualize the class distribution

# %%
target_feature = df.columns[-1]
labels, sizes = np.unique(df[target_feature], return_counts=True)

fig, ax = plt.subplots()
ax.pie(sizes, textprops={'color': "w", 'fontsize': '12'}, autopct=lambda pct: "{:.2f}%\n({:d})".format(pct, round(pct/100 * sum(sizes))))
ax.legend([i for i in labels], bbox_to_anchor=(1, 1))
ax.set_title(target_feature)
plt.show()

# %% [markdown]
# ## Preprocess the dataset

# %%
le = LabelEncoder()
df['Activity'] = le.fit_transform(df['Activity'])
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
# ## Train a Logistic Regression classifier

# %%
classifier = LogisticRegression(C=10, max_iter=1000)
classifier.fit(X_train, y_train)

# %% [markdown]
# ## Evaluate the Logistic Regression classifier

# %%
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.grid(False)
plt.show()
