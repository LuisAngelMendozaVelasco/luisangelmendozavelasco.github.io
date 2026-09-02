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
# # Heart Failure Prediction

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/Machine_Learning_Specialization/portfolio-3.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a XGBoost classifier to predict potential heart disease.

# %% [markdown]
# [XGBoost](https://en.wikipedia.org/wiki/XGBoost) is an open-source software library that provides a regularizing Gradient Boosting framework. It is an advanced implementation of Gradient Boosting, which is an ensemble technique using weak learners. XGBoost improves mistakes from previous trees by sequentially adding models to correct errors, making it a boosting algorithm. It is designed for efficiency, speed, and high performance, and it implements machine learning algorithms under the Gradient Boosting framework.

# %% [markdown]
# ## Import libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from xgboost import XGBClassifier
import kagglehub

sns.set_style("whitegrid")

# %% [markdown]
# ## Download the dataset

# %% [markdown]
# Download the dataset from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

# %%
kagglehub.dataset_download("fedesoriano/heart-failure-prediction", output_dir="/tmp/heart-failure-prediction")

# %% [markdown]
# ## Load the dataset

# %%
df = pd.read_csv('/tmp/heart-failure-prediction/heart.csv')
df.head()

# %% [markdown]
# ## Understand the dataset

# %% [markdown]
# Cardiovascular disease (CVD) is the number one cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of five CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs.
#
# People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management.
#
# This dataset contains 11 features that can be used to predict potential heart disease:
#
# - **Age**: age of the patient [years]
# - **Sex**: sex of the patient [M: Male, F: Female]
# - **ChestPainType**: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
# - **RestingBP**: resting blood pressure [mm Hg]
# - **Cholesterol**: serum cholesterol [mm/dl]
# - **FastingBS**: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
# - **RestingECG**: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
# - **MaxHR**: maximum heart rate achieved [Numeric value between 60 and 202]
# - **ExerciseAngina**: exercise-induced angina [Y: Yes, N: No]
# - **Oldpeak**: oldpeak = ST [Numeric value measured in depression]
# - **ST_Slope**: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
# - **HeartDisease**: output class [1: Heart Disease, 0: Normal]

# %%
df.info()

# %% [markdown]
# ## Visualize the dataset

# %%
number_features = df.shape[1] - 1
grid_rows = int(np.ceil(number_features / 3))
fig, axs = plt.subplots(grid_rows, 3, figsize=(15, 5 * grid_rows))

for ax, feature in zip(axs.flatten(), df.columns[:-1]):
    if df[feature].dtype == 'O':
        sns.countplot(data=df, x=feature, hue="HeartDisease", ax=ax)
        ax.set_xlabel("")
        ax.set_title(feature)
    elif len(df[feature].unique()) <= 10:
        sns.countplot(data=df, x=feature, hue="HeartDisease", ax=ax)
        ax.set_xlabel("")
        ax.set_title(feature)
    else:
        sns.histplot(data=df, x=feature, hue="HeartDisease", ax=ax)
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

fig, ax = plt.subplots()
ax.pie(sizes, textprops={'color': "w", 'fontsize': '12'}, autopct=lambda pct: "{:.2f}%\n({:d})".format(pct, round(pct/100 * sum(sizes))))
ax.legend([str(i) + " (normal)" if i == 0 else str(i) + " (heart disease)" for i in labels])
ax.set_title(target_feature)
plt.show()

# %% [markdown]
# ## Preprocess the dataset

# %%
categorical_features = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(data=df, columns=categorical_features, drop_first=True)
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
# ## Train a XGBoost classifier

# %%
xgb_classifier = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=5)
xgb_classifier.fit(X_train, y_train)

# %% [markdown]
# ## Evaluate the XGBoost classifier

# %%
y_pred = xgb_classifier.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["0 (normal)", "1 (heart disease)"])
plt.grid(False)
plt.show()
