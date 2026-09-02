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
# # Sale Price

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/IBM_Introduction_to_Machine_Learning_Specialization/portfolio-4.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a Linear Regression model to predict the sale price of cars based on various features.

# %% [markdown]
# ## Import libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

sns.set_style("whitegrid")

# %% [markdown]
# ## Load the dataset

# %%
file_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv"
df = pd.read_csv(file_url)
df.head()

# %% [markdown]
# ## Understand the dataset

# %% [markdown]
# We will be using the car sales dataset, that can also be found and downloaded from [kaggle](https://www.kaggle.com/datasets/goyalshalini93/car-data?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML240ENSkillsNetwork34171862-2022-01-01). The dataset contains all the information about cars, a name of a manufacturer, all car's technical parameters and a sale price of a car. We have 205 entries or rows, as well as 26 features. The **price** is our target, or response variable, and the rest of the features are our predictor variables.

# %%
df.info()

# %% [markdown]
# ## Preprocess the dataset

# %%
df['brand'] = df["CarName"].str.split(' ').str.get(0).str.lower()
df['brand'] = df['brand'].replace(['vw', 'vokswagen'], 'volkswagen')
df['brand'] = df['brand'].replace(['maxda'], 'mazda')
df['brand'] = df['brand'].replace(['porcshce'], 'porsche')
df['brand'] = df['brand'].replace(['toyouta'], 'toyota')
df.drop(columns=['car_ID', 'CarName'], inplace=True)

preprocessed_df = pd.get_dummies(df, drop_first=True)
preprocessed_df.head()

# %% [markdown]
# ## Visualize the dataset

# %%
number_features = df.shape[1] - 1
grid_rows = int(np.ceil(number_features / 3))
fig, axs = plt.subplots(grid_rows, 3, figsize=(15, 5 * grid_rows))

for ax, feature in zip(axs.flatten(), df.drop(columns=["price"]).columns):
    if df[feature].dtype == 'O':
        sns.countplot(data=df, x=feature, ax=ax, hue=feature, palette="tab10")
        ax.set_xlabel("")
        ax.set_title(feature)
        if feature == 'brand':
            unique_values = df[feature].unique()
            ax.set_xticks(range(len(unique_values)), rotation='vertical', labels=unique_values)
    elif len(df[feature].unique()) <= 10:
        sns.countplot(data=df, x=feature, ax=ax, hue=feature, palette="tab10", legend=False)
        ax.set_xlabel("")
        ax.set_title(feature)
    else:
        sns.histplot(data=df, x=feature, ax=ax)
        ax.set_xlabel("")
        ax.set_title(feature)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualize the target feature

# %%
target_feature = "price"
correlations = {}

for feature in df.drop(columns=[target_feature]).columns:
    if df[feature].dtype != 'O':
        correlations[feature] = stats.pearsonr(df[feature], df[target_feature])[0]

max_correlation_feature = max(correlations, key=lambda key: abs(correlations[key]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

sns.histplot(x=target_feature, data=df, ax=ax1)
ax1.set_title("Distribution of Price")

sns.scatterplot(x=df[max_correlation_feature], y=df[target_feature], ax=ax2)
ax2.set_title(f"Correlation coefficient = {correlations[max_correlation_feature]:.2f}")

plt.show()

# %% [markdown]
# ## Split the dataset into train and test subsets

# %%
X = preprocessed_df.drop(columns=[target_feature])
y = preprocessed_df[target_feature]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# %% [markdown]
# ## Train a Linear Regression model

# %%
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# %% [markdown]
# ## Evaluate the Linear Regression model

# %%
y_pred = regressor.predict(X_test)

print(f"MSE = {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² = {r2_score(y_test, y_pred):.2f}")

# %%
plt.figure()
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([0, y_test.max()], [0, y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.xlim(0, y_test.max()*1.05)
plt.ylim(0, y_test.max()*1.05)
plt.show()
