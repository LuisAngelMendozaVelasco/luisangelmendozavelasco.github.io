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
# # Price of Houses

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/IBM_AI_Engineering/portfolio-5.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png"/>Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a Decision Tree regressor to predict the median price of houses in various areas of Boston.

# %% [markdown]
# A [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree_learning#Decision_tree_types) regressor is a type of [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) algorithm used for regression tasks, where the goal is to predict a continuous value or a numerical output. It is a variant of the Decision Tree algorithm, which is commonly used for classification tasks.

# %% [markdown]
# ## Import libaries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
sns.set_style("whitegrid")

# %% [markdown]
# ## Load the dataset

# %%
file_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/real_estate_data.csv"
df = pd.read_csv(file_url)
df.head()

# %% [markdown]
# ## Understand the dataset

# %% [markdown]
# The dataset contains information on areas/towns not individual houses, the features are:
#
# - **CRIM**: Crime per capita
# - **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft.
# - **INDUS**: Proportion of non-retail business acres per town
# - **CHAS**: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# - **NOX**: Nitric oxides concentration (parts per 10 million)
# - **RM**: Average number of rooms per dwelling
# - **AGE**: Proportion of owner-occupied units built prior to 1940
# - **DIS**: Weighted distances to ﬁve Boston employment centers
# - **RAD**: Index of accessibility to radial highways
# - **TAX**: Full-value property-tax rate per $10,000
# - **PTRATIO**: Pupil-teacher ratio by town
# - **LSTAT**: Percent lower status of the population
# - **MEDV**: Median value of owner-occupied homes in $1000s

# %%
df.info()

# %% [markdown]
# ## Visualize the dataset

# %%
number_features = df.shape[1] - 1
grid_rows = int(np.ceil(number_features / 3))
fig, axs = plt.subplots(grid_rows, 3, figsize=(15, 5 * grid_rows))

for ax, feature in zip(axs.flatten(), df.columns):
    if len(df[feature].unique()) <= 10:
        sns.countplot(data=df, x=feature, hue=feature, ax=ax, palette="tab10", legend=False)
        ax.set_xlabel("")
        ax.set_title(feature)
    else:
        sns.histplot(data=df, x=feature, ax=ax, bins="doane")
        ax.set_xlabel("")
        ax.set_title(feature)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Preprocess the dataset

# %% [markdown]
# Drop rows with missing values:

# %%
df.dropna(inplace=True)
df.info()

# %%
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# %% [markdown]
# ## Visualize the target feature

# %% [markdown]
# Plot the distribution of the target feature and the relationship between the target feature and the feature with the highest [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).

# %%
correlations = {}

for feature in X.columns:
    correlations[feature] = stats.pearsonr(X[feature], y)[0]

max_correlation_feature = max(correlations, key=lambda key: abs(correlations[key]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

sns.histplot(x="MEDV", data=df, ax=ax1)
ax1.set_title("Distribution of MEDV")

sns.scatterplot(x=X[max_correlation_feature], y=y, ax=ax2)
ax2.set_title(f"Correlation coefficient = {correlations[max_correlation_feature]:.2f}")

plt.show()

# %% [markdown]
# ## Split the dataset into train and test subsets

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# %% [markdown]
# ## Train a Decision Tree regressor

# %%
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

# %% [markdown]
# ## Evaluate the Decision Tree regressor

# %%
y_pred = regressor.predict(X_test)

print(f"MSE = {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² = {r2_score(y_test, y_pred):.2f}")
