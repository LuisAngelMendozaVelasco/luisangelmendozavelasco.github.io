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
# # CO2 Emissions

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/IBM_AI_Engineering/portfolio-1.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train Simple and Multiple Linear Regression models to predict CO2 emissions from light-duty vehicles.

# %% [markdown]
# [Simple Linear Regression](https://en.wikipedia.org/wiki/Simple_linear_regression) is a statistical approach that models the relationship between a dependent variable and a single independent variable. The goal is to establish a linear relationship between the two variables, where the dependent variable is predicted based on the value of the independent variable.
#
# Multiple Linear Regression is an extension of Simple Linear Regression, where the goal is to model the relationship between a dependent variable and multiple independent variables. In this case, the model attempts to explain the variation in the dependent variable using a combination of multiple independent variables.

# %% [markdown]
# ## Import libraries

# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
sns.set_style("whitegrid")

# %% [markdown]
# ## Load the dataset

# %%
file_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df = pd.read_csv(file_url)
df.head()

# %% [markdown]
# ## Understand the dataset

# %% [markdown]
# The fuel consumption dataset contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for light-duty vehicles.
#
# - MODELYEAR e.g. **2014**
# - MAKE e.g. **Acura**
# - MODEL e.g. **ILX**
# - VEHICLE CLASS e.g. **SUV**
# - ENGINE SIZE e.g. **4.7**
# - CYLINDERS e.g **6**
# - TRANSMISSION e.g. **A6**
# - FUEL CONSUMPTION in CITY(L/100 km) e.g. **9.9**
# - FUEL CONSUMPTION in HWY (L/100 km) e.g. **8.9**
# - FUEL CONSUMPTION COMB (L/100 km) e.g. **9.2**
# - CO2 EMISSIONS (g/km) e.g. **182**

# %%
df.info()

# %% [markdown]
# ## Visualize some features of the dataset

# %%
features = []

for feature in df.columns[1:]:
    if df[feature].dtype == 'O':
        if len(df[feature].unique()) <= 10:
            features.append(feature)
    else:
        features.append(feature)

number_features = len(features)
grid_rows = int(np.ceil(number_features / 3))
fig, axs = plt.subplots(grid_rows, 3, figsize=(15, 5 * grid_rows))

for ax, feature in zip(axs.flatten(), features):
    if len(df[feature].unique()) <= 10:
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
# ## Visualize the relationship between the target and the feature with the highest correlation coefficient

# %%
features = ["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_CITY", "FUELCONSUMPTION_HWY", "FUELCONSUMPTION_COMB"]
correlations = {}

for feature in features:
    correlations[feature] = stats.pearsonr(df[feature], df["CO2EMISSIONS"])[0]

max_correlation_feature = max(correlations, key=correlations.get)

plt.figure()
sns.scatterplot(df, x=max_correlation_feature, y="CO2EMISSIONS")
plt.title(f"Correlation coefficient = {correlations[max_correlation_feature]:.2f}")
plt.show()

# %% [markdown]
# ## Preprocess the dataset

# %%
df_subset = df[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_CITY", "FUELCONSUMPTION_HWY", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]]
df_subset.head()

# %% [markdown]
# ## Split the dataset into train and test subsets

# %%
X = df_subset.drop("CO2EMISSIONS", axis=1)
y = df_subset["CO2EMISSIONS"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# %% [markdown]
# ## Train a Simple Linear Regression model

# %% [markdown]
# Use the `FUELCONSUMPTION_COMB` feature as training data.

# %%
simple_linear_regressor = LinearRegression()
simple_linear_regressor.fit(X_train["FUELCONSUMPTION_CITY"].to_frame(), y_train)

# %% [markdown]
# ## Evaluate the Simple Linear Regression model

# %% [markdown]
# The [Mean Square Error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error) is a measure used to quantify the average squared difference between the predicted values and the actual values in a dataset.
#
# The [coefficient of determination (R²)](https://en.wikipedia.org/wiki/Coefficient_of_determination) provides an indication of goodness of fit and therefore a measure of how well unseen samples are likely to be predicted by the model, through the proportion of explained variance. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).

# %%
y_pred = simple_linear_regressor.predict(X_test["FUELCONSUMPTION_CITY"].to_frame())

print(f"MSE = {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² = {r2_score(y_test, y_pred):.2f}")

plt.figure()
sns.scatterplot(x=X_test["FUELCONSUMPTION_CITY"], y=y_test)
sns.scatterplot(x=X_test["FUELCONSUMPTION_CITY"], y=y_pred)
plt.legend(["y_test", "y_pred"])
plt.show()

# %% [markdown]
# ## Train a Multiple Linear Regression model

# %%
multiple_linear_regressor = LinearRegression()
multiple_linear_regressor.fit(X_train, y_train)

# %% [markdown]
# ## Evaluate the Multiple Linear Regression model

# %%
y_pred = multiple_linear_regressor.predict(X_test)

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
