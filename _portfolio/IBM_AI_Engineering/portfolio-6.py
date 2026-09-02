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
# # Taxi Tip

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/IBM_AI_Engineering/portfolio-6.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png"/>Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a Decision Tree regressor to predict the amount of a taxi tip.

# %% [markdown]
# ## Import libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
sns.set_style("whitegrid")

# %% [markdown]
# ## Load the dataset

# %%
file_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/yellow_tripdata_2019-06.csv'
df = pd.read_csv(file_url)
df.head()

# %% [markdown]
# ## Understand the dataset

# %% [markdown]
# The data was collected and provided by the NYC Taxi and Limousine Commission (TLC). The dataset records include fields capturing pick-up and drop-off dates/times, pick-up and drop-off locations, trip distances, itemized fares, rate types, payment types, driver-reported passenger counts, and tip amount.
#
# Each row in the dataset represents a taxi trip taken in June 2019, the variable **tip_amount** represents the target variable.

# %%
df.info()

# %% [markdown]
# ## Preprocess the dataset

# %%
original_size = df.size

# Some trips report $0 tip, so it is assumed that these tips were paid in cash. We drop all these rows.
df = df[df['tip_amount'] > 0]

# We also remove some outliers, namely those where the tip was larger than the fare cost.
df = df[(df['tip_amount'] <= df['fare_amount'])]

# Convert 'tpep_dropoff_datetime' and 'tpep_pickup_datetime' columns to datetime objects.
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

# Extract dropoff hour.
df['dropoff_hour'] = df['tpep_dropoff_datetime'].dt.hour

# Extract dropoff day of the week (0 = Monday, 6 = Sunday).
df['dropoff_day'] = df['tpep_dropoff_datetime'].dt.weekday

# Calculate trip time in seconds.
df['trip_time'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()

# Drop unnecessary variables.
df = df.drop(['total_amount', 'VendorID', 'RatecodeID', 'store_and_fwd_flag', 'PULocationID', 'DOLocationID', 'mta_tax', 
              'improvement_surcharge', 'congestion_surcharge', 'tpep_pickup_datetime', 'tpep_dropoff_datetime'], axis=1)

print("The dataset was reduced by {:.2f}%".format((1 - df.size / original_size) * 100))

# %%
df.info()

# %% [markdown]
# ## Visualize the dataset

# %%
columns = list(df.columns)
columns.remove("tip_amount")
number_features = len(columns)
grid_rows = int(np.ceil(number_features / 3))
fig, axs = plt.subplots(grid_rows, 3, figsize=(15, 5 * grid_rows))

for ax, feature in zip(axs.flatten(), columns):
    if len(df[feature].unique()) <= 10:
        sns.countplot(data=df, x=feature, hue=feature, ax=ax, palette="tab10", legend=False)
        ax.set_xlabel("")
        ax.set_yscale("log")
        ax.set_title(feature)
    else:
        sns.histplot(data=df, x=feature, ax=ax, bins="doane")
        ax.set_xlabel("")
        ax.set_yscale("log")
        ax.set_title(feature)

for ax in axs.flatten()[number_features:]:
    ax.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualize the target feature

# %% [markdown]
# Plot the distribution of the target feature and the relationship between the target feature and the feature with the highest [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).

# %%
correlations = {}

for feature in columns:
    correlations[feature] = stats.pearsonr(df[feature], df["tip_amount"])[0]

max_correlation_feature = max(correlations, key=lambda key: abs(correlations[key]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

sns.histplot(x="tip_amount", data=df, ax=ax1, bins="doane")
ax1.set_yscale("log")
ax1.set_title("Distribution of tip_amount")

sns.scatterplot(x=df[max_correlation_feature], y=df["tip_amount"], ax=ax2)
ax2.set_title(f"Correlation coefficient = {correlations[max_correlation_feature]:.2f}")

plt.show()

# %% [markdown]
# ## Convert categorical variables into dummy/indicator variables

# %%
categorical_columns = ["payment_type", "dropoff_hour", "dropoff_day"]
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
df.head()

# %% [markdown]
# ## Split the dataset into train and test subsets

# %%
X = df.drop("tip_amount", axis=1)
y = df["tip_amount"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# %% [markdown]
# ## Train a Decision Tree regressor

# %%
regressor = DecisionTreeRegressor(max_depth=5)
regressor.fit(X_train, y_train)

# %% [markdown]
# ## Evaluate the Decision Tree regressor

# %%
y_pred = regressor.predict(X_test)

print(f"MSE = {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² = {r2_score(y_test, y_pred):.2f}")
