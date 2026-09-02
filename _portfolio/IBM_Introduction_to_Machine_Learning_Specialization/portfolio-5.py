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
# ## Housing Prices

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/IBM_Introduction_to_Machine_Learning_Specialization/portfolio-5.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a Decision Tree regressor to predict the sale price of houses based on their features.

# %% [markdown]
# ## Import libraries

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

sns.set_style("whitegrid")

# %% [markdown]
# ## Load the dataset

# %%
file_url = "https://raw.githubusercontent.com/LuisAngelMendozaVelasco/IBM_Introduction_to_Machine_Learning_Specialization/refs/heads/main/Supervised_Machine_Learning-Regression/Week2/Labs/data/Ames_Housing_Sales.csv"
df = pd.read_csv(file_url)
df.head()

# %% [markdown]
# ## Understand the dataset

# %% [markdown]
# We will be working with a dataset based on [housing prices in Ames, Iowa](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
#
# There are an extensive number of features:
#
# - **SalePrice** - the property's sale price in dollars. This is the target variable that you're trying to predict.
# - **MSSubClass**: The building class
# - **MSZoning**: The general zoning classification
# - **LotFrontage**: Linear feet of street connected to property
# - **LotArea**: Lot size in square feet
# - **Street**: Type of road access
# - **Alley**: Type of alley access
# - **LotShape**: General shape of property
# - **LandContour**: Flatness of the property
# - **Utilities**: Type of utilities available
# - **LotConfig**: Lot configuration
# - **LandSlope**: Slope of property
# - **Neighborhood**: Physical locations within Ames city limits
# - **Condition1**: Proximity to main road or railroad
# - **Condition2**: Proximity to main road or railroad (if a second is present)
# - **BldgType**: Type of dwelling
# - **HouseStyle**: Style of dwelling
# - **OverallQual**: Overall material and finish quality
# - **OverallCond**: Overall condition rating
# - **YearBuilt**: Original construction date
# - **YearRemodAdd**: Remodel date
# - **RoofStyle**: Type of roof
# - **RoofMatl**: Roof material
# - **Exterior1st**: Exterior covering on house
# - **Exterior2nd**: Exterior covering on house (if more than one material)
# - **MasVnrType**: Masonry veneer type
# - **MasVnrArea**: Masonry veneer area in square feet
# - **ExterQual**: Exterior material quality
# - **ExterCond**: Present condition of the material on the exterior
# - **Foundation**: Type of foundation
# - **BsmtQual**: Height of the basement
# - **BsmtCond**: General condition of the basement
# - **BsmtExposure**: Walkout or garden level basement walls
# - **BsmtFinType1**: Quality of basement finished area
# - **BsmtFinSF1**: Type 1 finished square feet
# - **BsmtFinType2**: Quality of second finished area (if present)
# - **BsmtFinSF2**: Type 2 finished square feet
# - **BsmtUnfSF**: Unfinished square feet of basement area
# - **TotalBsmtSF**: Total square feet of basement area
# - **Heating**: Type of heating
# - **HeatingQC**: Heating quality and condition
# - **CentralAir**: Central air conditioning
# - **Electrical**: Electrical system
# - **1stFlrSF**: First Floor square feet
# - **2ndFlrSF**: Second floor square feet
# - **LowQualFinSF**: Low quality finished square feet (all floors)
# - **GrLivArea**: Above grade (ground) living area square feet
# - **BsmtFullBath**: Basement full bathrooms
# - **BsmtHalfBath**: Basement half bathrooms
# - **FullBath**: Full bathrooms above grade
# - **HalfBath**: Half baths above grade
# - **Bedroom**: Number of bedrooms above basement level
# - **Kitchen**: Number of kitchens
# - **KitchenQual**: Kitchen quality
# - **TotRmsAbvGrd**: Total rooms above grade (does not include bathrooms)
# - **Functional**: Home functionality rating
# - **Fireplaces**: Number of fireplaces
# - **FireplaceQu**: Fireplace quality
# - **GarageType**: Garage location
# - **GarageYrBlt**: Year garage was built
# - **GarageFinish**: Interior finish of the garage
# - **GarageCars**: Size of garage in car capacity
# - **GarageArea**: Size of garage in square feet
# - **GarageQual**: Garage quality
# - **GarageCond**: Garage condition
# - **PavedDrive**: Paved driveway
# - **WoodDeckSF**: Wood deck area in square feet
# - **OpenPorchSF**: Open porch area in square feet
# - **EnclosedPorch**: Enclosed porch area in square feet
# - **3SsnPorch**: Three season porch area in square feet
# - **ScreenPorch**: Screen porch area in square feet
# - **PoolArea**: Pool area in square feet
# - **PoolQC**: Pool quality
# - **Fence**: Fence quality
# - **MiscFeature**: Miscellaneous feature not covered in other categories
# - **MiscVal**: $Value of miscellaneous feature
# - **MoSold**: Month Sold
# - **YrSold**: Year Sold
# - **SaleType**: Type of sale
# - **SaleCondition**: Condition of sale

# %%
df.info()

# %% [markdown]
# ## Preprocess the dataset

# %%
df.dropna(axis="columns", inplace=True)

preprocessed_df = pd.get_dummies(df, drop_first=True)
preprocessed_df.head()

# %% [markdown]
# ## Visualize the target feature

# %%
target_feature = "SalePrice"
correlations = {}

for feature in df.drop(columns=[target_feature]).columns:
    if df[feature].dtype != 'O' and df[feature].nunique() > 10:
        correlations[feature] = stats.pearsonr(df[feature], df[target_feature])[0]

max_correlation_feature = max(correlations, key=lambda key: abs(correlations[key]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

sns.histplot(x=target_feature, data=df, ax=ax1)
ax1.set_title("Distribution of Sale Price")

sns.scatterplot(x=df[max_correlation_feature], y=df[target_feature], ax=ax2)
ax2.set_title(f"Correlation coefficient = {correlations[max_correlation_feature]:.2f}")

plt.tight_layout()
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
# ## Train a Decision Tree regressor

# %%
max_depth = range(1, 11)
parameters = {'max_depth': max_depth}

decision_tree_regressor = DecisionTreeRegressor(random_state=100)
clf = GridSearchCV(decision_tree_regressor, parameters, scoring='r2')
clf.fit(X_train, y_train)

# %%
plt.figure()
plt.plot(max_depth, clf.cv_results_['mean_test_score'])
plt.scatter(clf.best_params_['max_depth'], clf.best_score_, color='red')
plt.xlabel("Max depth")
plt.ylabel("Mean $R^2$ score")
plt.show()

# %% [markdown]
# ## Evaluate the Decision Tree regressor

# %%
y_pred = clf.best_estimator_.predict(X_test)

print(f"MSE = {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² = {r2_score(y_test, y_pred):.2f}")

# %%
plt.figure()
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([0, y_test.max()], [0, y_test.max()], color='red', linestyle='--')
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.xlim(0, y_test.max()*1.05)
plt.ylim(0, y_test.max()*1.05)
plt.show()
