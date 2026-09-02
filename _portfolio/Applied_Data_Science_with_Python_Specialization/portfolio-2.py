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
# # Poisonous Mushrooms

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/Applied_Data_Science_with_Python_Specialization/portfolio-2.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png"/>Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a Support Vector Machine (SVM) model to predict whether or not a mushroom is poisonous.

# %% [markdown]
# ## Import libaries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import seaborn as sns
sns.set_style('whitegrid')

# %% [markdown]
# ## Load the dataset

# %%
file_url = 'https://raw.githubusercontent.com/LuisAngelMendozaVelasco/Applied_Data_Science_with_Python_Specialization/main/Applied_Machine_Learning_in_Python/Week2/Labs/data/mushrooms.csv'
df = pd.read_csv(file_url)
df.head()

# %% [markdown]
# ## Understand the dataset

# %% [markdown]
# The [UCI Mushroom Data Set](http://archive.ics.uci.edu/dataset/73/mushroom) includes descriptions of samples corresponding to 23 species of gilled mushrooms. Each species is identified as definitely edible or definitely poisonous.

# %% [markdown]
# |       Variable Name      |                                             Description                                             |
# |:------------------------:|:---------------------------------------------------------------------------------------------------:|
# | class                    | poisonous=p, edible=e                                                                               |
# | cap-shape                | bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s                                                |
# | cap-surface              | fibrous=f,grooves=g,scaly=y,smooth=s                                                                |
# | cap-color                | brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y                    |
# | bruises                  | bruises=t,no=f                                                                                      |
# | odor                     | almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s                        |
# | gill-attachment          | attached=a,descending=d,free=f,notched=n                                                            |
# | gill-spacing             | close=c,crowded=w,distant=d                                                                         |
# | gill-size                | broad=b,narrow=n                                                                                    |
# | gill-color               | black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y |
# | stalk-shape              | enlarging=e,tapering=t                                                                              |
# | stalk-root               | bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?                                    |
# | stalk-surface-above-ring | fibrous=f,scaly=y,silky=k,smooth=s                                                                  |
# | stalk-surface-below-ring | fibrous=f,scaly=y,silky=k,smooth=s                                                                  |
# | stalk-color-above-ring   | brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y                            |
# | stalk-color-below-ring   | brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y                            |
# | veil-type                | partial=p,universal=u                                                                               |
# | veil-color               | brown=n,orange=o,white=w,yellow=y                                                                   |
# | ring-number              | none=n,one=o,two=t                                                                                  |
# | ring-type                | cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z                      |
# | spore-print-color        | black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y                      |
# | population               | abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y                                 |
# | habitat                  | grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d                                       |

# %%
df.info()

# %% [markdown]
# ## Visualize some features of the dataset

# %%
column_index = [3, 4, 5, 9, 14, 15, 17, 19, 20, 21, 22]
number_features = len(column_index)
grid_rows = int(np.ceil(number_features / 3))
fig, axs = plt.subplots(grid_rows, 3, figsize=(15, 5 * grid_rows))

for ax, feature in zip(axs.flatten(), df.columns[column_index]):
    sns.countplot(data=df, x=feature, hue="class", ax=ax)
    ax.set_xlabel("")
    ax.set_title(feature)

for ax in axs.flatten()[number_features:]:
    ax.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualize the class distribution

# %%
target_feature = df.columns[0]
labels, sizes = np.unique(df[target_feature], return_counts=True)

fig, ax = plt.subplots()
ax.pie(sizes, textprops={'color': "w", 'fontsize': '12'}, autopct=lambda pct: "{:.2f}%\n({:d})".format(pct, round(pct/100 * sum(sizes))))
ax.legend(["poisonous" if i == "p" else "edible" for i in labels])
ax.set_title(target_feature)
plt.show()

# %% [markdown]
# ## Preprocess the dataset

# %%
preprocessed_df = pd.get_dummies(df, drop_first=True)
preprocessed_df.head()

# %% [markdown]
# ## Split the dataset into train and test subsets

# %%
X = preprocessed_df.drop("class_p", axis=1)
y = preprocessed_df["class_p"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# %% [markdown]
# ## Get the most important features

# %%
decision_tree_classifier = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
most_important_features = pd.DataFrame(data={"Feature": X_train.columns, "Gini_importance": decision_tree_classifier.feature_importances_})\
                        .sort_values(by="Gini_importance", ascending=False).reset_index(drop=True).query("Gini_importance >= 0.01")
most_important_features

# %% [markdown]
# ## Find the optimal regularization parameter value for a SVM model

# %% [markdown]
# A [SVM model](https://en.wikipedia.org/wiki/Support_vector_machine) aims to find the best possible line, or hyperplane, that separates data points into different classes by maximizing the margin between the closest points of each category, known as support vectors. SVMs are particularly effective in high-dimensional spaces and can handle both linearly separable and non-linearly separable datasets through the use of kernel functions, which transform data into higher-dimensional space where it is easier to find a boundary.

# %%
C = np.logspace(-2, 2, 5)
parameters = {'C': C}

svc = SVC()
clf = GridSearchCV(svc, parameters, scoring='recall')
clf.fit(X_train[most_important_features["Feature"]], y_train)

# %%
plt.figure()
plt.plot(C, clf.cv_results_['mean_test_score'])
plt.scatter(clf.best_params_['C'], clf.best_score_, color='red')
plt.xlabel("C")
plt.ylabel("Mean recall score")
plt.xscale("log")
plt.show()

# %% [markdown]
# ## Evaluate the best SVM model

# %%
y_pred = clf.best_estimator_.predict(X_test[most_important_features["Feature"]])
print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["False (edible)", "True (poisonous)"])
plt.grid(False)
plt.show()
