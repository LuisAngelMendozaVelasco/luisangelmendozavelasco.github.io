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
# # Blight Violations in the City of Detroit

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/Applied_Data_Science_with_Python_Specialization/portfolio-4.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png"/>Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a Decision Tree classifier to predict whether a given blight ticket will be paid on time.

# %% [markdown]
# ## Import libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score
import folium
import kagglehub
import seaborn as sns
sns.set_style('whitegrid')

# %% [markdown]
# ## Load the dataset

# %% [markdown]
# Download the dataset from [Kaggle](https://www.kaggle.com/datasets/arjunbhaybhang/property-maintenance-blight-ticket-compliance).

# %%
kagglehub.dataset_download("arjunbhaybhang/property-maintenance-blight-ticket-compliance", output_dir="/tmp/property-maintenance-blight-ticket-compliance")

# %%
df_data = pd.read_csv('/tmp/property-maintenance-blight-ticket-compliance/train.csv', encoding="ISO-8859-1", low_memory=False)
df_data.head()

# %%
df_addresses = pd.read_csv('/tmp/property-maintenance-blight-ticket-compliance/addresses.csv')
df_addresses.head()

# %%
df_latlons = pd.read_csv('/tmp/property-maintenance-blight-ticket-compliance/latlons.csv')
df_latlons.head()

# %% [markdown]
# ## Understand the dataset

# %% [markdown]
# [Blight violations](https://data.detroitmi.gov/datasets/detroitmi::blight-violations/about) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process.
#
# Each row of the dataset corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is `True` if the ticket was paid early, on time, or within one month of the hearing data, `False` if the ticket was paid after the hearing date or not at all, and `Null` if the violator was found not responsible.
#
# File descriptions:
#
# - train.csv -> the dataset (all tickets issued 2004-2011)
# - addresses.csv & latlons.csv -> mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#
# Data fields:
#
# - ticket_id -> unique identifier for tickets
# - agency_name -> Agency that issued the ticket
# - inspector_name -> Name of inspector that issued the ticket
# - violator_name -> Name of the person/organization that the ticket was issued to
# - violation_street_number, violation_street_name, violation_zip_code -> Address where the violation occurred
# - mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country -> Mailing address of the violator
# - ticket_issued_date -> Date and time the ticket was issued
# - hearing_date -> Date and time the violator's hearing was scheduled
# - violation_code, violation_description -> Type of violation
# - disposition -> Judgment and judgement type
# - fine_amount -> Violation fine amount, excluding fees
# - admin_fee -> $20 fee assigned to responsible judgments
# - state_fee -> $10 fee assigned to responsible judgments
# - late_fee -> 10% fee assigned to responsible judgments
# - discount_amount -> discount applied, if any
# - clean_up_cost -> DPW clean-up or graffiti removal cost
# - judgment_amount -> Sum of all fines and fees
# - grafitti_status -> Flag for graffiti violations
# - payment_amount -> Amount paid, if any
# - payment_date -> Date payment was made, if it was received
# - payment_status -> Current payment status as of Feb 1 2017
# - balance_due -> Fines and fees still owed
# - collection_status -> Flag for payments in collections
# - compliance [target variable for prediction] 
#     - Null = Not responsible
#     - 0 = Responsible, non-compliant
#     - 1 = Responsible, compliant
# - compliance_detail -> More information on why each ticket was marked compliant or non-compliant
#
# **Note**: All tickets where the violators were found not responsible are not considered for the prediction task.

# %%
df_data.info()

# %%
df_addresses.info()

# %%
df_latlons.info()

# %% [markdown]
# ## Preprocess the dataset

# %%
df_merged = df_data.merge(df_addresses.merge(df_latlons, on="address"), on="ticket_id")

irrelevant_features = ['ticket_id', 'inspector_name', 'violator_name', 'violation_street_number', 'violation_street_name', 'violation_zip_code',
                       'mailing_address_str_number', 'mailing_address_str_name', 'city', 'state', 'zip_code', 'non_us_str_code', 'country',
                       'ticket_issued_date', 'hearing_date', 'violation_description', 'violation_code', 'admin_fee', 'state_fee', 'clean_up_cost',
                       'grafitti_status', 'payment_amount', 'balance_due', 'payment_date', 'collection_status', 'compliance_detail', 'address']

df_reduced = df_merged[df_merged['country'] == "USA"].drop(irrelevant_features, axis=1).dropna()
df_reduced = pd.get_dummies(df_reduced, drop_first=True)
df_reduced.head()

# %%
df_reduced.info()

# %%
print("The dataset was reduced by {:.2f}%".format((1 - df_reduced.size / df_merged.size) * 100))

# %% [markdown]
# ## Visualize the class distribution

# %%
target_feature = df_reduced.columns[4]
class_names = ["non-compliant", "compliant"]
labels, sizes = np.unique(df_reduced[target_feature], return_counts=True)

fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(sizes, textprops={'color': "w", 'fontsize': '12'}, autopct=lambda pct: "{:.2f}%\n({:d})".format(pct, round(pct/100 * sum(sizes))))
ax.legend([str(i) + " (" + class_names[i] + ")" for i in labels.astype(int)])
ax.set_title(target_feature)
plt.show()

# %% [markdown]
# ## Visualize the location of some samples

# %% [markdown]
# Red markers indicate `compliant` tickets, while blue markers indicate `non-compliant` tickets.

# %%
map = folium.Map(location=[42.36, -83.10], zoom_start=12)
feature_group = folium.FeatureGroup()
df_sample = df_reduced.sample(n=1000)

for point, compliance in zip(list(df_sample[["lat", "lon"]].to_numpy()), list(df_sample["compliance"])):
    if compliance == 1:
        feature_group.add_child(folium.Marker(point, popup='compliant', icon=folium.Icon(color='red')))
    else:
        feature_group.add_child(folium.Marker(point, popup='non-compliant', icon=folium.Icon(color='blue')))

map.add_child(feature_group)
map

# %% [markdown]
# ## Split the dataset into train and test subsets

# %%
X = df_reduced.drop("compliance", axis=1)
y = df_reduced["compliance"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# %% [markdown]
# ##  Train a Random Forest classifier

# %%
random_forest_classifier = RandomForestClassifier()
random_forest_classifier.fit(X_train, y_train)

# %% [markdown]
# ## Evaluate the Random Forest classifier

# %%
y_pred = random_forest_classifier.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=[str(i) + " (" + class_names[i] + ")" for i in labels.astype(int)])
plt.grid(False)
plt.show()

# %% [markdown]
# A [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) is a graphical plot used to illustrate the diagnostic ability of a binary classifier model at various threshold settings. It plots the true positive rate (TPR) against the false positive rate (FPR) at each threshold value. The curve helps in understanding how well the model can distinguish between the positive and negative classes across all possible classification thresholds.
#
# In the context of binary classification, the ROC curve can be used to evaluate the performance of a model by measuring the area under the curve (AUC), which summarizes the classifier's ability to discriminate between positive and negative classes. A higher AUC indicates better performance, with a perfect model having an AUC of 1 and a random model having an AUC of 0.5.
#
# However, it's important to note that the ROC curve and its AUC do not capture certain aspects of performance, such as precision and recall values, and sometimes the inclusion of areas with low sensitivity and specificity can distort the overall performance assessment.

# %%
y_proba = random_forest_classifier.predict_proba(X_test)[:, -1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr)
plt.fill_between(fpr, tpr, alpha=0.25)
plt.title('ROC curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.text(0.5, 0.5, f'AUC = {auc:.4f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.show()
