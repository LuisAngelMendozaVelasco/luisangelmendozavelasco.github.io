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
# # Spam Detection

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/Applied_Data_Science_with_Python_Specialization/portfolio-5.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png"/>Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train multiple classifiers and evaluate their effectiveness in predicting whether a message is spam or not.

# %% [markdown]
# ## Import libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.svm import SVC
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
import seaborn as sns
sns.set_style("whitegrid")

# %% [markdown]
# ## Load the dataset

# %%
file_url = 'https://raw.githubusercontent.com/LuisAngelMendozaVelasco/Applied_Data_Science_with_Python_Specialization/main/Applied_Text_Mining_in_Python/Week3/Labs/data/spam.csv'
df = pd.read_csv(file_url)
df['target'] = np.where(df['target'] == 'spam', 1, 0)
df.head()

# %% [markdown]
# ## Understand the dataset

# %% [markdown]
# The [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) dataset is a widely used benchmark for text classification, specifically designed for identifying spam SMS messages. It contains a collection of 5,574 SMS messages in English, labeled as either “spam” or “non-spam”.

# %%
df.info()

# %% [markdown]
# ## Visualize the class distribution

# %%
target_feature = df.columns[-1]
class_names = ["non-spam", "spam"]
labels, sizes = np.unique(df[target_feature], return_counts=True)

fig, ax = plt.subplots()
ax.pie(sizes, textprops={'color': "w", 'fontsize': '12'}, autopct=lambda pct: "{:.2f}%\n({:d})".format(pct, round(pct/100 * sum(sizes))))
ax.legend([str(i) + " (" + class_names[i] + ")" for i in labels])
ax.set_title(target_feature)
plt.show()

# %% [markdown]
# ## Preprocess the dataset

# %% [markdown]
# ### Convert the messages to a matrix of token counts

# %% [markdown]
# Count vectorization is a method used in [Natural Language Processing (NLP)](https://en.wikipedia.org/wiki/Natural_language_processing) to convert text documents into numerical vectors based on the frequency of words or tokens. It involves tokenizing the text, which means breaking it down into individual words or tokens, and then counting the occurrences of each token in the document. This process results in a matrix where each row represents a document and each column represents a unique token, with the cell values indicating the frequency of each token in the corresponding document.

# %%
vectorizer = CountVectorizer()
vectorizer.fit(df['text'])

# %% [markdown]
# Tokens with the largest lengths:

# %%
tokens = [(token, len(token)) for token in vectorizer.vocabulary_.keys()]
print("There are {} unique tokens.".format(len(tokens)))
tokens = pd.DataFrame(sorted(tokens, key=lambda item: item[1], reverse=True), columns=["token", "length"])
tokens.head(10)

# %%
plt.figure()
sns.histplot(data=tokens, x="length", bins="doane")
plt.title("Distribution of token lengths")
plt.xlabel("Token length")
plt.yscale("log")
plt.show()

# %% [markdown]
# ### Get the length of the messages

# %%
df['length'] = df['text'].str.len()
nonspam_mean_length = np.mean(df[df['target'] == 0]['length'])
spam_mean_length = np.mean(df[df['target'] == 1]['length'])

print("Average length of spam messages: {:.2f}".format(spam_mean_length))
print("Average length of non-spam messages: {:.2f}".format(nonspam_mean_length))

# %%
plt.figure()
sns.histplot(data=df, x="length", hue="target", bins="doane")
plt.title("Distribution of message lengths")
plt.xlabel("Message length")
plt.yscale("log")
plt.show()

# %% [markdown]
# ### Get the number of digits in the messages

# %%
df['digits'] = df['text'].str.findall(r'\d').str.len()
nonspam_mean_digits = np.mean(df[df['target'] == 0]['digits'])
spam_mean_digits = np.mean(df[df['target'] == 1]['digits'])

print("Average number of digits in spam messages: {:.2f}".format(spam_mean_digits))
print("Average number of digits in not spam messages: {:.2f}".format(nonspam_mean_digits))

# %%
plt.figure()
sns.histplot(data=df, x="digits", hue="target", bins="doane")
plt.title("Distribution of message digit counts")
plt.xlabel("Message digit count")
plt.yscale("log")
plt.show()

# %% [markdown]
# ### Get the number of non-word characters in the messages

# %%
df['non_word'] = df['text'].str.findall(r'\W').str.len()
nonspam_mean_non_word = np.mean(df[df['target'] == 0]['non_word'])
spam_mean_non_word = np.mean(df[df['target'] == 1]['non_word'])

print("Average number of non-word characters in spam messages: {:.2f}".format(spam_mean_non_word))
print("Average number of non-word characters in non-spam messages: {:.2f}".format(nonspam_mean_non_word))

# %%
plt.figure()
sns.histplot(data=df, x="non_word", hue="target", bins="doane")
plt.title("Distribution of message non-word counts")
plt.xlabel("Message non-word count")
plt.yscale("log")
plt.show()

# %%
df.head()


# %%
def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

X_vectorized = vectorizer.transform(df['text'])
X_vectorized = add_feature(X_vectorized, [df['length'], df['digits'], df['non_word']])

# %% [markdown]
# ## Split the dataset into train and test subsets

# %%
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, df['target'], random_state=0)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# %% [markdown]
# ## Train a Multinomial Naive Bayes classifier

# %% [markdown]
# A [Multinomial Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) is particularly effective in text classification and natural language processing applications. It assumes that the features are discrete counts or frequencies, such as word counts in documents, and it models the likelihood of these features using a multinomial distribution. This classifier is based on Bayes' theorem and assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. It is widely used for tasks like spam filtering, document classification, sentiment analysis, and customer segmentation.

# %%
classifier = MultinomialNB(alpha=1e-3)
classifier.fit(X_train, y_train)

# %% [markdown]
# ## Evalute the Multinomial Naive Bayes classifier

# %%
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=[str(i) + " (" + class_names[i] + ")" for i in labels])
plt.grid(False)
plt.show()

# %% [markdown]
# ## Train a SVM model

# %%
classifier = SVC(C=1e3)
classifier.fit(X_train, y_train)

# %% [markdown]
# ## Evaluate the SVM model

# %%
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=[str(i) + " (" + class_names[i] + ")" for i in labels])
plt.grid(False)
plt.show()

# %% [markdown]
# ## Train a Logistic Regression classifier

# %%
classifier = LogisticRegression(C=100, max_iter=1000)
classifier.fit(X_train, y_train)

# %% [markdown]
# ## Evaluate the Logistic Regression classifier

# %%
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=[str(i) + " (" + class_names[i] + ")" for i in labels])
plt.grid(False)
plt.show()
