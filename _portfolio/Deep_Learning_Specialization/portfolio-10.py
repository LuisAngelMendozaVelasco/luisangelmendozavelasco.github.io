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
#     display_name: portfolio
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Emojify

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/Deep_Learning_Specialization/portfolio-10.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a Long Short-Term Memory (LSTM) network to predict the most appropriate emoji for a given text.

# %% [markdown]
# ## Import libraries

# %%
# # !pip install emoji
import numpy as np
from datetime import datetime
from keras import layers, Model, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import pandas as pd
import emoji
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# %% [markdown]
# ## Download GloVe pre-trained word vectors

# %% [markdown]
# [Global Vectors for Word Representation (GloVe)](https://en.wikipedia.org/wiki/GloVe) is a type of word embedding that encodes the co-occurrence probability ratio between two words as vector differences. It’s an unsupervised learning algorithm to obtain vector representations for words, mapping them into a meaningful space where the distance between words is related to semantic similarity.

# %% language="bash"
#
# if [ -e "/tmp/glove.6B.zip" ]; then
#     echo "glove.6B.zip already exists!"
# else
#     wget -nc https://nlp.stanford.edu/data/glove.6B.zip -P /tmp/
# fi
#
# unzip -qn /tmp/glove.6B.zip -d /tmp/

# %% [markdown]
# ## Load the dataset

# %%
url_train = 'https://raw.githubusercontent.com/LuisAngelMendozaVelasco/Deep_Learning_Specialization/main/Sequence_Models/Week2/Labs/data/train_emoji.csv'
url_test = 'https://raw.githubusercontent.com/LuisAngelMendozaVelasco/Deep_Learning_Specialization/main/Sequence_Models/Week2/Labs/data/tesss.csv'

data_train = pd.read_csv(url_train, header=None)
data_test = pd.read_csv(url_test, header=None)

X_train, y_train = data_train[0].to_numpy(), data_train[1].to_numpy()
X_test, X_validation, y_test, y_validation = train_test_split(data_test[0].to_numpy(), data_test[1].to_numpy(), test_size=0.5, random_state=0)

# %%
print("Number of training samples:", len(X_train))
print("Number of validation samples:", len(X_validation))
print("Number of test samples:", len(X_test))


# %% [markdown]
# ## Visualize the dataset

# %%
def label_to_emoji(label):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
    """
    
    emoji_dictionary = {"0": ":heart:",
                        "1": ":baseball:",
                        "2": ":smile:",
                        "3": ":disappointed:",
                        "4": ":fork_and_knife:"}

    return emoji.emojize(emoji_dictionary[str(label)], language='alias')


# %%
max_len = len(max(X_train, key=len).split())
min_len = len(min(X_train, key=len).split())
mean_len = sum([len(x.split()) for x in X_train]) / len(X_train)
print(f'Maximum number of words in a sentence: {max_len}')
print(f'Minimum number of words in a sentence: {min_len}')
print(f'Mean number of words in a sentence: {mean_len:.2f}\n')

for index in range(10):
    print(X_train[index], label_to_emoji(y_train[index]))

# %% [markdown]
# ## Visualize the class distribution

# %%
labels, sizes = np.unique(y_train, return_counts=True)

fig, ax = plt.subplots()
ax.pie(sizes, textprops={'color': "w", 'fontsize': '12'}, autopct=lambda pct: "{:.2f}%\n({:d})".format(pct, round(pct/100 * sum(sizes))))
ax.legend(labels)
ax.set_title("Class")
plt.show()


# %% [markdown]
# ## Load the pre-trained word vectors

# %% [markdown]
# The model will use the pre-trained 50-dimensional [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) to load the vector representations of words.

# %%
def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding='utf-8') as f:
        words = set()
        word_to_vec_map = {}

        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}

        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1

    return words_to_index, index_to_words, word_to_vec_map


# %%
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('/tmp/glove.6B.50d.txt')


# %% [markdown]
# ## Convert sentences into a list of indices

# %% [markdown]
# Most deep learning frameworks require that all sequences in the same batch have the same length. The common solution to handling sequences of different length is to use padding. 
#
# Specifically:
#
# - Set a maximum sequence length.
# - Pad all sequences to have the same length.

# %%
def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to 'keras.layers.Embedding()'.
    
    Arguments:
    X -- Array of sentences (strings), of shape (m, 1)
    word_to_index -- A dictionary containing the each word mapped to its index
    max_len -- Maximum number of words in a sentence. We can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- Array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """

    # Number of training examples
    m = X.shape[0]
    # Initialize X_indices as a numpy matrix of zeros and the correct shape
    X_indices = np.zeros((len(X), max_len))
    
    # Loop over training examples
    for i in range(m):
        # Convert the i-th training sentence in lower case and split it into words
        sentence_words = [w.lower() for w in X[i].split()]
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            # If w exists in the dictionary
            if w in word_to_index:
                # Set the (i, j)-th entry of X_indices to the index of the correct word
                X_indices[i, j] = word_to_index[w]
                # Increment j to j + 1
                j = j + 1
    
    return X_indices


# %% [markdown]
# ## Load a pre-trained embedding layer

# %% [markdown]
# In Keras, the embedding matrix is represented as a "layer".
#
# - The embedding matrix maps word indices to embedding vectors.
#     - The word indices are positive integers.
#     - The embedding vectors are dense vectors of fixed size.
#     - A "dense" vector is the opposite of a sparse vector. It means that most of its values are non-zero.
# - The embedding matrix can be derived in two ways:
#     - Training a model to derive the embeddings from scratch.
#     - Using a pre-trained embedding.

# %%
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- Dictionary mapping words to their GloVe vector representation
    word_to_index -- Dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- Pretrained layer Keras instance
    """
    
    vocab_size = len(word_to_index) + 1          # Adding 1 to fit Keras embedding (requirement)
    any_word = list(word_to_vec_map.keys())[0]
    emb_dim = word_to_vec_map[any_word].shape[0] # Define dimensionality of your GloVe word vectors (= 50)

    # Step 1
    # Initialize the embedding matrix as a numpy array of zeros.
    embedding_matrix = np.zeros((vocab_size, emb_dim))

    # Step 2
    # Set each row "idx" of the embedding matrix to be 
    # the word vector representation of the idx'th word of the vocabulary.
    for word, idx in word_to_index.items():
        embedding_matrix[idx, :] = word_to_vec_map[word]
    
    # Step 3
    # Define Keras embedding layer with the correct input and output sizes.
    # Make it non-trainable.
    embedding_layer = layers.Embedding(vocab_size, emb_dim, trainable=False)

    # Step 4
    # Build the embedding layer, it is required before setting the weights of the embedding layer. 
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix. The layer is now pretrained.
    embedding_layer.set_weights([embedding_matrix])

    return embedding_layer


# %% [markdown]
# ## Define a custom callback

# %%
class CustomVerbose(callbacks.Callback):
    def __init__(self, epochs_to_show):
        self.epochs_to_show = epochs_to_show

    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.epochs_to_show:
            self.epoch_start_time = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.epochs_to_show:
            self.epoch_stop_time = datetime.now()
            print(f"Epoch {epoch + 1}/{self.epochs_to_show[-1] + 1}")
            print(f"\telapsed time: {(self.epoch_stop_time - self.epoch_start_time).total_seconds():.3f}s - accuracy: {logs['categorical_accuracy']:.4f} - loss: {logs['loss']:.4f} - val_accuracy: {logs['val_categorical_accuracy']:.4f} - val_loss: {logs['val_loss']:.4f}")


# %% [markdown]
# ## Build a LSTM network

# %%
# Define sentence_indices as the input.
sentence_indices = layers.Input((max_len,), dtype='int32')

# Create the embedding layer pretrained with GloVe Vectors.
embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

# Propagate sentence_indices through the embedding layer.
embeddings = embedding_layer(sentence_indices)    

# Propagate the embeddings through an LSTM layer with 128-dimensional hidden state.
# The returned output should be a batch of sequences.
x = layers.LSTM(128, return_sequences=True)(embeddings)
# Add dropout with a probability of 0.5
x = layers.Dropout(0.5)(x) 
# Propagate x trough another LSTM layer with 128-dimensional hidden state.
# The returned output should be a single hidden state, not a batch of sequences.
x = layers.LSTM(128)(x)
# Add dropout with a probability of 0.5
x = layers.Dropout(0.5)(x) 
# Propagate x through a Dense layer with 5 units.
x = layers.Dense(5)(x)
# Add a softmax activation.
outputs = layers.Activation('softmax')(x)

# Create Model instance which converts sentence_indices into outputs.
model = Model(inputs=sentence_indices, outputs=outputs)
model.summary()

# %% [markdown]
# ## Compile and train the LSTM network

# %%
model.compile(optimizer=optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

epochs = 200
patience = int(epochs / 10)
epochs_to_show = [0] + [i for i in range(patience - 1, epochs, patience)]
custom_verbose = CustomVerbose(epochs_to_show)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
X_train_, y_train_ = sentences_to_indices(X_train, word_to_index, max_len), tf.one_hot(y_train, depth=5)
X_validation_, y_validation_ = sentences_to_indices(X_validation, word_to_index, max_len), tf.one_hot(y_validation, depth=5)
history = model.fit(X_train_, y_train_, epochs=epochs, batch_size=32, verbose=0, validation_data=(X_validation_, y_validation_), callbacks=[custom_verbose, early_stopping])

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.legend(["Training", "Validation"])

ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.legend(["Training", "Validation"])

plt.show()

# %% [markdown]
# ## Evaluate the LSTM network

# %%
X_test_ = sentences_to_indices(X_test, word_to_index, max_len)
y_test_ = tf.one_hot(y_test, depth=5)
prediction_proba = model.predict(X_test_, verbose=0)
y_pred = np.argmax(prediction_proba, axis=1)

print(classification_report(y_test, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.grid(False)
plt.show()

# %% [markdown]
# Show the mislabeled sentences:

# %%
print({i:label_to_emoji(i) for i in range(5)})
prediction = model.predict(X_test_, verbose=0)

for i in range(len(X_test)):
    num = np.argmax(prediction[i])
    
    if(num != y_test[i]):
        print("Sentence: " + X_test[i] + " -> Expected emoji: " + label_to_emoji(y_test[i]) + ", Prediction: " + label_to_emoji(num))
