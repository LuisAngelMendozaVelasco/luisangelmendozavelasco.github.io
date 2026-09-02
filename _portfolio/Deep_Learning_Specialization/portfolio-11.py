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
# # Date Converter

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/Deep_Learning_Specialization/portfolio-11.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a Neural Machine Translation (NMT) model to convert human-readable dates into machine-readable dates.

# %% [markdown]
# [Neural Machine Translation (NMT)](https://en.wikipedia.org/wiki/Neural_machine_translation) is a deep learning approach to machine translation that uses artificial neural networks to predict the likelihood of a sequence of words in the target language, given a sequence of words in the source language.
#
# For our case, the model will input dates written in a variety of possible formats (e.g. "the 29th of August 1958", "03/30/1968", "24 JUNE 1987") and translate them into standardized, machine-readable dates (e.g. "1958-08-29", "1968-03-30", "1987-06-24"). The model will be trained on a dataset of 10,000 human-readable dates and will learn to generate dates in the common machine-readable format YYYY-MM-DD.

# %% [markdown]
# ## Import libraries

# %%
# # !pip install faker babel
from faker import Faker
from babel.dates import format_date
from datetime import datetime
import random
from tqdm import tqdm
import numpy as np
from keras import utils, layers, activations, Model, callbacks, optimizers
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


# %% [markdown]
# ## Create the dataset

# %%
def load_date():
    """
    Loads some fake dates.
    Returns a tuple containing human readable string, machine readable string, and date object.
    """

    fake = Faker()
    dt = fake.date_object()
    
    # Define format of the data we would like to generate
    FORMATS = ['short', 'medium', 'long', 'full', 'full', 'full', 'full', 'full', 'full', 'full', 'full', 'full', 'full',
               'd MMM YYY', 'd MMMM YYY', 'dd MMM YYY', 'd MMM, YYY', 'd MMMM, YYY', 'dd, MMM YYY', 'd MM YY', 'd MMMM YYY',
               'MMMM d YYY', 'MMMM d, YYY', 'dd.MM.YY']
                
    try:
        human_readable = format_date(dt, format=random.choice(FORMATS), locale='en_US')
        human_readable = human_readable.lower()
        human_readable = human_readable.replace(',', '')
        machine_readable = dt.isoformat()       
    except AttributeError as e:
        return None, None, None

    return human_readable, machine_readable, dt

def load_dataset(n):
    """Loads a dataset with "n" examples and vocabularies, where "n" is the number of examples to generate."""
    
    human_vocab = set()
    machine_vocab = set()
    dataset = []
    
    for _ in tqdm(range(n)):
        h, m, _ = load_date()

        if h is not None:
            dataset.append((h, m))
            human_vocab.update(tuple(h))
            machine_vocab.update(tuple(m))
    
    human = dict(zip(sorted(human_vocab) + ['<unk>', '<pad>'], list(range(len(human_vocab) + 2))))
    inv_machine = dict(enumerate(sorted(machine_vocab)))
    machine = {v:k for k, v in inv_machine.items()}

    return dataset, human, machine, inv_machine


# %%
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(n=10000)
dataset[:10]


# %% [markdown]
# - **dataset**: A list of tuples of (human-readable date, machine-readable date).
# - **human_vocab**: A python dictionary mapping all characters used in human-readable dates to an integer-valued index.
# - **machine_vocab**: A python dictionary mapping all characters used in machine-readable dates to an integer-valued index.
# - **inv_machine_vocab**: The inverse dictionary of machine_vocab, mapping from indices back to characters.

# %% [markdown]
# ## Preprocess the dataset

# %%
def string_to_int(string, length, vocab):
    """
    Converts all strings in the vocabulary into a list of integers representing the positions of the
    input string's characters in the "vocab".
    
    Arguments:
    string -- Input string, e.g. 'Wed 10 Jul 2007'
    length -- The number of time steps, it determines if the output will be padded or cut
    vocab -- Vocabulary, dictionary used to index every character of the "string"
    
    Returns:
    rep -- List of integers (or '<unk>') (size = length) representing the position of the string's character in the vocabulary
    """
    
    # Make lower to standardize
    string = string.lower()
    string = string.replace(',', '')
    
    if len(string) > length:
        string = string[:length]
        
    rep = list(map(lambda x: vocab.get(x, '<unk>'), string))
    
    if len(string) < length:
        rep += [vocab['<pad>']] * (length - len(string))
    
    return rep

def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):
    X, Y = zip(*dataset)
    X = np.array([string_to_int(i, Tx, human_vocab) for i in X])
    Y = [string_to_int(t, Ty, machine_vocab) for t in Y]
    
    Xoh = np.array(list(map(lambda x: utils.to_categorical(x, num_classes=len(human_vocab)), X)))
    Yoh = np.array(list(map(lambda x: utils.to_categorical(x, num_classes=len(machine_vocab)), Y)))

    return X, np.array(Y), Xoh, Yoh


# %% [markdown]
# - **X**: A processed version of the human-readable dates in the training set.
#     - Each character in **X** is replaced by an index (integer) mapped to the character using **human_vocab**.
#     - Each date is padded to ensure a length of $T_x$ using a special character (< pad >).
#     - **X**.shape = (m, $T_x$) where m is the number of training examples in a batch.
# - **Y**: A processed version of the machine-readable dates in the training set.
#     - Each character in **Y** is replaced by an index (integer) mapped to the character using **machine_vocab**.
#     - **Y**.shape = (m, $T_y$).
# - **Xoh**: One-hot version of **X**.
#     - Each index in **X** is converted to the one-hot representation (if the index is 2, the one-hot version has the index position 2 set to 1, and the remaining positions are 0).
#     - **Xoh**.shape = (m, $T_x$, len(**human_vocab**)).
# - **Yoh**: One-hot version of **Y**.
#     - Each index in **Y** is converted to the one-hot representation.
#     - **Yoh**.shape = (m, $T_y$, len(**machine_vocab**)).
#     - len(**machine_vocab**) = 11 since there are 10 numeric digits (0 to 9) and the **-** symbol.

# %% [markdown]
# ## Split the dataset into train, validation and test subsets

# %%
Tx = 30 # Assume Tx is the maximum length of the human readable date
Ty = 10 # "YYYY-MM-DD" is 10 characters long
dataset_train = dataset[:8000]
dataset_validation = dataset[8000:9000]
dataset_test = dataset[9000:]

print("Number of training samples:", len(dataset_train))
print("Number of validation samples:", len(dataset_validation))
print("Number of test samples:", len(dataset_test))

# %%
X_train, Y_train, Xoh_train, Yoh_train = preprocess_data(dataset_train, human_vocab, machine_vocab, Tx, Ty)
X_validation, Y_validation, Xoh_validation, Yoh_validation = preprocess_data(dataset_validation, human_vocab, machine_vocab, Tx, Ty)
X_test, Y_test, Xoh_test, Yoh_test = preprocess_data(dataset_test, human_vocab, machine_vocab, Tx, Ty)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("Xoh_train shape:", Xoh_train.shape)
print("Yoh_train shape:", Yoh_train.shape)

print("\nSource date:", dataset_train[0][0])
print("Target date:", dataset_train[0][1])

print("\nSource after preprocessing (indices):", X_train[0])
print("Target after preprocessing (indices):", Y_train[0])

print("\nSource after preprocessing (one-hot):\n", Xoh_train[0])
print("Target after preprocessing (one-hot):\n", Yoh_train[0])


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
            print(f"\telapsed time: {(self.epoch_stop_time - self.epoch_start_time).total_seconds():.3f}s - loss: {logs['loss']:.4f} - val_loss: {logs['val_loss']:.4f}")


# %% [markdown]
# ## Build a NMT model with attention

# %% [markdown]
# The [attention mechanism](https://arxiv.org/pdf/1508.04025v5) tells a NMT model where it should pay attention at each step.

# %%
def softmax(x, axis=1):
    """Custom softmax activation function.
    # Arguments
        x: Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """

    ndim = x.shape.rank

    if ndim == 2:
        return activations.softmax(x)
    elif ndim > 2:
        e = tf.exp(x - tf.reduce_max(x, axis=axis, keepdims=True))
        s = tf.reduce_sum(e, axis=axis, keepdims=True)
        
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D!')


# %% [markdown]
# Define shared layers as global variables:

# %%
repeator = layers.RepeatVector(Tx) # Repeat the input n times
concatenator = layers.Concatenate(axis=-1) # Layer that concatenates a list of inputs
densor1 = layers.Dense(10, activation="tanh")
densor2 = layers.Dense(1, activation="relu")
activator = layers.Activation(softmax, name='attention_weights') # Use a custom softmax(axis=1)
dotor = layers.Dot(axes=1) # Layer that computes a dot product between samples in two tensors


# %%
def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- Hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- Previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- Context vector, input of the next (post-attention) LSTM cell
    """

    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that it can  be concatenated with all hidden states "a"
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis
    concat = concatenator([a, s_prev])
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies"
    e = densor1(concat)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies"
    energies = densor2(e)
    # Use activator on "energies" to compute the attention weights "alphas"
    alphas = activator(energies)
    # Use dotor together with "alphas" and "a", in this order, to compute the context vector to be given to the next (post-attention) LSTM-cell
    context = dotor([alphas, a])
    
    return context


# %%
n_a = 32 # Number of units for the pre-attention, bi-directional LSTM's hidden state "a"
n_s = 64 # Number of units for the post-attention LSTM's hidden state "s"

# This is the post attention LSTM cell
post_activation_LSTM_cell = layers.LSTM(n_s, return_state=True)
output_layer = layers.Dense(len(machine_vocab), activation=softmax)

# Define the inputs of the model with a shape (Tx,)
inputs = layers.Input(shape=(Tx, len(human_vocab)))

# Define s0 (initial hidden state) and c0 (initial cell state) for the decoder LSTM with shape (n_s,)
s0 = layers.Input(shape=(n_s,))
c0 = layers.Input(shape=(n_s,))
s, c = s0, c0

# Initialize empty list of outputs
outputs = []

# Define the pre-attention Bi-LSTM
a = layers.Bidirectional(layers.LSTM(n_a, return_sequences=True))(inputs)

# Iterate for Ty steps
for t in range(Ty):
    # Perform one step of the attention mechanism to get back the context vector at step t
    context = one_step_attention(a, s)
    # Apply the post-attention LSTM cell to the "context" vector
    s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])
    # Apply Dense layer to the hidden state output of the post-attention LSTM
    output = output_layer(s)
    # Append "output" to the "outputs" list
    outputs.append(output)

# Create model instance taking three inputs and returning the list of outputs
model = Model(inputs=[inputs, s0, c0], outputs=outputs)

model.summary()

# %% [markdown]
# ## Compile and train the NMT model

# %%
model.compile(optimizer=optimizers.Adam(1e-2), loss='categorical_crossentropy')

epochs = 100
patience = int(epochs / 10)
epochs_to_show = [0] + [i for i in range(patience - 1, epochs, patience)]
custom_verbose = CustomVerbose(epochs_to_show)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1)

# Initialize s0 and c0
s0_train = np.zeros((Xoh_train.shape[0], n_s))
c0_train = np.zeros((Xoh_train.shape[0], n_s))
s0_validation = np.zeros((Xoh_validation.shape[0], n_s))
c0_validation = np.zeros((Xoh_validation.shape[0], n_s))
Xoh_train_ = [Xoh_train, s0_train, c0_train]
Yoh_train_ = list(Yoh_train.swapaxes(0, 1)) # The "Yoh_train_" needs to be a list of 10 elements of shape (Xoh_train.shape[0], T_y)
Xoh_validation_ = [Xoh_validation, s0_validation, c0_validation]
Yoh_validation_ = list(Yoh_validation.swapaxes(0, 1))
history = model.fit(Xoh_train_, Yoh_train_, epochs=epochs, batch_size=128, verbose=0, validation_data=(Xoh_validation_, Yoh_validation_), callbacks=[custom_verbose, early_stopping])

# %%
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Training", "Validation"])
plt.show()

# %% [markdown]
# ## Evaluate the NMT model

# %%
s0_test = np.zeros((1000, n_s))
c0_test = np.zeros((1000, n_s))
y_pred = model.predict([Xoh_test, s0_test, c0_test], verbose=0)
y_pred = np.argmax(y_pred, axis=-1)
y_pred_ = []

for i in np.swapaxes(y_pred, 0, 1):
    date = [inv_machine_vocab[j] for j in i]
    y_pred_.append(''.join(date))

X_test, y_test = zip(*dataset_test)
is_correct_prediction = [True if i == j else False for i, j in zip(y_test, y_pred_)]
print(f"Test accuracy = {100 * (sum(is_correct_prediction) / len(is_correct_prediction)):.2f}%")

# %% [markdown]
# Show some mislabeled dates:

# %%
indexes = np.where([not x for x in is_correct_prediction])[0]

for i in indexes[:5]:
    print("\nSource date:", X_test[i])
    print("True date:", y_test[i])
    print("Predicted date:", y_pred_[i])
