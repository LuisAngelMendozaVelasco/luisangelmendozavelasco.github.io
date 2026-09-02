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
# # Poem Generator

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/Deep_Learning_Specialization/portfolio-8.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a Long Short-Term Memory (LSTM) network to generate Shakespeare-style poems.

# %% [markdown]
# A [Long Short-Term Memory (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory) network is a type of Recurrent Neural Network (RNN) designed to handle the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) and learn long-term dependencies in sequential data. LSTMs have become a fundamental component in many deep learning applications, including natural language processing, speech recognition, and time series forecasting.

# %% [markdown]
# ## Import libraries

# %%
import numpy as np
from keras import Sequential, Input, layers, initializers, callbacks
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# %% [markdown]
# ## Download the dataset

# %%
# !wget -nc --progress=bar:force:noscroll https://raw.githubusercontent.com/LuisAngelMendozaVelasco/Deep_Learning_Specialization/main/Sequence_Models/Week1/Labs/data/shakespeare.txt -P /tmp

# %% [markdown]
# ## Load the dataset

# %%
with open('/tmp/shakespeare.txt', encoding='utf-8') as f:
    text = f.read().lower()

chars = sorted(list(set(text)))
print('Number of unique characters in the dataset:', len(chars))
print("\nText:\n", text[:1000] + "...")


# %% [markdown]
# ## Create the training set

# %%
def build_data(text, Tx=40, stride=3):
    """
    Create a training set by scanning a window of size Tx over the text corpus, with stride 3.

    Arguments:
    text -- string, corpus of Shakespearian poem
    Tx -- sequence length, number of time-steps (or characters) in one training example
    stride -- how much the window shifts itself while scanning

    Returns:
    X -- list of training examples
    Y -- list of training labels
    """

    X = []
    Y = []

    for i in range(0, len(text) - Tx, stride):
        X.append(text[i: i + Tx])
        Y.append(text[i + Tx])

    print('Number of training examples:', len(X))

    return X, Y

def vectorization(X, Y, n_x, char_indexes, Tx=40):
    """
    Convert X and Y (lists) into arrays to be given to a recurrent neural network.

    Arguments:
    X --
    Y --
    Tx -- integer, sequence length

    Returns:
    x -- array of shape (m, Tx, len(chars))
    y -- array of shape (m, len(chars))
    """

    m = len(X)
    x = np.zeros((m, Tx, n_x), dtype=bool)
    y = np.zeros((m, n_x), dtype=bool)

    for i, sentence in enumerate(X):
        for t, char in enumerate(sentence):
            x[i, t, char_indexes[char]] = 1

        y[i, char_indexes[Y[i]]] = 1

    return x, y


# %%
char_indexes = dict((c, i) for i, c in enumerate(chars))
indexes_char = dict((i, c) for i, c in enumerate(chars))

X, Y = build_data(text)
x, y = vectorization(X, Y, n_x=len(chars), char_indexes=char_indexes)


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
            print(f"\telapsed time: {(self.epoch_stop_time - self.epoch_start_time).total_seconds():.3f}s - loss: {logs['loss']:.4f}")


# %% [markdown]
# ## Build a LSTM network

# %%
model = Sequential([Input(shape=(40, 38)),
                    layers.LSTM(units=128, recurrent_activation="hard_sigmoid", kernel_initializer=initializers.VarianceScaling(mode="fan_avg", distribution="uniform"), return_sequences=True),
                    layers.Dropout(rate=0.5),
                    layers.LSTM(units=128, recurrent_activation="hard_sigmoid", kernel_initializer=initializers.VarianceScaling(mode="fan_avg", distribution="uniform")),
                    layers.Dropout(rate=0.5),
                    layers.Dense(units=38, activation="linear", kernel_initializer=initializers.VarianceScaling(mode="fan_avg", distribution="uniform")),
                    layers.Activation(activation="softmax")])

model.summary()

# %% [markdown]
# ## Compile and train the LSTM network

# %%
model.compile(optimizer="adam", loss="categorical_crossentropy")

epochs = 500
epochs_to_show = [0] + [i for i in range(int(epochs / 10) - 1, epochs, int(epochs / 10))]
custom_verbose = CustomVerbose(epochs_to_show)
history = model.fit(x, y, batch_size=128, epochs=epochs, verbose=0, callbacks=[custom_verbose])

# %%
plt.figure()
plt.plot(history.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


# %% [markdown]
# ## Evaluate the LSTM network

# %%
def sample(preds, temperature=1.0):
    """Sample an index from a probability array"""

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    out = np.random.choice(range(len(chars)), p=probas.ravel())

    return out

def generate_output(length, Tx=40):
    """Prompt the user for input and generate a poem by transforming the model`s predictions into text"""

    generated = ''
    usr_input = input("Write the beginning of your poem, the Shakespeare machine will complete it. Your input is: ")
    sentence = ('{0:0>' + str(Tx) + '}').format(usr_input).lower() # zero pad the sentence to Tx characters
    generated += usr_input

    sys.stdout.write("Here is your poem: \n\n")
    sys.stdout.write(usr_input)

    for i in range(length):
        x_pred = np.zeros((1, Tx, len(chars)))

        for t, char in enumerate(sentence):
            if char != '0':
                x_pred[0, t, char_indexes[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature=1.0)
        next_char = indexes_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()

    sys.stdout.write("...")


# %%
generate_output(length=500)
