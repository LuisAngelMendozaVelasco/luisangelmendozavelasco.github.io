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
# # Music Generator

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/Deep_Learning_Specialization/portfolio-9.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a Long Short-Term Memory (LSTM) network to generate Jazz music.

# %% [markdown]
# ## Download the dataset and auxiliary files

# %% language="bash"
#
# if [ -e "/tmp/jazz_generator.zip" ]; then
#     echo "jazz_generator.zip already exists!"
# else
#     gdown 1yZ5vKsZiyZZaGfBP-ixfN1PeFljvrt3e -O /tmp/
# fi
#
# unzip -qn /tmp/jazz_generator.zip -d /tmp

# %% [markdown]
# ## Import libraries

# %%
# # !pip install music21==6.7.1 mido pydub
import sys
sys.path.insert(0, '/tmp')
import IPython
from datetime import datetime
import music21
from jazz_generator.data_utils import *
from keras import layers, Input, Model, optimizers, callbacks
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# %%
music21.__version__

# %% [markdown]
# ## Load the dataset

# %% [markdown]
# The preprocessing of the musical data is rendered in terms of [note values](https://en.wikipedia.org/wiki/Note_value). In music theory, a "value" captures the information needed to play multiple notes at the same time.

# %%
# Load the raw music data and preprocess it into "values"
X, Y, n_values, indices_values, chords = load_music_utils('/tmp/jazz_generator/original_metheny.mid')

print('Number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('Total number of unique values:', n_values)
print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)
print('Number of chords:', len(chords))

# %% [markdown]
# - **X**: A ($m$, $T_x$, 90) dimensional array.
#     - It has $m$ training examples, each of which is a snippet of $T_x$ musical values.
#     - At each time step, the input is one of 90 different possible values, represented as a one-hot vector.
#         - For example, X[$i$, $t$, :] is a one-hot vector representing the value of the $i-th$ example at time $t$.
#
# - **Y**: A ($T_y$, $m$, 90) dimensional array.
#     - It is essentially the same as X, but shifted one step to the left (to the past).
#     - The data in Y is reordered, where $T_y$ = $T_x$. This format makes it more convenient to feed into the LSTM.
#     - The model will use the previous values to predict the next value.
#         - So the sequence model will try to predict $y^{<t>}$ given $x^{<1>}$, ..., $x^{<t>}$.
#
# - **n_values**: The number of unique values in this dataset.
#
# - **indices_values**: Python dictionary mapping integers 0 through 89 to musical values.
#
# - **chords**: Chords used in the input midi.

# %% [markdown]
# An audio snippet from the training set:

# %%
IPython.display.Audio('/tmp/jazz_generator/30s_seq.mp3')


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

# %% [markdown]
# Implement the model composed of $T_x$ LSTM cells, where each cell is responsible for learning the following note based on the previous note and context. Each cell has the following schema: 
#
# [$X_{t}$, $a0_{t-1}$, $c0_{t-1}$] -> RESHAPE() -> LSTM() -> DENSE()
#
# The model will call the LSTM layer $T_x$ times using a for-loop. It is important that all $T_x$ copies have the same weights, the steps must have shared weights that aren't re-initialized. 

# %%
# Number of dimensions for the hidden state of each LSTM cell
n_a = 64

# Length of the sequences in the corpus
Tx = X.shape[1]

# Referencing a globally defined shared layer will utilize the same layer-object instance at each time step
reshaper = layers.Reshape((1, n_values))
LSTM_cell = layers.LSTM(n_a, return_state=True)
densor = layers.Dense(n_values, activation='softmax')

# Define inputs, the initial hidden state 'a0' and initial cell state 'c0'
inputs = Input(shape=(Tx, n_values))
a0 = Input(shape=(n_a,))
c0 = Input(shape=(n_a,))
a, c = a0, c0

# Create an empty list to append the outputs while iterate
outputs = []

# Loop over Tx
for t in range(Tx):
    # Select the t-th time step vector from inputs
    x = inputs[:, t, :]
    # Use Reshape() to reshape x to be (1, n_values)
    x = reshaper(x)
    # Perform one step of the LSTM_cell
    a, _, c = LSTM_cell(x, initial_state=[a, c])
    # Apply Dense() to the hidden state output of LSTM_Cell
    output = densor(a)
    # Add the output to "outputs"
    outputs.append(output)
    
# Create model instance
model = Model(inputs=[inputs, a0, c0], outputs=outputs)
model.summary()

# %% [markdown]
# ## Compile and train the LSTM network

# %%
model.compile(optimizer=optimizers.Adam(learning_rate=1e-2), loss='categorical_crossentropy')

epochs = 150
patience = int(epochs / 10)
epochs_to_show = [0] + [i for i in range(patience - 1, epochs, patience)]
custom_verbose = CustomVerbose(epochs_to_show)
early_stopping = callbacks.EarlyStopping(monitor='loss', patience=patience, verbose=1)
a0 = np.zeros((X.shape[0], n_a))
c0 = np.zeros((X.shape[0], n_a))
history = model.fit([X, a0, c0], list(Y), epochs=epochs, verbose=0, callbacks=[custom_verbose, early_stopping])

# %%
plt.figure()
plt.plot(history.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# %% [markdown]
# ## Evaluate the LSTM network

# %% [markdown]
# - Implement a music inference model to sample a sequence of musical values.
# - Use the trained "LSTM_cell" and "densor" variables from the main model to generate a sequence of musical values.

# %%
# Length of the sequences in the corpus
Ty = Tx

# Define the input of the model
x0 = layers.Input(shape=(1, n_values))

# Define initial hidden state and cell state for the decoder LSTM
a0 = layers.Input(shape=(n_a,))
c0 = layers.Input(shape=(n_a,))
x, a, c = x0, a0, c0

# Create an empty list of "outputs" to later store the predicted values
outputs = []

# A KerasTensor cannot be used as input to a TensorFlow function. The function should be wrapped in a layer.
class MyLayer(layers.Layer):
    def call(self, output):
        # Select the next value according to "output" and set "x" to be the one-hot representation of the selected value
        x = tf.math.argmax(output, axis=-1)
        x = tf.one_hot(x, depth=n_values)

        return x

# Loop over Ty and generate a value at every time step
for t in range(Ty):
    # Perform one step of LSTM_cell
    a, _, c = LSTM_cell(x, initial_state=[a, c])
    # Apply Dense layer to the hidden state output of the LSTM_cell
    output = densor(a)
    # Append the prediction "output" to "outputs"
    outputs.append(output)
    x = MyLayer()(output)
    # Use RepeatVector(1) to convert x into a tensor with shape=(None, 1, 90)
    x = layers.RepeatVector(1)(x)
    
# Create model instance with the correct "inputs" and "outputs"
inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)

# %% [markdown]
# The inference model generates a sequence of musical values, then these values are post-processed into musical chords (meaning that multiple values or notes can be played at the same time).
#
# Most [computational music algorithms](https://en.wikipedia.org/wiki/Computer_music) use some post-processing because it's difficult to generate music that sounds good without it. The post-processing does things like clean up the generated audio by making sure the same sound is not repeated too many times, or that two successive notes are not too far from each other in [pitch](https://en.wikipedia.org/wiki/Pitch_(music)), and so on.
#
# It could be argued that many of these post-processing steps are hacks, however much of the music generation literature has also focused on hand-crafting post-processors, and a lot of the output quality depends on the quality of the post-processing and not just the quality of the model. Anyway, this post-processing makes a big difference, which is why it is also used in this implementation.

# %%
_ = generate_music(inference_model, indices_values, chords, output_file='/tmp/jazz_generator/my_music.midi')
mid2wav('/tmp/jazz_generator/my_music.midi', output_file='/tmp/jazz_generator/my_music.wav')
IPython.display.Audio('/tmp/jazz_generator/my_music.wav')
