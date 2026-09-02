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
# # Trigger Word Detection

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/Deep_Learning_Specialization/portfolio-12.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Train a Recurrent Neural Network (RNN) to detect a trigger word in audio sequences.

# %% [markdown]
# [Trigger word detection](https://www.ijert.org/research/trigger-word-recognition-using-lstm-IJERTV9IS060092.pdf) is the technology that allows devices like Amazon Alexa, Google Home, and Apple Siri to activate when they hear a specific word. For this model, the trigger word will be "activate", every time it hears this specific word, it will make a buzzer sound.
#
# A [Recurrent Neural Network (RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network) utilizes recurrent connections, where the output of a neuron at one time step is fed back as input to the network at the next time step. This enables RNNs to capture temporal dependencies and patterns within sequences. The fundamental building block of RNNs is the recurrent unit, which maintains a hidden state, a form of memory that is updated at each time step based on the current input and the previous hidden state. This feedback mechanism allows the network to learn from past inputs and incorporate that knowledge into its current processing.

# %% [markdown]
# ## Import libraries

# %%
from IPython import display
from scipy.io import wavfile
import os
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
from keras import layers, optimizers, Model

# %% [markdown]
# ## Download the dataset

# %% language="bash"
#
# if [ -e "/tmp/word_detection.zip" ]; then
#     echo "word_detection.zip already exists!"
# else
#     gdown 1U-LOR2zg_yi1pLVXtyTyrB6igp29KCIk -O /tmp/
# fi
#
# unzip -qn /tmp/word_detection.zip -d /tmp

# %% [markdown]
# ## Load the dataset

# %%
display.Audio("/tmp/data/positives/1.wav")

# %%
display.Audio("/tmp/data/negatives/1.wav")

# %%
display.Audio("/tmp/data/backgrounds/1.wav")


# %% [markdown]
# The audio was sampled at 44.1 kHz, therefore a 10 second audio clip is represented by 441,000 numbers. In order to help the sequence model learn more easily to detect trigger words, a [spectrogram](https://en.wikipedia.org/wiki/Spectrogram) of the audio is computed. The spectrogram tells us how many different frequencies are present in an audio clip at any given time. 

# %%
def graph_spectrogram(wav_file):
    _, data = wavfile.read(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim

    if nchannels == 1:
        spectrum, _, _, _ = plt.specgram(data, Fs=fs, NFFT=nfft, noverlap=noverlap)
    elif nchannels == 2:
        spectrum, _, _, _ = plt.specgram(data[:, 0], Fs=fs, NFFT=nfft, noverlap=noverlap)
        
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    return data, spectrum


# %%
data, spectrum = graph_spectrogram("/tmp/data/example_train.wav")
display.Audio("/tmp/data/example_train.wav")

# %% [markdown]
# The color of the spectrogram shows the degree to which different frequencies are present in the audio at different points in time. Yellow means that a certain frequency is more present in the audio clip, blue indicates less active frequencies. 
#
# The size of the output spectrogram depends on the hyperparameters of the spectrogram software and the length of the input. We will work with 10 second audio clips as the "standard length" for the training examples. The spectrogram will be the input to the network, the number of time steps of the spectrogram will be 5511, so $T_x$ = 5511.

# %%
print("Time steps in audio recording before spectrogram:", data[:, 0].shape)
print("Time steps in input after spectrogram:", spectrum.shape)

# %% [markdown]
# The output of the model will divide 10 seconds into 1375 units. For each time step, the model predicts whether someone recently finished saying the trigger word "activate", so $T_y$ = 1375. These values were chosen within the standard range used for speech systems.

# %%
Tx = 5511 # The number of time steps input to the model from the spectrogram
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram
Ty = 1375 # The number of time steps in the output of our model

# %% [markdown]
# Audio segments are loaded using [pydub](https://github.com/jiaaro/pydub), which uses 1 ms as the discretization interval, which is why a 10 second audio clip is always represented using 10,000 steps.

# %%
positives = []
backgrounds = []
negatives = []

for filename in os.listdir("/tmp/data/positives"):
    if filename.endswith("wav"):
        activate = AudioSegment.from_wav("/tmp/data/positives/" + filename)
        positives.append(activate)

for filename in os.listdir("/tmp/data/backgrounds"):
    if filename.endswith("wav"):
        background = AudioSegment.from_wav("/tmp/data/backgrounds/" + filename)
        backgrounds.append(background)

for filename in os.listdir("/tmp/data/negatives"):
    if filename.endswith("wav"):
        negative = AudioSegment.from_wav("/tmp/data/negatives/" + filename)
        negatives.append(negative)


# %% [markdown]
# ## Preprocess the dataset

# %% [markdown]
# Create recordings with a mix of positive words ("activate") and negative words (random words other than "activate") on different background sounds.
#
# Process to synthesize an audio clip:
#
# - Pick a random 10 second background audio clip.
# - Randomly insert 0-4 audio clips of "activate" into this clip.
# - Randomly insert 0-2 audio clips of negative words into this clip.
#
# Since we have synthesized the word "activate" into the background clip, we know exactly when in the 10 second clip the word "activate" appears. This makes it easier to generate the labels $y^{\langle t \rangle}$.

# %%
def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.
    
    Arguments:
    segment_ms -- The duration of the audio clip in ms ("ms" stands for "milliseconds").
    
    Returns:
    segment_time -- A tuple of (segment_start, segment_end) in ms.
    """
    
    segment_start = np.random.randint(low=0, high=10000 - segment_ms) # Make sure segment doesn't run past the 10sec background 
    segment_end = segment_start + segment_ms - 1
    
    return segment_start, segment_end

def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.
    
    Arguments:
    segment_time -- A tuple of (segment_start, segment_end) for the new segment.
    previous_segments -- A list of tuples of (segment_start, segment_end) for the existing segments.
    
    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise.
    """
    
    segment_start, segment_end = segment_time
    
    # Initialize overlap as a "False" flag.
    overlap = False
    
    # Loop over the previous_segments start and end times.
    # Compare start/end times and set the flag to True if there is an overlap
    for previous_start, previous_end in previous_segments: 
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True
            break

    return overlap

def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the 
    audio segment does not overlap with existing segments.
    
    Arguments:
    background -- A 10 second background audio recording.  
    audio_clip -- The audio clip to be inserted/overlaid. 
    previous_segments -- Times where audio segments have already been placed.
    
    Returns:
    new_background -- The updated background audio.
    """
    
    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)
    
    # Use get_random_time_segment() function to pick a random time segment onto which to insert 
    # the new audio clip.
    segment_time = get_random_time_segment(segment_ms)
    
    # Check if the new segment_time overlaps with one of the previous_segments. If so, keep 
    # picking new segment_time at random until it doesn't overlap. To avoid an endless loop, retry 5 times
    retry = 5 
    
    while is_overlapping(segment_time, previous_segments) and retry >= 0:
        segment_time = get_random_time_segment(segment_ms)
        retry = retry - 1

    # If last try is not overlaping, insert it to the background
    if not is_overlapping(segment_time, previous_segments):
        # Append the new segment_time to the list of previous_segments
        previous_segments.append(segment_time)
        # Superpose audio segment and background
        new_background = background.overlay(audio_clip, position=segment_time[0])
    else:
        new_background = background
        segment_time = (10000, 10000)
    
    return new_background, segment_time

def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment 
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 following labels should be ones.
    
    Arguments:
    y -- Numpy array of shape (1, Ty), the labels of the training example.
    segment_end_ms -- The end time of the segment in ms.
    
    Returns:
    y -- Updated labels.
    """
    
    _, Ty = y.shape
    
    # Duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    
    if segment_end_y < Ty:
        # Add 1 to the correct index in the background label (y)
        for i in range(segment_end_y + 1, segment_end_y + 51):
            if i < Ty:
                y[0, i] = 1
    
    return y

def create_training_example(background, activates, negatives, Ty, output_file):
    """
    Creates a training example with a given background, activates, and negatives.
    
    Arguments:
    background -- A 10 second background audio recording.
    activates -- A list of audio segments of the word "activate".
    negatives -- A list of audio segments of random words that are not "activate".
    Ty -- The number of time steps in the output.

    Returns:
    x -- The spectrogram of the training example.
    y -- The label at each time step of the spectrogram.
    """
    
    # Make background quieter
    background = background - 20

    # Initialize y (label vector) of zeros
    y = np.zeros((1, Ty))

    # Initialize segment times as empty list
    previous_segments = []
    
    # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]
    
    # Loop over randomly selected "activate" clips and insert in background
    for random_activate in random_activates:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        _, segment_end = segment_time
        # Insert labels in "y" at segment_end
        y = insert_ones(y, segment_end)

    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    # Loop over randomly selected negative clips and insert in background
    for random_negative in random_negatives:
        # Insert the audio clip on the background 
        background, _ = insert_audio_clip(background, random_negative, previous_segments)
    
    # Standardize the volume of the audio clip 
    background = background.apply_gain(-20.0 - background.dBFS)

    # Export new training example 
    background.export(output_file, format="wav")
    
    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    _, x = graph_spectrogram(output_file)
    
    return x, y


# %%
plt.subplot(2, 1, 1)
x, y = create_training_example(backgrounds[0], positives, negatives, Ty, output_file="/tmp/data/train.wav")

plt.subplot(2, 1, 2)
plt.plot(y[0])
plt.xlabel("Time")
plt.grid("both")

plt.tight_layout()
plt.show()

display.Audio("/tmp/data/train.wav")

# %% [markdown]
# Create a smaller training set of 32 examples:

# %%
np.random.seed(4543)

nsamples = 32
X = []
Y = []

for i in range(0, nsamples):
    x, y = create_training_example(backgrounds[i % 2], positives, negatives, Ty, output_file="/tmp/data/train.wav")
    plt.close()
    X.append(x.swapaxes(0, 1))
    Y.append(y.swapaxes(0, 1))
    
X = np.array(X)
Y = np.array(Y)

# %% [markdown]
# ## Build a RNN

# %%
inputs = layers.Input(shape=(Tx, n_freq))

# CONV layer
# Conv1D with 196 units, kernel size of 15 and stride of 4
x = layers.Conv1D(filters=196, kernel_size=15, strides=4)(inputs)
# Batch normalization
x = layers.BatchNormalization()(x)
# ReLu activation
x = layers.Activation('relu')(x)
# Dropout
x = layers.Dropout(0.8)(x)                                  

# First GRU Layer
# GRU (128 units and return the sequences)
x = layers.GRU(units=128, return_sequences=True)(x) 
# Dropout
x = layers.Dropout(0.8)(x)  
# Batch normalization
x = layers.BatchNormalization()(x)                           

# Second GRU Layer
# GRU (use 128 units and return the sequences)
x = layers.GRU(units=128, return_sequences=True)(x) 
# Dropout
x = layers.Dropout(0.8)(x)         
# Batch normalization
x = layers.BatchNormalization()(x)
# Dropout
x = layers.Dropout(0.8)(x)                                  

# Time-distributed dense layer
# TimeDistributed with sigmoid activation 
outputs = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"))(x) 

model = Model(inputs=inputs, outputs=outputs)
model.summary()

# %% [markdown]
# ## Compile and train the RNN

# %% [markdown]
# Trigger word detection takes a long time to train, to save time, we will load a model which was trained using a large training set of about 4000 examples.

# %%
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=1e-6), metrics=["accuracy"])
model.load_weights('/tmp/data/model.h5')


# %% [markdown]
# ## Evaluate the RNN

# %%
def detect_triggerword(filename):
    """Runs audio (saved in a wav file) through the network"""

    plt.subplot(2, 1, 1)
    
    # Correct the amplitude of the input file before prediction 
    audio_clip = AudioSegment.from_wav(filename)
    audio_clip = audio_clip.apply_gain(-20.0 - audio_clip.dBFS)
    audio_clip.export("/tmp/data/tmp.wav", format="wav")
    filename = "/tmp/data/tmp.wav"

    _, x = graph_spectrogram(filename)
    # The spectrogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x, verbose=0)
    
    plt.subplot(2, 1, 2)
    plt.plot(predictions[0, :, 0])
    plt.xlim(0, len(predictions[0, :, 0]))
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.grid("both")

    plt.tight_layout()
    plt.show()
    
    return predictions

def chime_on_activate(filename, output_file, predictions, threshold):
    """
    Triggers a "chiming" sound to play when the probability is above a certain threshold.
    """

    chime_file = "/tmp/data/chime.wav"
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]

    # Initialize the number of consecutive output steps to 0
    consecutive_timesteps = 0

    # Loop over the output steps in the y
    for i in range(Ty):
        # Increment consecutive output steps
        consecutive_timesteps += 1
        # If prediction is higher than the threshold and more than 20 consecutive output steps have passed
        if consecutive_timesteps > 20:
            # Superpose audio and background using pydub
            audio_clip = audio_clip.overlay(chime, position=((i / Ty) * audio_clip.duration_seconds) * 1000)
            # Reset consecutive output steps to 0
            consecutive_timesteps = 0
        # If amplitude is smaller than the threshold reset the consecutive_timesteps counter
        if predictions[0, i, 0] < threshold:
            consecutive_timesteps = 0
        
    audio_clip.export(output_file, format='wav')


# %%
filename = "/tmp/data/dev/1.wav"
output_file = "/tmp/data/prediction_1.wav"
prediction = detect_triggerword(filename)
chime_on_activate(filename, output_file, prediction, threshold=0.5)
display.Audio(output_file)

# %%
filename = "/tmp/data/dev/2.wav"
output_file = "/tmp/data/prediction_2.wav"
prediction = detect_triggerword(filename)
chime_on_activate(filename, output_file, prediction, threshold=0.5)
display.Audio(output_file)
