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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TensorFlow Lite model for Cats and Dogs classification

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/TensorFlow-Data_and_Deployment/portfolio-6.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Create a TFLite model that can classify images of cats and dogs.

# %% [markdown]
# **TensorFlow Lite** (now rebranded as [LiteRT](https://developers.google.com/edge/litert/overview)) is an open-source deep learning framework developed by Google designed for on-device inference (edge computing). It enables developers to deploy trained machine learning models directly on resource-constrained devices such as mobile phones, embedded systems, IoT devices, and microcontrollers without relying on cloud connectivity.
#
# Key characteristics include:
#
# - **Optimization**: It converts standard TensorFlow models into a compact `.tflite` format using techniques like quantization and pruning to reduce model size and memory usage.
# - **Performance**: It ensures low latency and efficient execution by leveraging hardware acceleration and optimized kernels.
# - **Cross-Platform Support**: It runs on Android, iOS, embedded Linux, and microcontrollers with support for languages like Java, Swift, C++, and Python.
# - **Privacy & Connectivity**: By processing data locally, it enhances user privacy and functions without an internet connection. 
#
# Unlike the full TensorFlow framework, which is used for building and training models on high-performance servers, TensorFlow Lite is strictly for running inference on already trained models. 

# %% [markdown]
# ## Import libraries

# %%
import tensorflow as tf
import tensorflow_datasets as tfds
from keras import applications, Model, layers, Input
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import seaborn as sns
sns.set_style("whitegrid")
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

print(tf.config.list_physical_devices('GPU'))

# %% [markdown]
# ## Load the dataset

# %% [markdown]
# We use [TensorFlow Datasets](http://tensorflow.org/datasets) to load the cats and dogs dataset.
#
# The `tfds.load` method downloads and caches the data, and returns a `tf.data.Dataset` object. These objects provide powerful, efficient methods for manipulating data and piping it into your model.
#
# Since `"cats_vs_dog"` only has one defined split, `train`, we are going to divide that into (train, validation, test) with 80%, 10%, 10% of the data respectively.

# %%
(train_examples, validation_examples, test_examples), info = tfds.load('cats_vs_dogs',
                                                                       data_dir='/tmp/',
                                                                       with_info=True, 
                                                                       as_supervised=True, 
                                                                       split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'])

# %% [markdown]
# ## Data preprocessing

# %% [markdown]
# We use the `tf.image` module to format the images for the task, then we resize the images to a fixed input size, and rescale the input channels. Finally we shuffle and batch the data.

# %%
num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes
class_names = ['cat', 'dog']
image_size = (224, 224)
batch_size = 32

format_image = lambda image, label: (tf.image.resize(image, image_size), label)

train_ds = train_examples.shuffle(num_examples // 4).map(format_image).batch(batch_size).prefetch(1)
validation_ds = validation_examples.map(format_image).batch(batch_size).prefetch(1)
test_ds = test_examples.map(format_image).batch(1)

# %% [markdown]
# ## Visualize the dataset

# %%
images, labels = next(iter(train_ds))

fig, axs = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle('Random samples')

for i, ax in enumerate(axs.flatten()):
    ax.imshow(images[i] / 255.0)
    ax.set_title(class_names[labels[i].numpy()])
    ax.axis("off")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualize the class distribution

# %%
_, sizes = np.unique([labels.numpy() for _, labels in train_examples], return_counts=True)

fig, ax = plt.subplots()
ax.pie(sizes, textprops={'color': "w", 'fontsize': '12'}, autopct=lambda pct: "{:.2f}%\n({:d})".format(pct, round(pct/100 * sum(sizes))))
ax.legend(["0 (cats)", "1 (dogs)"])
ax.set_title("Class")
plt.show()

# %% [markdown]
# ## Build the model

# %% [markdown]
# All it takes is to put a linear classifier on top of the a pre-trained [MobileNet V2](https://arxiv.org/pdf/1801.04381) as image `feature_extractor`.
#
# For speed, we start out with a non-trainable `feature_extractor`.

# %%
input_shape = image_size + (3,)
MobileNetV2 = applications.MobileNetV2(include_top=False, input_shape=input_shape, pooling='avg')
MobileNetV2.trainable = False

inputs = Input(shape=input_shape)
x = applications.mobilenet_v2.preprocess_input(inputs) # Preprocess our images the same way the images used to train MobileNetV2 model were processed
x = MobileNetV2(x, training=False)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)
model.summary()

# %% [markdown]
# ## Compile and train the model

# %%
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

epochs = 5
history = model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_xticks(range(epochs))
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.legend(["Training", "Validation"])

ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_xticks(range(epochs))
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.legend(["Training", "Validation"])

plt.show()

# %% [markdown]
# ## Convert the model into a TFLite model

# %% [markdown]
# Load the TFLiteConverter with the Keras model.

# %%
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# %% [markdown]
# The simplest form of [post-training quantization](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization) quantizes weights from floating point to 8-bits of precision. This technique is enabled as an option in the TensorFlow Lite converter. At inference, weights are converted from 8-bits of precision to floating point and computed using floating-point kernels. This conversion is done once and cached to reduce latency.
#
# To further improve latency, hybrid operators [dynamically quantize](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization#dynamic_range_quantization) activations to 8-bits and perform computations with 8-bit weights and activations. This optimization provides latencies close to fully fixed-point inference. However, the outputs are still stored using floating point, so that the speedup with hybrid ops is less than a full fixed-point computation.

# %%
converter.optimizations = [tf.lite.Optimize.DEFAULT]


# %% [markdown]
# We can get further latency improvements, reductions in peak memory usage, and access to integer only hardware accelerators by making sure all model math is quantized using [integer quantization](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant). To do this, we need to measure the dynamic range of activations and inputs with a representative data set. You can simply create an [input data generator](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant#convert_using_float_fallback_quantization) and provide it to our converter.

# %%
def representative_data_gen():
    for input_value, _ in test_ds.take(100):
        yield [input_value]

converter.representative_dataset = representative_data_gen

# %% [markdown]
# The resulting model will be fully quantized but still take float input and output for convenience.
#
# Ops that do not have quantized implementations will automatically be left in floating point. This allows conversion to occur smoothly but may restrict deployment to accelerators that support float.
#
# To require the converter to only [output integer operations](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization#integer_only), one can specify:

# %%
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# %% [markdown]
# Finally, we convert the model and save it. The resulting model is a `.tflite` file that can be deployed to edge devices for inference.

# %%
tflite_model = converter.convert()
tflite_model_file = '/tmp/model.tflite'

with open(tflite_model_file, "wb") as f:
    f.write(tflite_model)

# %% [markdown]
# ## Evaluate the TFLite model

# %% [markdown]
# Load the TFLite model and allocate tensors.

# %%
interpreter = Interpreter(model_path=tflite_model_file)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]["index"]
output_details = interpreter.get_output_details()[0]["index"]

# %% [markdown]
# Gather results for the test images.

# %%
test_predictions = []
test_labels, test_images = [], []

for image, label in tqdm(test_ds):
    interpreter.set_tensor(input_details, image)
    interpreter.invoke()

    test_predictions.append(interpreter.get_tensor(output_details))
    test_labels.append(label.numpy()[0])
    test_images.append(image)

# %%
fig, axs = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle('Random samples')

for i, ax in enumerate(axs.flatten()):
    ax.imshow(test_images[i][0] / 255.0)
    ax.set_title("Prediction: " + class_names[np.argmax(test_predictions[i].squeeze())])
    ax.axis("off")

plt.tight_layout()
plt.show()

# %%
y_pred = [np.argmax(prediction.squeeze()) for prediction in test_predictions]

print(classification_report(test_labels, y_pred, digits=4))

ConfusionMatrixDisplay.from_predictions(test_labels, y_pred, display_labels=["0 (cat)", "1 (dog)"])
plt.grid(False)
plt.show()
