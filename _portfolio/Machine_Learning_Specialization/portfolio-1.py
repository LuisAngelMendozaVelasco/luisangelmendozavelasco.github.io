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
# # Content-Based Filtering

# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/LuisAngelMendozaVelasco/luisangelmendozavelasco.github.io/blob/master/_portfolio/Machine_Learning_Specialization/portfolio-1.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Run in Google Colab</a>

# %% [markdown]
# **Objective**: Build a movie recommendation system using content-based filtering.

# %% [markdown]
# [Content-based filtering](https://developers.google.com/machine-learning/recommendation/content-based/basics) is a technique that recommends items similar to a user's preferences by analyzing item features and user interactions. It uses item attributes to provide recommendations, focusing on the features of items a user likes to suggest similar ones. For example, if a user enjoys a particular movie, the system will recommend other movies with similar attributes, such as genre, director, or cast. This method does not require user data from other users, making it effective for niche markets or when user data is limited.

# %% [markdown]
# ## Import libraries

# %%
import numpy as np
import pickle
import csv
from collections import defaultdict
import tabulate
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import layers, Sequential, Input, Model, callbacks

sns.set_style("whitegrid")


# %% [markdown]
# ## Download the dataset

# %% language="bash"
#
# files=("content_item_train.csv" "content_item_train_header.txt" "content_item_vecs.csv" "content_movie_list.csv" \
#     "content_user_to_genre.pickle" "content_user_train.csv" "content_user_train_header.txt" "content_y_train.csv")
#
# for file in "${files[@]}"; do
#     wget -nc --progress=bar:force:noscroll https://raw.githubusercontent.com/LuisAngelMendozaVelasco/Machine_Learning_Specialization/refs/heads/main/Unsupervised_Learning-Recommenders-Reinforcement_Learning/Week2/Labs/data/$file -P /tmp
# done

# %% [markdown]
# ## Load the dataset

# %% [markdown]
# The dataset is derived from the [MovieLens ml-latest-small dataset](https://grouplens.org/datasets/movielens/latest/). The original dataset has 9000 movies rated by 600 users with ratings on a scale of 0.5 to 5 in 0.5 step increments. The dataset has been reduced in size to focus on movies from the years after 2000 and on popular genres, it has $n_u$ = 395 users and $n_m$ = 395 movies. For each movie, the dataset provides a movie title, release date, and one or more genres. For example "Toy Story 3" was released in 2010 and has several genres: "Adventure|Animation|Children|Comedy|Fantasy|IMAX". This dataset contains little information about users other than their ratings.
#
# The movie content provided to the network is a combination of the original data and some "engineered features". The original features are the year the movie was released and the movie's genre (14 different genres) presented as a one-hot vector. The engineered feature is an average rating derived from the user ratings. Movies with multiple genre have a training vector per genre.
#
# The user content is composed of only "engineered features", a per genre average rating is computed per user. Additionally, a user id, rating count and rating average are available, but are not included in the training or prediction content. They are useful in interpreting data.
#
# The training set consists of all the ratings made by the users in the dataset. The user and movie/item vectors are presented to the neural network together as a training set. The user vector is the same for all the movies rated by the user.

# %%
def load_data():
    item_train = np.genfromtxt('/tmp/content_item_train.csv', delimiter=',')
    user_train = np.genfromtxt('/tmp/content_user_train.csv', delimiter=',')
    y_train = np.genfromtxt('/tmp/content_y_train.csv', delimiter=',')

    with open('/tmp/content_item_train_header.txt', newline='') as f:    
        item_features = list(csv.reader(f))[0] # csv reader handles quoted strings better
        
    with open('/tmp/content_user_train_header.txt', newline='') as f:
        user_features = list(csv.reader(f))[0]

    item_vecs = np.genfromtxt('/tmp/content_item_vecs.csv', delimiter=',')
    movie_dict = defaultdict(dict)
    count = 0

    with open('/tmp/content_movie_list.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')

        for line in reader:
            if count == 0: 
                count +=1 # skip header
            else:
                count +=1
                movie_id = int(line[0])  
                movie_dict[movie_id]["title"] = line[1]  
                movie_dict[movie_id]["genres"] = line[2]  

    with open('/tmp/content_user_to_genre.pickle', 'rb') as f:
        user_to_genre = pickle.load(f)

    return (item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre)


# %%
item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict, user_to_genre = load_data()

# Set configuration variables
num_user_features = user_train.shape[1] - 3 # Remove userid, rating count and average rating during training
num_item_features = item_train.shape[1] - 1 # Remove movie id at train time
uvs = 3 # User genre vector start
ivs = 3 # Item genre vector start
u_s = 3 # Start of columns to use in training, user
i_s = 1 # Start of columns to use in training, items
scaledata = True # Applies the standard scalar to data if true
print(f"Number of training vectors: {len(item_train)}")


# %% [markdown]
# ## Visualize the dataset

# %%
def split_str(ifeatures, smax):
    ofeatures = []

    for s in ifeatures:
        if ' ' not in s:  # skip string that already have a space            
            if len(s) > smax:
                mid = int(len(s) / 2)
                s = s[:mid] + " " + s[mid:]

        ofeatures.append(s)

    return ofeatures


def pprint_train(x_train, features, vs, u_s, maxcount=5, user=True):
    """ Prints user_train or item_train nicely """

    if user:
        flist = [".0f", ".0f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f", ".1f"]
    else:
        flist = [".0f", ".0f", ".1f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f", ".0f"]

    head = features[:vs]

    if vs < u_s: print("error, vector start {vs} should be greater then user start {u_s}")

    for i in range(u_s):
        head[i] = "[" + head[i] + "]"

    genres = features[vs:]
    hdr = head + genres
    disp = [split_str(hdr, 5)]
    count = 0

    for i in range(0, x_train.shape[0]):
        if count == maxcount: break

        count += 1
        disp.append([x_train[i, 0].astype(int),
                     x_train[i, 1].astype(int),
                     x_train[i, 2].astype(float),
                     *x_train[i, 3:].astype(float)])

    table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow", floatfmt=flist, numalign='center')

    return table


# %% [markdown]
# We can see that movie 6874 is an action movie released in 2003. User 2 rates action movies as 3.9 on average. Further, movie 6874 was also listed in the Crime and Thriller genre. MovieLens users gave the movie an average rating of 4. A training example consists of a row from both tables and a rating from y_train.

# %%
pprint_train(user_train, user_features, uvs, u_s, maxcount=5)

# %%
pprint_train(item_train, item_features, ivs, i_s, maxcount=5, user=False)

# %%
print(f"y_train[:5]: {y_train[:5]}")

# %% [markdown]
# ## Preprocess the dataset

# %%
scalerItem = StandardScaler()
scalerItem.fit(item_train)
item_train = scalerItem.transform(item_train)

scalerUser = StandardScaler()
scalerUser.fit(user_train)
user_train = scalerUser.transform(user_train)

scalerY = MinMaxScaler((-1, 1))
scalerY.fit(y_train.reshape(-1, 1))
y_train = scalerY.transform(y_train.reshape(-1, 1))

item_train, item_right = train_test_split(item_train, train_size=0.8, shuffle=True, random_state=1)
item_validation, item_test = train_test_split(item_right, train_size=0.5, shuffle=True, random_state=1)

user_train, user_right = train_test_split(user_train, train_size=0.8, shuffle=True, random_state=1)
user_validation, user_test = train_test_split(user_right, train_size=0.5, shuffle=True, random_state=1)

y_train, y_right = train_test_split(y_train, train_size=0.8, shuffle=True, random_state=1)
y_validation, y_test = train_test_split(y_right, train_size=0.5, shuffle=True, random_state=1)

# %%
print("Number of training samples:", item_train.shape)
print("Number of validation samples:", item_validation.shape)
print("Number of test samples:", item_test.shape)


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
# ## Build the model

# %% [markdown]
# The Neural Network will have two identical networks that are combined by a dot product. Note that these networks do not need to be the same. If the user content was substantially larger than the movie content, we might elect to increase the complexity of the user network relative to the movie network. In this case, the content is similar, so the networks are the same.

# %%
class L2Normalization(layers.Layer):
    def call(self, x):
        return tf.linalg.l2_normalize(x, axis=1)


# %%
num_outputs = 32
tf.random.set_seed(1)

user_NN = Sequential([layers.Dense(256, activation='relu'),
                      layers.Dense(128, activation='relu'),
                      layers.Dense(num_outputs, activation='linear'),
                      L2Normalization()])

item_NN = Sequential([layers.Dense(256, activation='relu'),
                      layers.Dense(128, activation='relu'),
                      layers.Dense(num_outputs, activation='linear'),
                      L2Normalization()])

# Create the user input and point to the base network
input_user = Input(shape=(num_user_features,))
vu = user_NN(input_user)

# Create the item input and point to the base network
input_item = Input(shape=(num_item_features,))
vm = item_NN(input_item)

# Compute the dot product of the two vectors vu and vm
output = layers.Dot(axes=1)([vu, vm])

# Specify the inputs and output of the model
model = Model([input_user, input_item], output)

model.summary()

# %% [markdown]
# ## Compile and train the model

# %%
model.compile(optimizer="adam", loss="mean_squared_error")

epochs = 100
patience = int(epochs / 10)
epochs_to_show = [0] + [i for i in range(patience - 1, epochs, patience)]
custom_verbose = CustomVerbose(epochs_to_show)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
history = model.fit([user_train[:, u_s:], item_train[:, i_s:]], y_train, epochs=epochs, verbose=0, validation_data=([user_validation[:, u_s:], item_validation[:, i_s:]], y_validation), callbacks=[custom_verbose, early_stopping])

# %%
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Training", "Validation"])
plt.show()

# %% [markdown]
# ## Evaluate the model

# %%
test_loss = model.evaluate([user_test[:, u_s:], item_test[:, i_s:]], y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}")


# %% [markdown]
# ### Predictions for a new user

# %%
def gen_user_vecs(user_vec, num_items):
    """ Given a user vector return:
        user predict maxtrix to match the size of item_vecs
    """
    
    user_vecs = np.tile(user_vec, (num_items, 1))

    return user_vecs


# Predict on  everything, filter on print/use
def predict_uservec(user_vecs, item_vecs, model, u_s, i_s, scaler, ScalerUser, ScalerItem, scaledata=False):
    """ Given a user vector, does the prediction on all movies in item_vecs returns
        an array predictions sorted by predicted rating,
        arrays of user and item, sorted by predicted rating sorting index
    """

    if scaledata:
        scaled_user_vecs = ScalerUser.transform(user_vecs)
        scaled_item_vecs = ScalerItem.transform(item_vecs)
        y_p = model.predict([scaled_user_vecs[:, u_s:], scaled_item_vecs[:, i_s:]], verbose=0)
    else:
        y_p = model.predict([user_vecs[:, u_s:], item_vecs[:, i_s:]], verbose=0)

    y_pu = scaler.inverse_transform(y_p)

    if np.any(y_pu < 0) : 
        print("Error, expected all positive predictions")

    sorted_index = np.argsort(-y_pu, axis=0).reshape(-1).tolist()  #negate to get largest rating first
    sorted_ypu = y_pu[sorted_index]
    sorted_items = item_vecs[sorted_index]
    sorted_user = user_vecs[sorted_index]

    return (sorted_index, sorted_ypu, sorted_items, sorted_user)


def print_pred_movies(y_p, user, item, movie_dict, maxcount=10):
    """ Print results of prediction of a new user. inputs are expected to be in
        sorted order, unscaled. 
    """
    
    count = 0
    movies_listed = defaultdict(int)
    disp = [["y_p", "movie id", "rating ave", "title", "genres"]]

    for i in range(0, y_p.shape[0]):
        if count == maxcount:
            break

        count += 1
        movie_id = item[i, 0].astype(int)

        if movie_id in movies_listed:
            continue

        movies_listed[movie_id] = 1
        disp.append([y_p[i, 0], item[i, 0].astype(int), item[i, 2].astype(float),
                    movie_dict[movie_id]['title'], movie_dict[movie_id]['genres']])

    table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow")

    return table


# %% [markdown]
# We'll create a new user and have the model suggest movies for that user. Note that ratings are between 0.5 and 5.0, in half-step increments.

# %%
new_user_id = 5000
new_rating_ave = 1.0
new_action = 1
new_adventure = 1
new_animation = 1
new_childrens = 1
new_comedy = 5
new_crime = 1
new_documentary = 1
new_drama = 1
new_fantasy = 1
new_horror = 1
new_mystery = 1
new_romance = 5
new_scifi = 5
new_thriller = 1
new_rating_count = 3

user_vec = np.array([[new_user_id, new_rating_count, new_rating_ave,
                      new_action, new_adventure, new_animation, new_childrens,
                      new_comedy, new_crime, new_documentary,
                      new_drama, new_fantasy, new_horror, new_mystery,
                      new_romance, new_scifi, new_thriller]])

# %% [markdown]
# Let's look at the top-rated movies for the new user. Recall, the user vector had genres that favored Comedy, Romance and Science Fiction. Below, we'll use a set of movie/item vectors, **item_vecs**, that have a vector for each movie in the training/test set. This is matched with the user vector above and the scaled vectors are used to predict ratings for all the movies for our new user.

# %%
# Generate and replicate the user vector to match the number movies in the data set.
user_vecs = gen_user_vecs(user_vec, len(item_vecs))

# Scale the vectors and make predictions for all movies. Return results sorted by rating.
sorted_index, sorted_ypu, sorted_items, sorted_user = predict_uservec(user_vecs, item_vecs, model, u_s, i_s, 
                                                                      scalerY, scalerUser, scalerItem, scaledata=scaledata)

print_pred_movies(sorted_ypu, sorted_user, sorted_items, movie_dict, maxcount=10)


# %% [markdown]
# ### Predictions for an existing user

# %% [markdown]
# Let's look at the predictions for "user 36", one of the users in the dataset. We can compare the predicted ratings with the model's ratings. Note that movies with multiple genre's show up multiple times in the training data. For example,"The Time Machine" has three genre's: Adventure, Action, Sci-Fi.

# %%
def get_user_vecs(user_id, user_train, item_vecs, user_to_genre):
    """ Given a user_id, return:
        user train/predict matrix to match the size of item_vecs
        y vector with ratings for all rated movies and 0 for others of size item_vecs
    """

    if user_id not in user_to_genre:
        print("error: unknown user id")

        return None
    else:
        user_vec_found = False

        for i in range(len(user_train)):
            if user_train[i, 0] == user_id:
                user_vec = user_train[i]
                user_vec_found = True
                break

        if not user_vec_found:
            print("error in get_user_vecs, did not find uid in user_train")

        num_items = len(item_vecs)
        user_vecs = np.tile(user_vec, (num_items, 1))

        y = np.zeros(num_items)

        for i in range(num_items):  # walk through movies in item_vecs and get the movies, see if user has rated them
            movie_id = item_vecs[i, 0]

            if movie_id in user_to_genre[user_id]['movies']:
                rating = user_to_genre[user_id]['movies'][movie_id]
            else:
                rating = 0

            y[i] = rating

    return (user_vecs, y)


def print_existing_user(y_p, y, user, items, item_features, ivs, uvs, movie_dict, maxcount=10):
    """ Print results of prediction a user who was in the datatbase. inputs are expected to be in sorted order, unscaled. """

    count = 0
    movies_listed = defaultdict(int)
    disp = [["y_p", "y", "user", "user genre ave", "movie rating ave", "title", "genres"]]
    listed = []
    count = 0

    for i in range(0, y.shape[0]):
        if y[i, 0] != 0:
            if count == maxcount:
                break

            count += 1
            movie_id = items[i, 0].astype(int)

            offset = np.where(items[i, ivs:] == 1)[0][0]
            genre_rating = user[i, uvs + offset]
            genre = item_features[ivs + offset]
            disp.append([y_p[i, 0], y[i, 0],
                        user[i, 0].astype(int), # userid
                        genre_rating.astype(float),
                        items[i, 2].astype(float), # movie average rating
                        movie_dict[movie_id]['title'], genre])

    table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow", floatfmt=[".1f", ".1f", ".0f", ".2f", ".2f"])
    
    return table


# %%
uid = 36 
# Form a set of user vectors. This is the same vector, transformed and repeated.
user_vecs, y_vecs = get_user_vecs(uid, scalerUser.inverse_transform(user_train), item_vecs, user_to_genre)

# Scale the vectors and make predictions for all movies. Return results sorted by rating.
sorted_index, sorted_ypu, sorted_items, sorted_user = predict_uservec(user_vecs, item_vecs, model, u_s, i_s, scalerY, 
                                                                      scalerUser, scalerItem, scaledata=scaledata)
sorted_y = y_vecs[sorted_index]

# Print sorted predictions
print_existing_user(sorted_ypu, sorted_y.reshape(-1, 1), sorted_user, sorted_items, item_features, ivs, uvs, movie_dict, maxcount=10)
