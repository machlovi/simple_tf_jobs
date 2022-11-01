
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.datasets import mnist
from time import perf_counter, sleep
current_dir=os.getcwd()
import urllib.request





(train_images, train_labels), (test_images, test_labels) = mnist.load_data(current_dir + './mnist.npz')

# reshape images to specify that it's a single channel
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# %% [markdown]
# We scale these values to a range of 0 to 1 before feeding to the neural network model. For this, we divide the values by 255. It's important that the *training set* and the *testing set* are preprocessed in the same way:

# %%
def preprocess_images(imgs): # should work for both a single image and multiple images
    sample_img = imgs if len(imgs.shape) == 2 else imgs[0]
    assert sample_img.shape in [(28, 28, 1), (28, 28)], sample_img.shape # make sure images are 28x28 and single-channel (grayscale)
    return imgs / 255.0

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

# %% [markdown]
# Display the first 5 images from the *training set* and display the class name below each image. Verify that the data is in the correct format and we're ready to build and train the network.

# %% [markdown]
# ### Build the model

# Building the neural network requires configuring the layers of the model, then compiling the model. In many cases, this can be reduced to simply stacking together layers:

i=0
while i<5:
    start = perf_counter()
    model = keras.Sequential()
    # 32 convolution filters used each of size 3x3
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    # 64 convolution filters used each of size 3x3
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # choose the best features via pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # randomly turn neurons on and off to improve convergence
    model.add(Dropout(0.25))
    # flatten since too many dimensions, we only want a classification output
    model.add(Flatten())
    # fully connected to get all relevant data
    model.add(Dense(128, activation='relu'))
    # one more dropout
    model.add(Dropout(0.5))
    # output a softmax to squash the matrix into output probabilities
    model.add(Dense(10, activation='softmax'))

    # %%
    model.compile(optimizer=tf.keras.optimizers.Adam(), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # %%
    history = model.fit(train_images, train_labels, epochs=1)

    i+=1



    end = perf_counter()

    print(f"Time taken to execute code : {end-start}")

with open('time.txt', 'a') as f:
    print(end-start, file=f)
# %%

