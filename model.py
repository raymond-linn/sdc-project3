from keras.models import Sequential
from keras.layers import ZeroPadding2D, Convolution2D, Flatten, Dense
from keras.layers import Dropout, MaxPooling2D, ELU, Lambda
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import cv2

# tensorflow backend error of using dropout and maxpooling fix AttributeError:
# module 'tensorflow.python' has no attribute 'control_flow_ops'
# ref: https://github.com/udacity/sdc-issue-reports/issues/125
import tensorflow as tf
tf.python.control_flow_ops = tf

# some constants which can be tweaked or tuned through out the experiments
# pixel to be cropped from the top of the input image (160px X 320px)
CROP_FROM_TOP = 40
# crop from the bottom but (0, 0) is left top corner and (160px - 25px)
CROP_FROM_BOTTOM = 135

BRIGHTNESS_COSNT = 0.25  # factor for adjustment
RESIZE_DIM = (64, 64)  # resizing dimension of the image

BATCH_SIZE = 32   # batch size
DRIVING_LOG_FILE = 'data/driving_log.csv'   # log file
ANGLE_ADJUSTMENT = 0.25   # angle adjustment factor


# all the images are preprocessed by cropping, resizing and normalizing
# use this function from drive.py to be the same as training input to the model
def process_image(image):
    # crop the 40 px from top and 25 px from the bottom
    processed = image[CROP_FROM_TOP:CROP_FROM_BOTTOM, 0:320]    
    # resize it to 64px by 64px
    processed = cv2.resize(processed, RESIZE_DIM, interpolation=cv2.INTER_AREA)
    return processed


# adjust brightness randomly
def adjust_brightness_image(image):
    adjusted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = BRIGHTNESS_COSNT + np.random.uniform()
    adjusted[:, :, 2] = adjusted[:, :, 2] * brightness
    return cv2.cvtColor(adjusted, cv2.COLOR_HSV2RGB)


# image augmentation
def augment_image(row):

    # save angle
    angle = row['steering']

    # get random image from the center, left or right
    random_image = np.random.choice(['center', 'left', 'right'])

    # adjust the angle with respect to center, left or righ image
    if random_image == 'left':
        adjust_angle = ANGLE_ADJUSTMENT
    elif random_image == 'right':
        adjust_angle = -ANGLE_ADJUSTMENT
    else:
        adjust_angle = 0

    angle = angle + adjust_angle

    # load image
    image = cv2.imread("data/" + row[random_image].strip())
    # change color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # adjust brightness randomly
    image = adjust_brightness_image(image)
    # process the image to be cropped an resized 
    image = process_image(image) 
    image = np.array(image)   

    # flip image 50% of the time
    if np.random.randint(2) == 0:
        angle = -angle
        image = cv2.flip(image, 1)

    # return image and angle
    return image, angle


# loading driving_log.csv into pandas.dataframe for getting images
def get_dataframe_and_shuffle(filename):
    # read only four columns of all the rows from the driving_log.csv file
    df = pd.read_csv(filename, usecols=[0, 1, 2, 3])

    # shuffle all data frame rows from
    # http://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    df = df.sample(frac=1).reset_index(drop=True)
    # and return data frame
    return df


# function to return generator tuple for keras model.fit_generator to use
# this function is used for training and validation generation
# credit to cohorts from Oct and Nov, from slack chanels and carnd forums
def generate_batch_data_from_dataframe(df, batch_size=BATCH_SIZE):

    num_rows = df.shape[0]
    batch_per_epoch = num_rows//batch_size

    i_batch = 0
    while 1:
        # set the start and end of each batch eg. 0...31, 32...64, etc. "i_batch" needs to be incremented
        # by 0, 1, 2,... to move the batch
        start = i_batch * batch_size
        end = start + batch_size - 1

        # initialize images and steering angles varaibles that will be filled with
        # augment_image function
        # initialize and populate features and labels 
        X_input = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
        y_input = np.zeros((batch_size,), dtype=np.float32)

        # loop through the rows of data frame
        j = 0
        for idx, row in df.loc[start:end].iterrows():
            # load image and augment image from the row data
            X_input[j], y_input[j] = augment_image(row)
            j += 1

        i_batch += 1

        # reset to start over df again
        if i_batch == batch_per_epoch - 1:
            i_batch = 0

        # and return the python generator for X_input and y_input using yield keyword
        yield X_input, y_input


# model structure
# https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
def get_model():

    model = Sequential()

    # new model
    # Normalizing the input using keras Lambda 
    model.add(Lambda(lambda x: x/255. - 0.5, input_shape=(64, 64, 3)))
    
    model.add(ZeroPadding2D((1, 1), input_shape=(64, 64, 3)))
    # Layer CNN1
    model.add(Convolution2D(32, 3, 3))
    model.add(ELU())

    model.add(ZeroPadding2D((1, 1)))
    # Layer CNN2
    model.add(Convolution2D(32, 3, 3))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(ZeroPadding2D((1, 1), input_shape=(64, 64, 3)))
    # Layer CNN3
    model.add(Convolution2D(64, 3, 3))
    model.add(ELU())

    model.add(ZeroPadding2D((1, 1)))
    # Layer CNN4
    model.add(Convolution2D(64, 3, 3))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(ZeroPadding2D((1, 1), input_shape=(64, 64, 3)))
    # Layer CNN5
    model.add(Convolution2D(128, 3, 3))
    model.add(ELU())

    model.add(ZeroPadding2D((1, 1)))
    # Layer CNN6
    model.add(Convolution2D(128, 3, 3))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten(input_shape=(3, 64, 64)))
    # Layer FC1
    model.add(Dense(256))
    model.add(ELU())
    model.add(Dropout(0.5))
    #Layer FC2
    model.add(Dense(64))
    model.add(ELU())
    model.add(Dropout(0.5))
    # Layer FC3
    model.add(Dense(16))
    model.add(ELU())
    model.add(Dropout(0.5))
    # Layer FC4
    model.add(Dense(1))

    # use adam optimizer with mean square loss function
    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt, loss="mse", metrics=['accuracy'])

    print(model.summary())

    return model


# main
if __name__ == "__main__":

    # load data
    df = get_dataframe_and_shuffle(DRIVING_LOG_FILE)
    print("loaded, shuffled driving_log into data frames")

    # splitting to training and validation data with 80-20
    num_rows = int(df.shape[0]*0.8)
    print(num_rows)
    df_train = df.loc[0:num_rows-1]
    df_val = df.loc[num_rows:]
    print("splitted data to training and validation")

    # set the generator for the training data and validation
    gen_train =  generate_batch_data_from_dataframe(df_train, batch_size=BATCH_SIZE)
    gen_val = generate_batch_data_from_dataframe(df_val, batch_size=BATCH_SIZE)
    print("Done setting the generator")

    # compile a model
    model = get_model()
    print("Done setting model")

    model.fit_generator(gen_train, validation_data=gen_val, samples_per_epoch=20000, nb_epoch=3, nb_val_samples=3000)
    print("Finished training")

    # save weights
    model.save_weights('model.h5')
    print("saved weights in model.h5")

    # save model
    with open('model.json', 'w') as model_file:
        model_file.write(model.to_json())
    print("saved model in model.json")
    