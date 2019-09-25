from keras.layers import Conv2D, AveragePooling2D, Dropout, Flatten, Dense
from keras.models import Sequential


# Convolutional neural network for image classification
def customcnn():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=2, activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=2, activation='relu'))
    model.add(Conv2D(128, kernel_size=2, activation='relu'))
    model.add(AveragePooling2D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    return model
