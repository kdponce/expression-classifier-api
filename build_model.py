from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils import class_weight
from model import customcnn
import numpy as np
import tensorflow as tf
import wget
import os


# Creates the model and checks for classes file during startup
def build_model():
    model = customcnn()

    if not os.path.exists('lib/models/expression_classifier_weights.hdf5'):
        # Downloads weights file from releases
        url = 'https://github.com/kdponce/expression-classifier-api/releases/download/v0.1.0/expression_classifier_weights.hdf5'
        print('Downloading weights from {}'.format(url))
        wget.download(url, 'lib/models/expression_classifier_weights.hdf5')

    if not os.path.exists('lib/data/classes.txt'):
        # Downloads classes file from releases
        url = 'https://github.com/kdponce/expression-classifier-api/releases/download/v0.1.0/classes.txt'
        print('Downloading classes file from {}'.format(url))
        wget.download(url, 'lib/data/classes.txt')

    model.load_weights('lib/models/expression_classifier_weights.hdf5')
    return model


# Generates a new model. Used for development purposes only.
# Original dataset used in a randomly sampled mixture of the CK+ and FER2013 datasets.
def train_model():

    # Only allocates a subset of the available GPU Memory and take more as needed.
    # Prevents "Failed to get convolution algorithm" error on Elementary OS Juno.
    # Use only in conjunction with keras-gpu
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    # Constants
    epochs = 20
    batch_size = 32

    model = customcnn()
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory('lib/data/Training',
                                                        target_size=(48, 48),
                                                        color_mode='grayscale',
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=True)

    validation_generator = valid_datagen.flow_from_directory('lib/data/Validation',
                                                             target_size=(48, 48),
                                                             color_mode='grayscale',
                                                             batch_size=batch_size,
                                                             class_mode='categorical',
                                                             shuffle=False)

    step_size_train = train_generator.n // train_generator.batch_size
    step_size_valid = validation_generator.n // validation_generator.batch_size

    # Since the dataset is significantly imbalanced, weights
    # will be used to provide emphasis on the minority classes
    class_weights = class_weight.compute_class_weight(
        'balanced',
        np.unique(train_generator.classes),
        train_generator.classes
    )

    # Save weights after every epoch with improvements in val_loss
    mc_fit = ModelCheckpoint('lib/checkpoints/model_fit_best.hdf5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='auto',
                             period=1)

    # Terminates training when validation loss is no longer decreasing
    early_stop = EarlyStopping(monitor='val_loss',
                               patience=3,
                               verbose=1,
                               mode='auto',
                               restore_best_weights=True)

    # Train model
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=step_size_train,
                        validation_data=validation_generator,
                        validation_steps=step_size_valid,
                        callbacks=[mc_fit, early_stop],
                        epochs=epochs,
                        class_weight=class_weights)

    # Saves the full model
    model.save('lib/models/model.h5')
    return model
