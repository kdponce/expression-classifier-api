from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from sklearn.utils import class_weight
from model import customcnn
import numpy as np
import tensorflow as tf

# Constants
learning_rate = 0.001
epochs = 20
batch_size = 32


def build_model():
    # Only allocates a subset of the available GPU Memory and take more as needed.
    # Prevents "Failed to get convolution algorithm" error on Elementary OS Juno.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = customcnn()
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory('lib/data/Training',
                                                        target_size=(48, 48),
                                                        color_mode='rgb',
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=True)

    validation_generator = valid_datagen.flow_from_directory('lib/data/Validation',
                                                             target_size=(48, 48),
                                                             color_mode='rgb',
                                                             batch_size=batch_size,
                                                             class_mode='categorical',
                                                             shuffle=False)

    step_size_train = train_generator.n // train_generator.batch_size
    step_size_valid = validation_generator.n // validation_generator.batch_size

    class_weights = class_weight.compute_class_weight(
        'balanced',
        np.unique(train_generator.classes),
        train_generator.classes
    )

    # Save model after every epoch with improvements in val_loss
    mc_fit = ModelCheckpoint('lib/checkpoints/model_fit_best.hdf5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto',
                             period=1)

    early_stop = EarlyStopping(monitor='val_loss',
                               patience=3,
                               verbose=1,
                               mode='auto',
                               restore_best_weights=True)

    # Train model
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=step_size_train,
                                  validation_data=validation_generator,
                                  validation_steps=step_size_valid,
                                  callbacks=[mc_fit, early_stop],
                                  epochs=epochs,
                                  class_weight=class_weights)

    model.save('lib/models/model.h5')
    return model


def load_model():
    return load_model('lib/models/model.h5')
