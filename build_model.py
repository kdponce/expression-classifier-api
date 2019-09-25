from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
from model import mobilenet
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

    model = mobilenet()

    for layer in model.layers[:117]:
        layer.trainable = False

    model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory('lib/data/Training',
                                                        target_size=(224, 224),
                                                        color_mode='rgb',
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle=True)

    validation_generator = valid_datagen.flow_from_directory('lib/data/Validation',
                                                             target_size=(224, 224),
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
    mc_fit = ModelCheckpoint('lib/checkpoints/mobilenet_fit_best.hdf5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto',
                             period=1)

    # Train model
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=step_size_train,
                                  validation_data=validation_generator,
                                  validation_steps=step_size_valid,
                                  callbacks=[mc_fit],
                                  epochs=epochs,
                                  class_weight=class_weights)

    model.save('lib/models/mobilenet.h5')

    print('Model Accuracy')
    for i in history.history['acc']:
        print(i)

    print('Validation Accuracy')
    for i in history.history['val_acc']:
        print(i)

    print('Model Loss')
    for i in history.history['loss']:
        print(i)

    print('Validation Loss')
    for i in history.history['val_loss']:
        print(i)


build_model()
