from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model


def mobilenet():
    base_model = MobileNetV2(include_top=False, weights='imagenet')

    # Add top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    preds = Dense(units=7, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=preds)
    return model