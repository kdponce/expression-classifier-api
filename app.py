from flask import Flask
from flask_restful import reqparse, Api, Resource
from build_model import build_model
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import werkzeug

app = Flask(__name__)
api = Api(app)

# Suppress Tensorflow verbosity
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Create model
model = build_model()


# Performs the main classification task
class PredictExpression(Resource):
    def get(self):
        return 'Facial expression classifier. Test with \'curl -F \'file=@photo.jpg/png\' localhost:5000'

    def post(self):
        # Argument parsing
        parser = reqparse.RequestParser()
        parser.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args = parser.parse_args()

        # Terminates with warning if no image has been detected.
        if args['file'] is None:
            return 'Invalid syntax. Try again with \'curl -F \'file=@photo.jpg/png\' localhost:5000'

        # Convert image input to 48*48 grayscale
        img = np.asarray(Image.open(args['file']))[:, :, ::-1]
        img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.reshape((1, img.shape[0], img.shape[1], 1))

        # Open Class labels dictionary
        classes = eval(open('lib/data/classes.txt', 'r').read())

        # Only allocates a subset of the available GPU Memory and take more as needed.
        # Prevents "Failed to get convolution algorithm" error on Elementary OS Juno.
        # Use only in conjunction with keras-gpu
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # sess = tf.Session(config=config)

        # Predict
        preds = model.predict(img)

        # Return results in JSON format
        results = dict()
        results['class'] = classes[np.argmax(preds)]
        results['certainty'] = str(preds[0][np.argmax(preds)])
        results['probs'] = dict(zip(classes.values(), preds.tolist()[0]))
        return results


api.add_resource(PredictExpression, '/')

if __name__ == '__main__':
    app.run(debug=False, threaded=False)
