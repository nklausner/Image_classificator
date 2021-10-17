import logging
from numpy import reshape, argmax
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions


class CutleryPredictor:

    def __init__(self):
        """
        holds the pretrained model
        """
        self.classes = ['fork', 'knife', 'spoon']
        self.model = load_model('models/mobilenetV2_cutlery.h5')
        logging.info('MobileNetV2 ready to predict cutlery')

    
    def predict(self, a):
        """
        asks the model for a prediction
        a: numpy array of shape(224, 224, 3)
        """
        a = preprocess_input(a)
        a = a.reshape(1, 224, 224, 3)
        pred = self.model.predict(a)[0]
        pred_perc = round(max(pred) * 100)
        pred_name = self.classes[argmax(pred)]
        #logging.info(f'{pred_perc} % - {pred_name}')
        return pred_name, pred_perc