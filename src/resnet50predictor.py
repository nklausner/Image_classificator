import logging
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions
from numpy import expand_dims

class ResNet50Predictor:

    def __init__(self):
        """
        holds the pretrained model
        """
        self.model = ResNet50(weights='imagenet')
        logging.info('ResNet50 ready to predict out of 1000 objects')
    
    def predict(self, img_array):
        """
        asks the model for a prediction
        img_array: numpy array of shape(224, 224, 3)
        """
        img_array = expand_dims(img_array, axis=0)
        pred = self.model.predict(img_array)
        labels = decode_predictions(pred, top=1)
        pred_name = labels[0][0][1]
        pred_perc = round(labels[0][0][2] * 100)
        #logging.info(f'{pred_perc} % - {pred_name}')
        return pred_name, pred_perc
