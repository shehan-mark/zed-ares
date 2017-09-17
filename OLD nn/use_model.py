import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

# load model
fileName = '3 - trained model/ares_nn.sav'
loadedModel = pickle.load(open(fileName, 'rb'))

def startPredict(arr):
  pred = loadedModel.predict(arr)
  return pred

# print(startPredict([[1,2,2,3]]))
