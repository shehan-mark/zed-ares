import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

# load data
b_data = pd.read_csv('2 - preprocessed data/processed_data.csv', names = ["class", "numOfEnemies", "healthLevel", "enemyDistanceTotal", "actionTook"])
b_data.head()
b_data.describe().transpose()
b_data.shape
# print(b_data)

# split data and labels
X = b_data.drop('class',axis=1)
y = b_data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y)

# def constants
fileName = "3 - trained model/ares_nn.sav"
dimentions = 4

# transformations 
scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# define nn
mlp = MLPClassifier(hidden_layer_sizes=(dimentions,dimentions,dimentions),max_iter=500,random_state=1,verbose=True)
# train
mlp.fit(X_train,y_train)
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(dimentions,dimentions,dimentions), learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)

# export model
pickle.dump(mlp, open(fileName, 'wb'))