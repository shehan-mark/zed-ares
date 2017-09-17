import pandas as pd
import numpy as np
import csv
import math

_data = pd.read_csv('training-data.csv', names = ["numOfEnemies", "healthLevel", "enemyDistanceTotal", "actionTook"])

def clean_data():
  with open("init_data.csv", 'w', newline='') as f:
    writer = csv.writer(f, dialect='excel')
    for index, row in _data.iterrows():
      arr = row.as_matrix(columns=None)
      arr.astype(np.int64)
      if arr[0] != '' and arr[1] != '' and arr[2] != '' and arr[3] != '':
        if  math.isnan(float(arr[0])) != True and math.isnan(float(arr[1])) != True and math.isnan(float(arr[2])) != True and math.isnan(float(arr[3])) != True:
          h = arr[1] * 100
          arr[0] = arr[0]
          arr[1] = h
          arr[2] = np.ceil(arr[2])
          arr[3] = arr[3]
          writer.writerow(arr)
          print(arr)

clean_data()