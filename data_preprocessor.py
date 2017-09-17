import pandas as pd
import numpy as np
import csv

_data = pd.read_csv('1 - initial data/init_data.csv', names = ["numOfEnemies", "healthLevel", "enemyDistanceTotal", "actionTook"])
# print(_data)

# def constants
max_enemy_count = 3
min_health_level = 70
min_min_health_level = 50
min_tot_distance = 2500

action_engage_identifier = 1

def feature_identifier(eCount, hlvl, eDis):
  retVal = -1
  if eCount >= max_enemy_count:
    # if hlvl < min_health_level and eDis < min_tot_distance:
    #   retVal = 1
    # elif eDis < min_tot_distance:
    #   retVal = 1
    # else:
    retVal = 1
  elif hlvl <= min_min_health_level and eDis < min_tot_distance:
    retVal = 1
  else:
    retVal = 0
  
  return retVal

def labelData():
  with open("2 - preprocessed data/processed_data.csv", 'w', newline='') as f:
    writer = csv.writer(f, dialect='excel')
    for index, row in _data.iterrows():
      arr = row.as_matrix(columns=None)
      if arr[3] == action_engage_identifier:
        res = feature_identifier(arr[0], arr[1], arr[2])
        if res > -1:
          arr = np.insert(arr, 0, res)
          writer.writerow(arr)
          print(arr)
          
labelData()