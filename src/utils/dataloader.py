import json
import os
import numpy as np

def load(data_path):
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), data_path)
    with open(path, "r") as st_json:

        json_data = json.load(st_json)['data']

    print('num of data {}'.format(len(json_data)))

    X = np.empty((0,)+np.array(json_data[0]['actuation']).shape)
    y = np.empty((0,)+np.array(json_data[0]['position']).shape)

    for data in json_data:
        X = np.concatenate((X, [data['actuation']]), axis=0)
        y = np.concatenate((y, [data['position']]), axis=0)

    return X, y