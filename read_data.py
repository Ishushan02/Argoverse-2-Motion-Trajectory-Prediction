import numpy as np

object_type = {
    0:'vehicle', 
    1:'pedestrian', 
    2:'motorcyclist', 
    3:'cyclist', 
    4:'bus', 
    5:'static', 
    6:'background', 
    7:'construction', 
    8:'riderless_bicycle', 
    9:'unknown'
    }

def getData(path):
    train_file = np.load(path+"/train.npz")
    train_data = train_file['data']
    test_file = np.load(path+"/test_input.npz")
    test_data = test_file['data']
    print(f"train_data's shape {train_data.shape} and test data is {train_data.shape}")
    return train_data, test_data

# print(train_data[0][0].shape)