import h5py
import numpy as np

filepath = r'C:\Users\mmill\Desktop\2014 VSD data\vsd_data.mat'

arrays = {}
f = h5py.File(filepath)

for k, v in f.items():
    arrays[k] = np.array(v)
    
Raw_Data = arrays['vsd_data']

Data = np.transpose(Raw_Data, (0,1,3,2))