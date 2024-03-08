import numpy as np
import h5py

#Blank trials (spontaneous waves) in awake monkey
data_awake_blank = np.load(r'C:\Users\mmill\Desktop\Data VSD\Awakeness\blank_ziggy_210908.npy')

#Evoked activity in awake monkey
data_awake_evoked = np.load(r'C:\Users\mmill\Desktop\Data VSD\Awakeness\gaussian_pos4_ziggy_210908.npy')

#Blank trials (spontaneous waves) in anesthetised monkey
f_blank = h5py.File(r'C:\Users\mmill\Desktop\Data VSD\Anesthesia\signal_240306_blank.mat')['signal']
data_anesth_blank = np.array(f_blank)

#Evoked activity in anesthetised monkey
f_evokD = h5py.File(r'C:\Users\mmill\Desktop\Data VSD\Anesthesia\signal_240306_gaussianD.mat')['signal']
data_anesth_evokD = np.array(f_evokD)

#Evoked activity in anesthetised monkey
f_evokU = h5py.File(r'C:\Users\mmill\Desktop\Data VSD\Anesthesia\signal_240306_gaussianU.mat')['signal']
data_anesth_evokU = np.array(f_evokU)

