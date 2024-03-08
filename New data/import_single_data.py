import numpy as np
import h5py


#filepath = r'C:\Users\mmill\Desktop\Data VSD\Awakeness\gaussian_pos4_ziggy_210908.npy'
#filepath = r'C:\Users\mmill\Desktop\Data VSD\Awakeness\blank_ziggy_210908.npy'
#filepath = r'C:\Users\mmill\Desktop\Data VSD\Anesthesia\signal_240306_gaussianD.mat'
#filepath = r'C:\Users\mmill\Desktop\Data VSD\Anesthesia\signal_240306_gaussianU.mat'
filepath = r'C:\Users\mmill\Desktop\Data VSD\Anesthesia\signal_240306_blank.mat'

if 'gaussian' in filepath:
    Label=['Evoked']
elif 'blank' in filepath:
    Label=['Spontaneous']

if 'Awakeness' in filepath:
    Data = np.load(filepath)
    Label.append('awake')

if 'Anesthesia' in filepath:
    f = h5py.File(filepath)['signal']
    Data = np.array(f)
    Label.append('anesthetised')
    