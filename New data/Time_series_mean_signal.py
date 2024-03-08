import scipy.ndimage as img
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

from import_single_data import Data, Label
#%%____________________________________________________________________________
N_trials, Nt, Nx, Ny = Data.shape

n_trial = 13

Sample = np.copy(Data[n_trial])
#%%____________________________________________________________________________
mean_signal = np.nanmean(Sample, axis=(1,2))


fs=110 #Hz
Time=np.array([k/fs*1000 for k in range(Nt)])

# Compute the power spectrum using FFT
fft_result = np.fft.fft(mean_signal)
power_spectrum = np.abs(fft_result)**2

# Frequency axis for plotting
freqs = np.fft.fftfreq(Nt, d=1/fs)

cutoff = [10]
filt_type = 'lowpass' if len(cutoff)==1 else 'bandpass'

if filt_type=='lowpass':
    lab = ' under '+str(cutoff[0])+' Hz'
elif filt_type=='bandpass':    
    lab = ' between '+str(cutoff[0])+' and '+str(cutoff[1])+' Hz'

butt_filt = signal.butter(N=4, Wn=cutoff, btype=filt_type, fs=110, output='sos')
filtered=signal.sosfiltfilt(butt_filt, mean_signal, axis=0)


#%%____________________________________________________________________________
# Plot the signal and its power spectrum
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(8, 6))
fig.suptitle('Mean amplitude of ' + Label[0]+' activity in '+Label[1]+" monkey, trial n°"+str(n_trial))

ax1.plot(Time, mean_signal, label='raw signal')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Time series')
ax1.plot(Time, filtered, label=filt_type+lab)
ax1.legend()

ax2.plot(freqs[:len(freqs)//2-1], power_spectrum[:len(freqs)//2-1], 'k')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Power')
ax2.set_title('Power Spectrum')
ax2.vlines(10, 0, max(power_spectrum),'gray', linestyles='dashed')
ax2.hlines(0, 0, fs/2, 'gray')
ax2.set_xlim(0, fs/2)  # Limit the x-axis to the positive frequencies

plt.tight_layout()
plt.show()