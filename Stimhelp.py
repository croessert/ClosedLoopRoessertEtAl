# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 19:12:05 2011

@author: chris

"""

from __future__ import division

#import sys
#sys.path.insert(0, "/home/chris/lib/python")  # for new matplotlib!!!

from pylab import *
from numpy import round, random, any
from units import *
import time
from NeuroTools import stgen
import h5py
import os
import shutil

         
def make_colormap(seq):
    import matplotlib.colors as mcolors
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)
    
def create_colnoise(t, sexp, cutf = None, seed = None, onf = None):
    '''x = create_colnoise(t, sexp, cutf)
    Make coloured noise signal
    t    = vector of times
    sexp = spectral exponent - Power ~ 1 / f^sexp
    cutf = frequency cutoff  - Power flat (white) for f <~ cutf
    
    cutf == None OR sexp == 0: white noise
    
    output:  mean = 0, std of signal = 1/2 => max 95% is 1
    '''
    
    from scipy import signal
    
    nt = len(t)
    dt = (t[-1] - t[0]) / (nt - 1)
    random.seed(seed)
    
    #if sexp == 0:  # no smooth cutoff
    #    N = 10001   # number of filter taps
    #    xp = int((N-1)/2)
    #    x = random.standard_normal(size = len(t)+xp)   
    #else:
    #    x = random.standard_normal(size = shape(t)) 
        
    x = random.standard_normal(size = shape(t))/2
    #print std(x)
    if cutf == None:
        pass
    else:
        
        if sexp == 0:  # no cutoff
            pass
        

        elif sexp == -1:  # sharp cutoff
            x, freq, freq_wp, freq_used = create_multisines(t, freq_used=None, cutf=cutf, onf=onf)
            #print std(x)
            x = x / std(x) / 2         # std of signal = 1, mean = 0
                        
        else:  # smooth cutoff   
            
            x = fft(x)
            f = fftfreq(nt, dt)
            x[nonzero(f == 0)] = 0               # remove zero frequency contribution
        
            i = nonzero(f != 0)              # find indices i of non-zero frequencies
            x[i]=x[i] / (cutf ** 2 + f[i] ** 2) ** (sexp / 4)  # using i allows cutf = 0
        
            x = real(ifft(x))      # ignore imaginary part (numerical error)
            x = x / std(x) / 2         # std of signal = 1/2, mean = 0
            
    return x


def create_multisines(t, freq_used=array([1]), cutf = None, onf = None):
    """
    This function will produce a colored noise signal using the time points 
    as defined in array t. The signal will be constructed using sinosoids with 
    frequencies as defined in the array freq_used and randomized phases.

    The output consists of signal x        
    """

    tstop = t[-1]
    df = 1 / tstop # frequency stepsize
    dt = t[2] - t[1]
    data_points = len(t) # number of time or frequency steps  

    vector_freq = zeros(data_points) 
    vector_phase = zeros(data_points)

    if cutf != None:
        f = arange(0,data_points)*df
        if onf != None:
            freq_used = f[nonzero((f <= cutf) & (f >= onf))]
            #print freq_used
        else:            
            freq_used = f[nonzero((f <= cutf) & (f > 0))]
        
    index_f_used = round(freq_used / df).astype('int') # get indices of used frequencies in frequency vector
    
    index_fneg_used = (data_points - index_f_used).astype('int') # indices of negative frequencies 

    index_fall_used = concatenate((index_f_used, index_fneg_used)) # indices of pos+neg frequencies

    vector_freq[index_fall_used] = data_points / 2 # each frequency used ???
    
    phase = 2*pi*(random.rand(len(freq_used),1)-0.5) # pick phases randomly shifted by +-pi (sould there by another 2* to shift +-2pi???)

    vector_phase[index_f_used] = phase # assign positive phases to full vector
    vector_phase[index_fneg_used] = -phase # assign negative phases to full vector

    freqtemp = vector_freq * exp(1j * vector_phase) # generate frequency domain response
    x = real(ifft(freqtemp)) # convert into time domain

    #print "- Number of msine frequencies: " + str(2 * std(x) ** 2)
    
    noise_data = x/max(abs(x)) # scale so that signal amplitude is 1
    
    freq = fftfreq(data_points, dt)[ 0:round(data_points / 2) ] # positive frequency vector
   
    noise_power = abs(fft(noise_data))[ 0:round(data_points / 2) ] # compute noise power
    freq_wp = find(noise_power > 2 * std(noise_power)) # threshold to discriminate the indexes of peak frequencies
    freq_used = freq[freq_wp] # vector of used frequencies [Hz]
    
    return noise_data, freq, freq_wp, freq_used 
    

def create_singlesine(fu = 5, amp = 0.5, ihold = 1, dt = 0.025*ms, periods = 10, minlength = 1*s, t_prestim = 2*s, l_zeros = 2):
    """
    This function will produce a single sine signal of frequency fu with holding current ihold
    Signal has at least the length periods*T (s) or minlength (s).
    Use stimulate with pre stimulus of length t_prestim (s)        
    """
    
    fs = 1 / dt  # sampling rate 
    
    tnext = 0
    # delay for no noise input
    start_zeros =  zeros(l_zeros * fs)   
    t_zeros = tnext + arange(0, l_zeros, dt)
    
    tnext = t_zeros[-1] + dt
    l_pre_signal = ceil(t_prestim / (1. / fu)) * 1. / fu # length of pre stimulus should to be at least t_prestim seconds but with length of full periods
    t_pre_signal = arange(0, l_pre_signal, dt) # create pre time vector
    
    pre_signal = amp * sin(2 * pi * t_pre_signal * fu) # create pre signal vector
    t_pre_signal = t_pre_signal + tnext
    
    tnext = t_pre_signal[-1] + dt
    l_t = max(minlength, periods * 1 / fu) # length of input_signal: stimulate for at least periods*T or minlength
    t_input_signal = arange(0, l_t, dt) # create stimulus time vector
    
    #window = sin(2 * pi * t_input_signal * 1/l_t/2)  # not really good if nonlinear membrane function!!!!
    input_signal = amp * sin(2 * pi * t_input_signal * fu)
    t_input_signal = t_input_signal + tnext
    
    i_start = len(start_zeros) + len(pre_signal)
    i_stop = len(start_zeros) + len(pre_signal) + len(input_signal) 
    
    tnext = t_input_signal[-1] + dt
    l_post_signal = 1 # length of post stimulus should only be 1 s, equivalent to 1 Hz lower bound for spike cutoff 
    t_post_signal = arange(0, l_post_signal, dt) # create pre time vector
    post_signal = amp * sin(2 * pi * t_post_signal * fu) # create pre signal vector
    t_post_signal = t_post_signal + tnext

    all_data = concatenate((start_zeros, pre_signal, input_signal, post_signal)) # combine all
    t = concatenate((t_zeros, t_pre_signal, t_input_signal, t_post_signal)) # combine all
    t1 = arange(0, size(all_data) * dt, dt) # time vector of stimulus [s]  
    
    i_startstop = array([i_start, i_stop])
    t_startstop = array([t[i_start], t[i_stop]])
    
    iholdvec = concatenate((zeros(1 * fs), ones(len(all_data) - 1 * fs) * ihold)) # create holding current vector
    #iholdvec = concatenate((zeros(1 * fs), ones(len(all_data) - 2 * fs) * ihold, zeros(1 * fs))) # create holding current vector
    
    stimulus = all_data + iholdvec # create full stimulus vector
    
    return t, stimulus, i_startstop, t_startstop

        
    

def syn_kernel(t, tau1 = 5, tau2 = 10):    

    if tau1 == 0:
        
        G = exp(-t/tau2)    

    else:
        
        if (tau1/tau2 > .9999):
            tau1 = .9999*tau2
    
        tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
        factor = -exp(-tp/tau1) + exp(-tp/tau2)
        factor = 1/factor                        
        G = factor * (exp(-t/tau2) - exp(-t/tau1))
    
    return G
    

def construct_Stimulus(noise_data, fs, amp = 1, ihold = 0, tail_points = 2, delay_baseline = 4):
    """
    Construct Stimulus from cnoise/msine input and other parameters.
    """

    #inin = 8 # stimulate before with 10 s of signal
    inin = np.array((len(noise_data)/fs)*0.1).clip(max=8)

    stim_data = concatenate((noise_data[-inin*fs:], noise_data))  # increase length of stimulus # no normalization here: / max(abs(noise_data))
    
    stimulus = concatenate((concatenate((zeros(round(delay_baseline*fs)), amp * stim_data)), zeros(round(tail_points*fs))))  # construct stimulus
            
    iholdvec = concatenate((zeros(round(fs)), ones(round(len(stimulus) - 1 * fs)) * ihold))
    stimulus = stimulus + iholdvec
    
    dt = 1 / fs
    t = arange(0, len(stimulus) * dt,dt)  # time vector of stimulus [s]
            
    t_startstop = np.array([inin+delay_baseline, inin+delay_baseline+len(noise_data)/fs])
    
    return stimulus, t, t_startstop
    

def construct_Pulsestim(dt = 0.025e-3, pulses = 1, latency = 10e-3, stim_start = 0.02, stim_end = 0.02, len_pulse = 0.5e-3, amp_init = 1, amp_next = None):
    """
    Construct a pulse stimulus in the form of |---stim_start---|-len_pulse-|--(latency-len_pulse)--|...|-len_pulse-|--(latency-len_pulse)--|--stim_end--|
    For stim_end shorter than pulse: stim_end = stim_end + len_pulse
    """
    #print dt
    
    fs = 1 / dt
    
    if len(np.shape(amp_next)) > 0:
        pulses = len(amp_next)
        amp_vec = amp_next
    else:
        if amp_next == None:
            amp_vec = np.ones(pulses)*amp_init
        else:
            amp_vec = np.ones(pulses)*amp_next
            amp_vec[0] = amp_init

    if len(np.shape(latency)) > 0: 
        pass
    else:
        latency = np.ones(pulses)*latency
        
    if len(amp_vec) != len(latency):    
        raise ValueError('amp_vec and latency vectors do not have the same size!!!')
        
     
    if stim_end < len_pulse:
        print "From construct_Pulsestim: stim_end shorter than pulse, setting stim_end = stim_end + len_pulse"
        stim_end = stim_end + len_pulse
           
    ivec = zeros(round((stim_start + sum(latency) + stim_end)*fs))  # construct zero vector to begin with 
    t = arange(0, len(ivec))*dt  # construct time vector
    
    ivec[round(stim_start*fs):round((stim_start+len_pulse)*fs)] = amp_vec[0]
                    
    for i in range(1, pulses):
        
        ivec[round((stim_start+sum(latency[0:i]))*fs):round((stim_start+sum(latency[0:i])+len_pulse)*fs)] = amp_vec[i]
                               
    return t, ivec

    
def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    
    From Django's "django/template/defaultfilters.py".
    """
    
    for c in r'[]/\;,><&*:%=+@!#^\'()$| ?^':
        value = value.replace(c,'')
        
    return value

    
def nans(shape, dtype=float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a

    
def nanmean(a, dim=0):
    return np.mean(np.ma.masked_array(a,np.isnan(a)),dim).filled(np.nan)  


def rw_hdf5(filepath = "data_dict.hdf5", data_dict = None, export = False):
    
    if data_dict == None: # load
        
        data_dict = {}
        
        print "rw_hdf5 load:", filepath
        f = h5py.File(filepath, 'r')

        for i in f.items():
            data_dict[i[0]]  = np.array(i[1])    
            
        f.close() 
            
        if export:
            if not os.path.exists(export):
                os.makedirs(export)
            shutil.copy(filepath, export)

    else:
        
        f = h5py.File(filepath, 'w')
        
        for name  in data_dict:

            data0 = data_dict[name]
            
            if (type(data0) == np.ndarray):
                pass
            else:
                data0 = np.array(data0)
            
            #print data0
            if len(np.shape(data0)) < 1:
                f.create_dataset(name, data=data0)
            else:
                f.create_dataset(name, data=data0, compression='lzf')
                

        f.close() 
    
    return data_dict
    

# test code
if __name__ == '__main__': 
    
    pass