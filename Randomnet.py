# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 17:01:25 2013

@author: chris

mpiexec -f ~/machinefile0 -enable-x -n 15 python Randomnet.py --noplot 2>&1 | tee logs/test.txt

"""

from __future__ import with_statement
from __future__ import division


import os
import warnings
warnings.filterwarnings("ignore")

import sys
argv = sys.argv

if "-nompi" not in argv:
    try:
        from mpi4py import MPI
    except ImportError:
        pass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import random as rnd

try:
    import neuronpy.util.spiketrain
    from sklearn import linear_model
except ImportError:
    pass

#set_printoptions(threshold='nan')

from Stimhelp import *
from units import *


try:
    import cPickle as pickle
except:
    import pickle

import gzip
import h5py
import math
import copy

import scipy
from scipy import io

from scipy.optimize import fmin, leastsq, anneal, brute, fmin_bfgs, fmin_l_bfgs_b, fmin_slsqp, fmin_tnc, basinhopping


def save_results(pickle_prefix, p = None, err = None, vaf = None, mstdx = None, ly = False, fstd = None, params = False, use_pc = False, use_h5 = True, run = 0, export = False, data_dir = "./data/"):

        filepath = data_dir + pickle_prefix + "_fit_results"
        if run > 0: filepath = filepath + "_run" + str(run)

        if use_h5:
            filepath = filepath + '.hdf5'
        else:
            filepath = filepath + '.p'

        if p is not None: # save

            open(data_dir + pickle_prefix + "_fit_results.txt", 'w').write(str(p) + " " + str(err) + " " + str(vaf) + " " + str(ly) + "\n")

            fit = {}
            #fit['params'] = params.copy()
            fit['p'] = p
            fit['err'] = err
            fit['vaf'] = vaf
            fit['ly'] = ly
            fit['mstdx'] = mstdx
            fit['fstd'] = fstd

            if use_h5:
                rw_hdf5(filepath, fit)
            else:
                pickle.dump( fit, gzip.GzipFile( filepath, "wb" ) )

        else:

            #print filepath
            if use_h5:
                fit = rw_hdf5(filepath, export = export)
            else:
                fit = pickle.load( gzip.GzipFile( filepath, "rb" ) )

            p = fit['p']
            err = fit['err']
            vaf = fit['vaf']
            ly = fit['ly']
            mstdx = fit['mstdx']
            if 'fstd' in fit: fstd = fit['fstd']

        return p, err, vaf, ly, mstdx, fstd


def func(params, xdata, ydata):
    return (ydata - np.dot(xdata,params))


def barrier(use_mpi = True, use_pc = False):
    if use_mpi:
        if use_pc:
            pc.barrier()
        else:
            MPI.COMM_WORLD.Barrier()


def randomnet(params):

    np.random.seed(444)

    params2 = copy.deepcopy(params) #params.copy() # copy.deepcopy(params) # params.copy()
    fitter = "ridge"

    for key, value in params2.items():
        exec(key + "= params2.get(\"" + key + "\")")

    fs = 1/dt

    if use_pc:
        imgf = ".svg"
    else:
        imgf = ".pdf"

    if ("_poster_" in do):

        color1 = '#00A0E3' # Cyan
        color2 = '#E5097F' # Magenta
        color3 = '#808080' # Gray
        color4 = '#78317B' # Lila
        color5 = '#EC671F' # Orange
        color6 = '#009A47' # Dark Green
        color7 = '#FFED00' # Yellow
        color8 = '#393476' # Uni Blue
        color9 = '#E42A24' # Red

        linewidth = 1.5

        output_dim_plot = 5

        fig_size =  [4.86,2.5] # 1.5-Column 6.83
        params = {'backend': 'ps',
          'axes.labelsize': 8,
          'axes.linewidth' : 0.5,
          'title.fontsize': 8,
          'text.fontsize': 10,
          'font.size':10,
          'axes.titlesize':8,
          'legend.fontsize': 6,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'legend.borderpad': 0.2,
          'legend.linewidth': 0.1,
          'legend.loc': 'best',
          'legend.ncol': 4,
          'text.usetex': False,
          'figure.figsize': fig_size}
        rcParams.update(params)


        if ("_pca_" in do):

            fig_size =  [6.83, 6.83] # 2-Column
            params['figure.figsize'] = fig_size
            rcParams.update(params)

            plt.figure("special_pca")

            gs0 = matplotlib.gridspec.GridSpec(2, 1, width_ratios=[1], height_ratios=[0.4,0.6])
            gs0.update(wspace=0.3, hspace=0.2, bottom=0.13, top=0.95, left=0.11, right=0.97)

            gs00 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0])
            ax1 = plt.subplot(gs00[0])
            gs01 = matplotlib.gridspec.GridSpecFromSubplotSpec(output_dim_plot, 1, subplot_spec=gs0[1])

        elif ("_cnoise_" in do):

            fig_size =  [4.86, 4]
            params['figure.figsize'] = fig_size
            rcParams.update(params)

            plt.figure("special_basis")

            gs = matplotlib.gridspec.GridSpec(4, 1,
               width_ratios=[1],
               height_ratios=[1.5,1,1,1]
               )
            gs.update(bottom=0.11, top=0.94, left=0.11, right=0.92, wspace=0.4, hspace=0.175)
            ax1 = plt.subplot(gs[0,0])
            ax2 = plt.subplot(gs[1,0])
            ax3 = plt.subplot(gs[2,0])
            ax4 = plt.subplot(gs[3,0])


            fig_size =  [4.86, 3]
            params['figure.figsize'] = fig_size
            rcParams.update(params)

            plt.figure("special_cnoise")

            gs = matplotlib.gridspec.GridSpec(3, 2,
               width_ratios=[1,1],
               height_ratios=[1,1,1]
               )
            gs.update(bottom=0.12, top=0.94, left=0.09, right=0.97, wspace=0.4, hspace=0.2)
            ax11 = plt.subplot(gs[0,0])
            ax21 = plt.subplot(gs[1,0])
            ax31 = plt.subplot(gs[2,0])

            ax12 = plt.subplot(gs[0,1])
            ax22 = plt.subplot(gs[1,1])
            ax32 = plt.subplot(gs[2,1])

        elif ("_step_" in do):

            fig_size =  [6.83, 6.83] # 2-Column
            params['figure.figsize'] = fig_size
            rcParams.update(params)

            plt.figure("special_step")

            gs = matplotlib.gridspec.GridSpec(1, 1,
               width_ratios=[1],
               height_ratios=[1]
               )
            gs.update(bottom=0.12, top=0.94, left=0.09, right=0.97, wspace=0.4, hspace=0.2)
            ax11 = plt.subplot(gs[0,0])


    simprop = pickle_prefix
    filename = slugify(simprop)
    pickle_prefix = filename

    ifun = False
    if "ifun" in celltype[0]:
        ifun = True

    if ifun:

        myid = 0

    elif do_run:

        pop = None

        pop = Population(cellimport = cellimport, celltype = celltype, cell_exe = cell_exe, N = N, temperature = temperature,
                         ihold = ihold, ihold_sigma = ihold_sigma, give_freq = give_freq, do_run = do_run, pickle_prefix = pickle_prefix,
                         istart = istart, istop = istop, di = di, dt = dt, use_mpi = use_mpi, use_pc = use_pc)

        #pop.force_run = True

        pop.method_interpol = np.array(["bin"])
        pop.bin_width = bin_width
        pop.factor_celltype = factor_celltype
        myid = pop.id
        pop.seed = seed0

    cutf = 20
    sexp = -1

    signals = []
    times2 = []
    x = []

    ##### SETUP STIM #####

    if nulltest is False:

        # TRAIN

        rec_x = []
        for ir in range(len(tau2_basis)):
            rec_x.append(np.zeros(N[0]))
        rec_x = np.concatenate((rec_x))

        if ("_lyap" in do):

            if teacher_forcing:
                filename_ly = slugify(pickle_prefix_ly)
                filepath_recx = data_dir + str(filename_ly) + "_seed" + str(seed0) + '_recx.hdf5'
                results_recx = rw_hdf5(filepath_recx)
                rec_x = results_recx.get('rec_x')
                print filepath_recx, max(rec_x), min(rec_x)

            t_noise = arange(0, tstep[1]-tstep[0], dt)
            seed = seed0
            noise_data = create_colnoise(t_noise, sexp, cutf, seed)
            ivec, t, t_startstop0 = construct_Stimulus(noise_data, fs, amp = 0, ihold = 0, tail_points = 0*s, delay_baseline = tstep[0])

            tstop = t[-1]
            stimulus0 = ivec

            amod = [1]*len(N)

            if "_lyap1_" in do:
                if "_ifun" in do:
                    stimulus0[2*s/dt] = stimulus0[2*s/dt]+1e-14
                else:
                    stimulus0[2*s/dt] = stimulus0[2*s/dt]+1e1


        elif ("_basis_" in do) or ("_pca_" in do):

            t, ivec = construct_Pulsestim(dt = dt, latency = np.append(diff(tstep), 0), stim_start = tstep[0], stim_end = 1*s, len_pulse = tdur, amp_next = istep[0])
            t_startstop0 = [tstep[0]-t_analysis_delay, tstep[-1]+t_analysis_stop]

            # TEST
            if len(ttest) > 0:
                t, stimulus = construct_Pulsestim(dt = dt, latency = np.append(diff(ttest), 0), stim_start = ttest[0], stim_end = 3*s, len_pulse = tdur, amp_next = itest[0])
                t_startstop = [ttest[0]-t_analysis_delay, ttest[-1]+t_analysis_stop]

                tstop = t[-1]
                stimulus0 = concatenate(([ivec, np.zeros(len(stimulus)-len(ivec))])) + stimulus
            else:
                tstop = t[-1]
                stimulus0 = ivec


        elif ("_noibas_" in do):

            t_noise = arange(0, tstep[1]-tstep[0], dt)
            seed = 1
            noise_data = create_colnoise(t_noise, sexp, cutf, seed)
            ivec, t, t_startstop0 = construct_Stimulus(noise_data, fs, istep[0], ihold = 0, tail_points = 1*s, delay_baseline = tstep[0])

            t, stimulus = construct_Pulsestim(dt = dt, latency = np.append(diff(ttest), 0), stim_start = ttest[0], stim_end = 3*s, len_pulse = tdur, amp_next = itest[0])
            t_startstop = [ttest[0]-t_analysis_delay, ttest[-1]+t_analysis_stop]

            tstop = t[-1]
            stimulus0 = concatenate(([ivec, np.zeros(len(stimulus)-len(ivec))])) + stimulus


        elif ("_basnoi_" in do):

            t, ivec = construct_Pulsestim(dt = dt, latency = np.append(diff(tstep), 0), stim_start = tstep[0], stim_end = 3*s, len_pulse = tdur, amp_next = istep[0])
            t_startstop0 = [tstep[0]-t_analysis_delay, tstep[-1]+t_analysis_stop]

            t_noise = arange(0, ttest[1]-ttest[0], dt)
            seed = 2
            noise_data = create_colnoise(t_noise, sexp, cutf, seed)
            stimulus, t, t_startstop = construct_Stimulus(noise_data, fs, itest[0], ihold = 0, tail_points = 1*s, delay_baseline = t[-1])

            tstop = t[-1]
            stimulus0 = concatenate(([ivec, np.zeros(len(stimulus)-len(ivec))])) + stimulus


        elif "_cnoise_" in do:

            t_noise = arange(0, tstep[1]-tstep[0], dt)
            seed = seed0
            noise_data = create_colnoise(t_noise, sexp, cutf, seed)
            noise_data[int(len(noise_data)*0.5):] = 0
            #print "seed:", seed
            ivec, t, t_startstop0 = construct_Stimulus(noise_data, fs, istep[0], ihold = 0, tail_points = 10*s, delay_baseline = tstep[0])

            ts = len(ivec)-4*s/dt
            te = ts + tdur/dt
            ivec[ts:te] = 1

            t_noise = arange(0, ttest[1]-ttest[0], dt)
            seed = seed0+1000
            #sexp = 5
            noise_data = create_colnoise(t_noise, sexp, cutf, seed)
            noise_data[int(len(noise_data)*0.5):] = 0
            stimulus, t, t_startstop = construct_Stimulus(noise_data, fs, itest[0], ihold = 0, tail_points = 0.1*s, delay_baseline = t[-1])

            tstop = t[-1]
            stimulus0 = concatenate(([ivec, np.zeros(len(stimulus)-len(ivec))])) + stimulus

            t = np.arange(0, len(stimulus0) * dt,dt)[0:len(stimulus0)]

            #print "t:"+str(len(t))+str(len(stimulus0))


        elif "_updown_" in do:

            t = arange(0, 35, dt)

            stimulus0 = zeros(len(t))
            stimulus1 = zeros(len(t))

            stimulus0[5*s/dt:10*s/dt]=-0.95 # 0.05
            stimulus0[10*s/dt:15*s/dt]=-0.75 # 0.2
            stimulus0[20*s/dt:25*s/dt]=1.5 # 2
            stimulus0[25*s/dt:30*s/dt]=2.75 # 3

            stimulus1[5*s/dt:10*s/dt]=2.75 # 3
            stimulus1[10*s/dt:15*s/dt]=1.5 # 2
            stimulus1[20*s/dt:25*s/dt]=-0.75 # 0.2
            stimulus1[25*s/dt:30*s/dt]=-0.95 # 0.05

            t_startstop0 = [0, 0]
            t_startstop = [4,35]

            amod = [1]*len(N)


        elif "_step_" in do:

            t = arange(0, tstep[0], dt)
            stimulus0 = zeros(len(t))
            stimulus0[tstep[1]/dt:] = istep[0]
            t_startstop0 = [2, tstep[1]]
            t_startstop = [0,0]


    if ((("_basis_" in do) or ("_cnoise_" in do) or ("_noibas_" in do) or ("_basnoi_" in do)) and ("_lyap" not in do)) and (do_run_constr is not 2) and ("_pca" not in do):

        filepath = data_dir + str(filename) + "_basis"

        if use_h5:
            filepath = filepath + '.hdf5'
        else:
            filepath = filepath + '.p'

        if do_run_constr or (os.path.isfile(filepath) is False):

            times2 = arange(0, tstop+1*ms, 1*ms)
            dt2 = 1*ms

            print "dt2:", dt2, "len(times2):", len(times2)

            t_in = times2[int(t_startstop0[0]/dt2):int(t_startstop0[1]/dt2)]
            t_test = times2[int(t_startstop[0]/dt2):int(t_startstop[1]/dt2)]
            t_basis = times2[int((t_startstop0[1]+4)/dt2):int((t_startstop0[1]+10)/dt2)]

            filtered_est_v = zeros((len(t_in), len(tau2_basis)))
            filtered_test_v = zeros((len(t_test), len(tau2_basis)))
            filtered_basis_v = zeros((len(t_basis), len(tau2_basis)))
            filtered_v = zeros((len(times2), len(tau2_basis)))

            stimulus1 = interp(times2,t,stimulus0)
            est = stimulus1[int(t_startstop0[0]/dt2):int(t_startstop0[1]/dt2)]
            test = stimulus1[int(t_startstop[0]/dt2):int(t_startstop[1]/dt2)]
            basis = stimulus1[int((t_startstop0[1]+4)/dt2):int((t_startstop0[1]+10)/dt2)]

            for i in range(len(tau2_basis)):
                t_kernel = arange(0, abs(tau2_basis[i])*10, dt2)
                kernel = np.sign(tau2_basis[i])*syn_kernel(t_kernel, tau1_basis[i], abs(tau2_basis[i]))
                kernel = np.concatenate((zeros(len(kernel)-1),kernel))

                print "- Basis convolution"
                filtered = np.convolve(stimulus1, kernel, mode='same')
                filtered_est = filtered[int(t_startstop0[0]/dt2):int(t_startstop0[1]/dt2)]
                filtered_test = filtered[int(t_startstop[0]/dt2):int(t_startstop[1]/dt2)]
                filtered_basis = filtered[int((t_startstop0[1]+4)/dt2):int((t_startstop0[1]+10)/dt2)]

                filtered_est_v[:,i] = filtered_est
                filtered_test_v[:,i] = filtered_test
                filtered_basis_v[:,i] = filtered_basis
                filtered_v[:,i] = filtered

            #run_recurrent_filter0 = run_recurrent_filter

    else:
        run_recurrent_filter = 1
        teacher_forcing = False


    teacher = np.zeros(len(stimulus0)*len(tau2_basis))

    for ire in range(run_recurrent_filter):

        if (ire == 0) and (teacher_forcing):

            teacher = []
            for ir in range(len(tau2_basis)):
                if ir in recurrent_filter:
                    teacher.append(filtered_v[:,ir])

                    if delay_recurrent:
                        print "delay:", len(np.zeros(int(delay_recurrent/dt2)))
                        teacher[-1] = np.concatenate((np.zeros(int(delay_recurrent/dt2)), teacher[-1][int(delay_recurrent/dt2)::]))

                    teacher[-1] = teacher[-1] + np.random.normal(0, 1, np.shape(teacher[-1]) )
                    #teacher = teacher + 1*create_colnoise(times2, sexp=2, cutf=10, seed=123)
                    teacher[-1][0] = 0 # to set individual noise (not really working)

                else:
                    teacher.append(np.zeros(len(stimulus0)))

            teacher = np.concatenate((np.array(teacher)))

            print 'Setting up teacher with shape:', np.shape(teacher), 'for teacher forcing'

        if ifun is False:

            print 'Not implemented'

        else:

            pca_var = None
            pca_start = 0
            pca_max = 0
            mean_spike_freq = None
            basis_error = None
            basis_vaf = None


            if celltype[0] == "ifun":

                exec cellimport[0]

                fdt = 1. / (dt/ms)  # modify times for C code to adjust dt

                N0 = int(N[0])
                Pr = conntype[0]['conv']
                tau = conntype[0]['tau1']/ms
                kappa = conntype[0]['w']

                T = len(t)

                r = [seed0+1, anoise[0]]
                ih = np.array([ihold[0], ihold_sigma[0], factor_celltype[0][2], conntype[0]['var']])  # ihold, ihold_sigma, propability for -1
                I = amod[0] * stimulus0

                print "- running ifun"
                z = ifun.ifun(r, N0, T, Pr, tau, kappa, ih, I)
                z = np.array(z).reshape((T,N0))

                print tau
                print T
                print fdt
                print np.shape(z)
                print np.shape(t)

                times2 = t
                signals = []
                if fdt > 1.:
                    signals.append([])
                    times2 = arange(0, times2[-1], 1*ms)
                    for zi in z.T:

                        signals[0].append( np.interp(times2, t, np.convolve(zi, np.ones(fdt)/float(fdt), mode='same')) )


                else:
                    signals.append(z.T)

                print np.shape(signals)

                if plot_all > 0:

                    plt.figure("gsyn")
                    subplot(2,1,1)
                    plt.plot(t, z[:,0:10])
                    subplot(2,1,2)
                    plt.plot(t, I)

                    plt.title("Gsyn " + simprop)
                    plt.savefig('./figs/dump/' + filename + "_gsyn" + imgf, dpi = 300)  # save it
                    plt.clf()



            elif celltype[0] == "ifun2":

                exec cellimport[0]

                N0 = np.array(N, dtype=int32)
                C = np.array([ int(conntype[0]['conv']), int(conntype[1]['conv']) ], dtype=int32)
                tau = np.array([ conntype[0]['tau1']/ms, conntype[1]['tau1']/ms, conntype[2]['tau1']/ms ])
                kappa = np.array([ conntype[0]['w'], conntype[1]['w'], conntype[2]['w'] ])
                T = len(t)  #int(t[-1]/ms+1)

                r = [seed0+1, anoise[0], anoise[1]]
                ih = np.array([ihold[0], ihold_sigma[0], factor_celltype[0][2], ihold[1], ihold_sigma[1], factor_celltype[1][2], conntype[0]['var'], conntype[1]['var'], conntype[2]['var']])  # ihold, ihold_sigma, propability for -1

                I = concatenate((amod[0] * stimulus0, amod[1] * stimulus0))

                print "- running ifun2", amod
                z = ifun2.ifun2(r, N0, T, C, tau, kappa, ih, I)

                z = np.array(z).reshape((T,sum(N)))
                zgr = z[:,0:N[0]]
                zgo = z[:,N[0]:]

                if plot_all > 0:

                    plt.figure("gsyn")
                    subplot(3,1,1)
                    plt.plot(t, zgr[:,0:10])
                    subplot(3,1,2)
                    plt.plot(t, zgo[:,0:10])
                    subplot(3,1,3)
                    plt.plot(t, I[0:len(t)])

                    plt.title("Gsyn " + simprop)
                    plt.savefig('./figs/dump/' + filename + "_gsyn" + imgf, dpi = 300)  # save it
                    plt.clf()

                times2 = t
                signals = []
                signals.append(zgr.T)
                signals.append(zgo.T)


            elif celltype[0] == "ifun2re":

                exec cellimport[0]

                N0 = np.array(N, dtype=int32)
                C = np.array([ int(conntype[0]['conv']), int(conntype[1]['conv']) ], dtype=int32)
                tau = np.array([ conntype[0]['tau1']/ms, conntype[1]['tau1']/ms, conntype[2]['tau1']/ms ])
                kappa = np.array([ conntype[0]['w'], conntype[1]['w'], conntype[2]['w'] ])
                T = len(t)  #int(t[-1]/ms+1)

                r = [seed0+1, anoise[0], anoise[1]]
                ih = np.array([ihold[0], ihold_sigma[0], factor_celltype[0][2], ihold[1], ihold_sigma[1], factor_celltype[1][2],
                              conntype[0]['var'], conntype[1]['var'], conntype[2]['var'],
                              factor_recurrent[0][0], factor_recurrent[1][0],
                              factor_recurrent[0][1], factor_recurrent[1][1],
                              factor_recurrent[0][2], factor_recurrent[1][2],
                              factor_recurrent[0][3], factor_recurrent[1][3]])  # ihold, ihold_sigma, propability for -1

                I = concatenate((amod[0] * stimulus0, amod[1] * stimulus0))
                print "len(t):", T
                print "- running ifun2re", amod

                nfilt = np.concatenate(([[len(tau2_basis)], tau2_basis*1e3])).astype(int32)
                z = ifun2re.ifun2re(r, N0, T, C, tau, kappa, ih, I, rec_x, nfilt, teacher)

                z = np.array(z).reshape((T,sum(N)))
                zgr = z[:,0:N[0]]
                zgo = z[:,N[0]:]

                if plot_all > 0:

                    plt.figure("gsyn")
                    subplot(3,1,1)
                    plt.plot(t, zgr[:,0:10])
                    subplot(3,1,2)
                    plt.plot(t, zgo[:,0:10])
                    subplot(3,1,3)
                    plt.plot(t, I[0:len(t)])

                    plt.title("Gsyn " + simprop)
                    plt.savefig('./figs/dump/' + filename + "_gsyn" + imgf, dpi = 300)  # save it
                    plt.clf()

                times2 = t
                signals = []
                signals.append(zgr.T)
                signals.append(zgo.T)


            elif celltype[0] == "ifun2b":

                exec cellimport[0]

                N0 = np.array(N, dtype=int32)
                C = np.array([ int(conntype[0]['conv']), int(conntype[1]['conv']) ], dtype=int32)
                tau = np.array([ conntype[0]['tau1']/ms, conntype[1]['tau1']/ms, conntype[2]['tau1']/ms ])
                kappa = np.array([ conntype[0]['w'], conntype[1]['w'], conntype[2]['w'] ])
                T = len(t)  #int(t[-1]/ms+1)

                r = [seed0+1, anoise[0], anoise[1]]
                ih = np.array([ihold[0], ihold_sigma[0], factor_celltype[0][2], ihold[1], ihold_sigma[1], factor_celltype[1][2], conntype[0]['var'], conntype[1]['var'], conntype[2]['var'], conntype[2]['prob'], factor_celltype[0][3]])  # ihold, ihold_sigma, propability for -1

                if "_updown_" in do:
                    I = concatenate((amod[0] * stimulus0, amod[1] * stimulus0, amod[0] * stimulus1, amod[1] * stimulus1))
                else:
                    I = concatenate((amod[0] * stimulus0, amod[1] * stimulus0, -1 * amod[0] * stimulus0, -1 * amod[1] * stimulus0))

                print "- running ifun2b", amod
                z = ifun2b.ifun2b(r, N0, T, C, tau, kappa, ih, I)

                z = np.array(z).reshape((T,sum(N)))
                zgr = z[:,0:N[0]]
                zgo = z[:,N[0]:]

                if plot_all > 0:

                    plt.figure("gsyn")
                    subplot(3,1,1)
                    plt.plot(t, zgr[:,0:10])
                    subplot(3,1,2)
                    plt.plot(t, zgo[:,0:10])
                    subplot(3,1,3)
                    plt.plot(t, I[0:len(t)])

                    plt.title("Gsyn " + simprop)
                    plt.savefig('./figs/dump/' + filename + "_gsyn" + imgf, dpi = 300)  # save it
                    plt.clf()


                times2 = t
                signals = []
                signals.append(zgr.T)
                signals.append(zgo.T)


        if (myid == 0) or (do_run_constr==3):

            ##### PCA #####

            if len(ttest) > 0:
                rp = 2
            else:
                rp = 1

            color_vec2 = np.array(['Red', 'Blue'])

            if ("_pca_" in do) or ("_ica_" in do):

                filepath = data_dir + str(filename) + "_pca"

                if use_h5:
                    filepath = filepath + '.hdf5'
                else:
                    filepath = filepath + '.p'

                if do_run_constr or (os.path.isfile(filepath) is False):

                    if ("_ica_" in do):
                        do_ica = 1
                    else:
                        do_ica = 0

                    pcas = []
                    for j in range(len(N)):

                        plt.figure('pca')
                        ax_pca = []
                        for i in range(0,output_dim):
                            ax_pca.append(plt.subplot(output_dim,1,i+1))

                        plt.figure('ica')
                        ax_ica = []
                        for i in range(0,output_dim):
                            ax_ica.append(plt.subplot(output_dim,1,i+1))

                        pcas.append([])

                        run = False
                        cname = "null"

                        if ("_grc_" in do) and j==0:
                            run = True
                            cname = "grc"
                        if ("_goc_" in do) and j==1:
                            run = True
                            cname = "goc"
                        if ("_stl_" in do) and j==2:
                            run = True
                            cname = "stl"

                        if run:
                            for k in range(rp):
                                if k == 1:
                                    tstep0 = ttest[0]
                                else:
                                    tstep0 = tstep[0]

                                pop = Population(cellimport = None, do_run = 0, pickle_prefix = pickle_prefix, dt = dt, use_mpi = use_mpi, use_pc = use_pc)

                                results = pop.do_pca_ica(t_analysis_delay = tstep0-t_analysis_delay, t_analysis_stop = tstep0+t_analysis_stop, time=times2-tstep0, signals=signals, output_dim = output_dim, do_ica=do_ica, n_celltype = j)
                                t, pca, pca_var, pca_var_expl, ica = results.get('t'), results.get('pca'), results.get('pca_var'), results.get('pca_var_expl'), results.get('ica')

                                pcas[-1].append(pca)

                                if ("_poster_" in do):

                                    plt.figure("special_train_pca")

                                    import matplotlib.patches as patches
                                    rect = patches.Rectangle((0,-2725), 0.01, 3790, color=color2, alpha=0.10)
                                    rect.set_clip_on(False)
                                    ax1.add_patch(rect)

                                    for i in range(0,output_dim_plot):

                                        ax = plt.subplot(gs01[i])
                                        if i == output_dim_plot-1:
                                            adjust_spines(ax, ['bottom'])
                                            ax.xaxis.set_ticks(array([-1,-0.5,0,0.5,1,1.5,2]))
                                            ax.set_xlabel("Time (s)", labelpad=1)
                                        elif i == 0:
                                            ax.set_title("PCA components")
                                            adjust_spines(ax, []) # 'left'
                                        else:
                                            adjust_spines(ax, []) # 'left'

                                        ax.text(-(t_analysis_delay+0.07), 0, str(round(pca_var[i],3)*100)+'%', ha="center", va="center", size=params['text.fontsize'])#, bbox=bbox_props)
                                        if i == 2: ax.set_ylabel("Variance explained", labelpad=25)
                                        plt.plot(t,pca[:,i],color1, linewidth=linewidth)
                                        ax.axis(xmin=-t_analysis_delay, xmax=t_analysis_stop)

                                    plt.savefig("./figs/Pub/" + filename + "_pub.pdf", dpi = 300, transparent=True) # save it
                                    plt.savefig("./figs/Pub/" + filename + "_pub.png", dpi = 300) # save it , transparent=True
                                    #plt.clf()

                                elif plot_all > 1:

                                    for i in range(0,output_dim):

                                        plt.figure('pca')
                                        ax = ax_pca[i]
                                        if i == output_dim-1:
                                            adjust_spines(ax, ['left','bottom'])
                                        else:
                                            adjust_spines(ax, ['left'])

                                        ax.plot(t,pca[:,i], color = color_vec2[k])
                                        ax.set_ylabel(str(round(pca_var[i],4)), color = color_vec2[k])

                                        pca_starti = sum(abs(pca[0:1/dt,i])) / (1/dt)
                                        pca_start += pca_starti
                                        pca_max += max(abs(pca[:,i]))

                                        ax.set_xlabel('ms')
                                        ax.axis(xmin=-t_analysis_delay, xmax=t_analysis_stop)

                                    plt.suptitle('PCA, exp. var:' + str(pca_var_expl) + ' ' + simprop)
                                    plt.savefig('./figs/dump/' + filename + '_' + cname + '_pca' + str(k) + '.png', dpi = 300) #, transparent=True
                                    #plt.clf()

                                    if ("_ica_" in do):

                                        for i in range(0,output_dim):
                                            ax = ax_ica[i]
                                            if i == output_dim-1:
                                                adjust_spines(ax, ['left','bottom'])
                                            else:
                                                adjust_spines(ax, ['left'])
                                            ax.plot(t,ica[:,i], color = color_vec2[k])

                                            ax.set_xlabel('ms')
                                            ax.axis(xmin=t_analysis_delay-step1, xmax=t_analysis_stop-step1)

                                        plt.suptitle('ICA ' +  simprop)
                                        plt.savefig('./figs/dump/' + filename + '_' + cname + '_ica.png', dpi = 300) #, transparent=True


                        plt.clf()

                    results = {'t':t, 'pca':pca, 'pca_var':pca_var}

                    if use_h5:
                        rw_hdf5(filepath, results)
                    else:
                        pickle.dump( results, gzip.GzipFile( filepath, "wb" ) )

                else:

                    if use_h5:
                        results = rw_hdf5(filepath, export = export)
                    else:
                        results = pickle.load( gzip.GzipFile( filepath, "rb" ) )


            ##### TRAINING AND RECONSTRUCTION #####

            if ((("_basis_" in do) or ("_cnoise_" in do) or ("_noibas_" in do) or ("_basnoi_" in do)) and ("_lyap" not in do)) and (do_run_constr is not 2) and ("_pca" not in do):

                filepath = data_dir + str(filename) + "_basis"

                if use_h5:
                    filepath = filepath + '.hdf5'
                else:
                    filepath = filepath + '.p'

                if do_run_constr or (os.path.isfile(filepath) is False):

                    #dt2 = times2[1]-times2[0]

                    #print "dt2:", dt2

                    # t_in = times2[int(t_startstop0[0]/dt2):int(t_startstop0[1]/dt2)]
                    # t_test = times2[int(t_startstop[0]/dt2):int(t_startstop[1]/dt2)]
                    # t_basis = times2[int((t_startstop0[1]+4)/dt2):int((t_startstop0[1]+10)/dt2)]
                    #t_basis = times2

                    c = 0
                    for j in range(len(N)):

                        use=False
                        if (("_grc_" in do) and j==0): use=True; fac=1
                        if (("_goc_" in do) and j==1): use=True; fac=1
                        if (("_stl_" in do) and j==2): use=True; fac=-1 # output is inhibitory
                        if (("_ubc_" in do) and j==2): use=True; fac=1
                        #if (("_mf_" in do) and j==2): use=True; fac=1

                        if use:

                            if mean_spike_freq is not None:
                                useonly = np.array(np.where((mean_spike_freq[j]>0) & (mean_spike_freq[j]<max_freq))[0], dtype=int).tolist()
                                sig = np.array(signals[j])
                                est_in0 = fac*np.array(sig[useonly]).T[int(t_startstop0[0]/dt2):int(t_startstop0[1]/dt2),::downsample]
                                test_in0 = fac*np.array(sig[useonly]).T[int(t_startstop[0]/dt2):int(t_startstop[1]/dt2),::downsample]
                                basis_in0 = fac*np.array(sig[useonly]).T[int((t_startstop0[1]+4)/dt2):int((t_startstop0[1]+10)/dt2),::downsample]
                                sig_in0 = fac*np.array(sig[useonly]).T[:,::downsample]

                            else:
                                sig = np.array(signals[j])
                                est_in0 = fac*np.array(sig).T[int(t_startstop0[0]/dt2):int(t_startstop0[1]/dt2),::downsample]
                                test_in0 = fac*np.array(sig).T[int(t_startstop[0]/dt2):int(t_startstop[1]/dt2),::downsample]
                                basis_in0 = fac*np.array(sig).T[int((t_startstop0[1]+4)/dt2):int((t_startstop0[1]+10)/dt2),::downsample]
                                sig_in0 = fac*np.array(sig).T[:,::downsample]


                            print "- shape(est_in0):", shape(est_in0), ", shape(sig_in0):", shape(sig_in0)

                            if noise_out > 0:
                                est_in0 = est_in0 + np.random.normal(0, noise_out, np.shape(est_in0) )
                                test_in0 = test_in0 + np.random.normal(0, noise_out, np.shape(test_in0) )
                                basis_in0 = basis_in0 + np.random.normal(0, noise_out, np.shape(basis_in0) )
                                sig_in0 = sig_in0 + np.random.normal(0, noise_out, np.shape(sig_in0) )
                            c += 1

                            if c > 1:
                                est_in = np.append(est_in, est_in0, axis=1)
                                test_in = np.append(test_in, test_in0, axis=1)
                                basis_in = np.append(basis_in, basis_in0, axis=1)
                                sig_in = np.append(sig_in, sig_in0, axis=1)

                            else:
                                est_in = est_in0
                                test_in = test_in0
                                basis_in = basis_in0
                                sig_in = sig_in0

                    dn = 0

                    # calculate mean rate instead, especially for ifun models
                    if mean_spike_freq is None:
                        mean_spike_freq = []
                        for n in range(len(N)):
                            mean_spike_freq.append(zeros(N[n]))
                            sig = np.array(signals[n])
                            test_in0 = np.array(sig).T[int(t_startstop[0]/dt2):int(t_startstop[1]/dt2),::downsample]
                            for i in range(N[n]):
                                mean_spike_freq[n][i] = mean(test_in0[:,i])


                    print "- Training"

                    if fitter == "ridge":
                        print "Fitter: Ridge"
                        regr = linear_model.Ridge(alpha=ridge_alpha)
                        regr.fit(est_in,filtered_est_v)
                        x = regr.coef_.T

                    elif fitter == "lasso":
                        if (ire != 1) and (ire != 3):
                            print "Fitter: Lasso"
                            regr = linear_model.Lasso(alpha=ridge_alpha) #, max_iter=1 , warm_start=True, max_iter=1) , normalize=True, , warm_start=True
                            #print est_in
                            regr.fit(est_in,filtered_est_v)
                        x = regr.coef_.T
                        #print regr.n_iter_

                    elif fitter == "lassocv":
                        print "Fitter: Lasso CV"
                        regr = linear_model.MultiTaskLassoCV(n_alphas=10, eps=0.0001)
                        regr.fit(est_in,filtered_est_v)
                        x = regr.coef_.T
                        print regr.mse_path_
                        print regr.alphas_
                        print regr.alpha_

                    elif fitter == "lassolarscv":
                        print "Fitter: Lasso Lars CV"
                        regr = linear_model.LassoLarsCV()
                        regr.fit(est_in,filtered_est_v)
                        x = regr.coef_.T
                        print regr.alphas_
                        print regr.alpha_

                    elif fitter == "lassopos":
                        print "Fitter: positive Lasso"
                        regr = linear_model.Lasso(alpha=ridge_alpha, positive=True)
                        regr.fit(est_in,filtered_est_v)
                        x = regr.coef_.T

                    elif fitter == "elastic":
                        if (ire != 1) and (ire != 3):
                            print "Fitter: Elastic"
                            regr = linear_model.ElasticNet(alpha=ridge_alpha, l1_ratio=l1r, max_iter=1000)
                            regr.fit(est_in,filtered_est_v)
                        x = regr.coef_.T

                    elif fitter == "elasticcv":
                        print "Fitter: Elastic CV"
                        regr = linear_model.MultiTaskElasticNetCV(l1_ratio=[0.1,0.5,0.7,0.9,0.95,0.99,1])
                        regr.fit(est_in,filtered_est_v)
                        x = regr.coef_.T
                        print regr.mse_path_
                        print regr.alphas_
                        print regr.l1_ratio_
                        print regr.alpha_

                    elif fitter == "elasticpos":
                        print "Fitter: positive Elastic"
                        regr = linear_model.ElasticNet(alpha=ridge_alpha, l1_ratio=l1r, max_iter=1000, positive=True)
                        regr.fit(est_in,filtered_est_v)
                        x = regr.coef_.T

                    uest = regr.predict(est_in) #dot(est_in,x)
                    #x[find(abs(x)>max_weight)]=0
                    utest = regr.predict(test_in) #dot(test_in,x)
                    ubasis = regr.predict(basis_in) #dot(basis_in,x)
                    usig = regr.predict(sig_in)


                    if shape(filtered_test_v)[1] == 1:
                        utest = np.array([utest]).T
                        uest = np.array([uest]).T
                        ubasis = np.array([ubasis]).T
                        usig = np.array([usig]).T
                        x = np.array([x]).T


                    print "- Finished"

                    results = {'t_in':t_in, 't_test':t_test, 't_basis':t_basis, 'x':x, 'filtered_est_v':filtered_est_v, 'filtered_test_v':filtered_test_v, 'filtered_basis_v':filtered_basis_v, 'uest':uest, 'utest':utest, 'ubasis':ubasis, 'est':est, 'test':test, 'basis':basis, 'basis_in':basis_in[:,0:100]}

                    if ire == run_recurrent_filter-1:
                        if use_h5:
                            rw_hdf5(filepath, results)
                        else:
                            pickle.dump( results, gzip.GzipFile( filepath, "wb" ) )

                else:

                    if use_h5:
                        results = rw_hdf5(filepath, export = export)
                    else:
                        results = pickle.load( gzip.GzipFile( filepath, "rb" ) )

                    t_in, t_test, t_basis, x  = results.get('t_in'), results.get('t_test'), results.get('t_basis'), results.get('x')
                    est, test, basis = results.get('est'), results.get('test'), results.get('basis')
                    filtered_est_v, filtered_test_v, filtered_basis_v, uest, utest, ubasis = results.get('filtered_est_v'), results.get('filtered_test_v'), results.get('filtered_basis_v'), results.get('uest'), results.get('utest'), results.get('ubasis')

                n = shape(filtered_test_v)[1]

                basis_error = []
                basis_vaf = []

                for i in range(n):
                    vaf = 1 - ( var(filtered_test_v[:,i]-utest[:,i]) / var(filtered_test_v[:,i] ) ) # 1 - ( sum((filtered_test_v[:,i]-utest[:,i])**2) / sum(filtered_test_v[:,i]**2) )
                    if vaf < 0: vaf = 0
                    error = (scipy.stats.pearsonr(utest[:,i], filtered_test_v[:,i])[0])**2

                    basis_error.append(error)
                    basis_vaf.append(vaf)
                    print "Basis error:", basis_error[-1], "vaf:", basis_vaf[-1], "weight range:", max(x[:,i]), min(x[:,i]), "mean:", mean(abs(np.array([a for a in x[:,i] if a != 0]))), "zero weight percentage:", float(len(np.where(x[:,i]==0)[0]))/len(x[:,i])*100

                print "Basis error all:", mean(basis_error), "vaf all:", mean(basis_vaf)


                if (run_recurrent_filter > 1) or exprecx:

                    print '- Recurrent run: ', ire

                    rec_x = []
                    for ir in range(len(tau2_basis)):
                        if ir in recurrent_filter:
                            rec_x.append(x[:,ir])
                        else:
                            rec_x.append(np.zeros(N[0]))
                    rec_x = np.concatenate((rec_x))

                    print '- Updating learned weights, shape:', np.shape(rec_x)

                if exprecx:

                    filepath_recx = data_dir + str(slugify(pickle_prefix_ly)) + "_seed" + str(seed0) + '_recx.hdf5'
                    results_recx = {'rec_x':rec_x}
                    rw_hdf5(filepath_recx, results_recx)
                    print filepath_recx

                if teacher_forcing and (ire == 0):
                    teacher = np.zeros(len(teacher))
                    print '- Removing teacher forcing, nullvec shape:', np.shape(teacher)


        if ((("_basis_" in do) or ("_cnoise_" in do) or ("_noibas_" in do) or ("_basnoi_" in do)) and ("_lyap" not in do)) and (do_run_constr is not 2) and ("_pca" not in do):

            if ("_poster_" in do):

                plt.figure("special_basis")
                for i, a in enumerate([ax2, ax3, ax4]):

                    basis = basis_v[i]
                    uest = uest_v[i]
                    utest = utest_v[i]
                    basis_test = basis_test_v[i]
                    stimulus2 = stimulus2_v[i]
                    stimulus3 = stimulus3_v[i]
                    ubasis = ubasis_v[i]
                    basis_test2 = basis_test2_v[i]
                    stimulus4 = stimulus4_v[i]
                    x = x_v[i]

                    print int((1-0.4)/dt2),int(3/dt2)
                    l1 = a.plot((t_basis-(t_startstop0[1]+5))[int((1-0.4)/dt2):int(3/dt2)], stimulus4[int((1-0.4)/dt2):int(3/dt2)], ':', color='black', linewidth=1, clip_on = False)
                    l2 = a.plot((t_basis-(t_startstop0[1]+5))[int((1-0.4)/dt2):int(3/dt2)], (basis_test2/max(basis_test2))[int((1-0.4)/dt2):int(3/dt2)], '-', color=color2, linewidth=linewidth, clip_on = False)
                    l3 = a.plot((t_basis-(t_startstop0[1]+5))[int((1-0.4)/dt2):int(3/dt2)], (ubasis/max(basis_test2))[int((1-0.4)/dt2):int(3/dt2)], color=color1, linewidth=linewidth, clip_on = False)
                    a.axis(xmin=-0.4, xmax=1, ymin=-0.7, ymax=1.5)


                    if i == 2:
                        adjust_spines(a, ['left','bottom'], d_out = 10, d_down = 10)
                        a.set_xlabel("s", labelpad=0)
                        a.set_ylabel("a.u.", labelpad=0)
                    else:
                        adjust_spines(a, ['left'], d_out = 10, d_down = 10)
                        a.set_ylabel("a.u.", labelpad=0)

                    a.yaxis.set_ticks(array([-0.5,0,0.5,1]))
                    a.set_yticklabels(('','0','','1'))

                    if i == 0:
                        a.text(-0.2, 0.9, r"$\tau$ = 10 ms", ha="center", va="center", size=params['title.fontsize'])
                        a.text(0.25, 1.6, r"Impulse responses of constructed basis functions", ha="center", va="center", size=params['title.fontsize'])
                    if i == 1:
                        a.text(-0.2, 0.9, r"$\tau$ = 100 ms", ha="center", va="center", size=params['title.fontsize'])
                        a.text(0, 1.5, r"signal", ha="center", va="center", size=params['title.fontsize'], color="black")
                        a.text(0.6, -0.5, r"filtered signal", ha="center", va="center", size=params['title.fontsize'], color=color2)
                        a.text(0.3, 1, r"constructed response", ha="center", va="center", size=params['title.fontsize'], color=color1)
                    if i == 2:
                        a.text(-0.2, 0.9, r"$\tau$ = 500 ms", ha="center", va="center", size=params['title.fontsize'])

                #if use_pc is False:
                plt.savefig("./figs/Pub/" + filename + "_basis.pdf", dpi = 300, transparent=True) # save it
                plt.savefig("./figs/Pub/" + filename + "_basis" + imgf, dpi = 300) # save it , transparent=True


                plt.figure("special_cnoise")
                for i, a in enumerate([[ax11,ax12], [ax21,ax22], [ax31,ax32]]):

                    basis = basis_v[i]
                    uest = uest_v[i]
                    utest = utest_v[i]
                    basis_test = basis_test_v[i]
                    stimulus2 = stimulus2_v[i]
                    stimulus3 = stimulus3_v[i]
                    ubasis = ubasis_v[i]
                    basis_test2 = basis_test2_v[i]
                    stimulus4 = stimulus4_v[i]
                    x = x_v[i]

                    print "smaller than 0.1:", len(find(abs(x)<0.1))
                    bins = concatenate(( np.arange(-19.9,-0.1,0.2), [-0.1] ,np.arange(0.1,20,0.2) ))
                    a[1].set_yscale('log')

                    #bins = np.arange(-20,20,0.1)
                    n, bins, patches = a[1].hist(x, bins, histtype='bar', facecolor='k') #normed=1

                    a[0].plot(t_test-(t_startstop[1]-2.5), stimulus3, ':', color='black', linewidth=1)
                    a[0].plot(t_test-(t_startstop[1]-2.5), basis_test/max(basis_test2), '-', color=color2, linewidth=linewidth)
                    a[0].plot(t_test-(t_startstop[1]-2.5), utest/max(basis_test2), color=color1, linewidth=linewidth)
                    a[0].axis(ymin=-1.5, ymax=1.5, xmin=0, xmax=1)

                    if i == 2:
                        adjust_spines(a[0], ['left','bottom'], d_out = 10, d_down = 10)
                        a[0].set_xlabel("s", labelpad=0)
                        a[0].set_ylabel("a.u.", labelpad=0)
                        adjust_spines(a[1], ['left','bottom'], d_out = 10, d_down = 10)
                    else:
                        adjust_spines(a[0], ['left'], d_out = 10, d_down = 10)
                        a[0].set_ylabel("a.u.", labelpad=0)
                        adjust_spines(a[1], ['left'], d_out = 10, d_down = 10)

                    a[0].yaxis.set_ticks(array([-1.5,-1,-0.5,0,0.5,1,1.5]))
                    a[0].set_yticklabels(('','-1','','0','','1',''))

                    a[1].yaxis.set_ticks(array([1,2,3,4,5,6,7,8,9, 10, 20, 30, 40, 50])*5)
                    a[1].set_yticklabels(('1%','','','','','','','','','10%','', '', '', '50%'))
                    a[1].xaxis.set_ticks(array([-20,-15,-10,-5,0,5,10,15,20]))
                    a[1].set_xticklabels(('-20','','-10','','0','','10','','20'))
                    a[1].axis(ymax=250)

                    if i == 0:
                        a[0].set_title(r"$\tau$ = 10 ms")
                        a[1].set_title(r"Weight distribution")
                        a[0].text(0.15, 1.2, r"signal", ha="center", va="center", size=params['title.fontsize'], color="black")
                        a[0].text(0.2, -1.1, r"filtered signal", ha="center", va="center", size=params['title.fontsize'], color=color2)
                        a[0].text(0.7, 1.2, r"constructed response", ha="center", va="center", size=params['title.fontsize'], color=color1)
                    if i == 1:
                        a[0].set_title(r"$\tau$ = 100 ms")
                    if i == 2:
                        a[0].set_title(r"$\tau$ = 500 ms")


                #if use_pc is False:
                plt.savefig("./figs/Pub/" + filename + "_cnoise.pdf", dpi = 300, transparent=True) # save it
                plt.savefig("./figs/Pub/" + filename + "_cnoise" + imgf, dpi = 300) # save it , transparent=True


            elif plot_all > 0:

                plt.figure('weights')
                plt.clf()
                gs1 = matplotlib.gridspec.GridSpec(n, 1,
                   width_ratios=[1],
                   height_ratios=[1]*n
                   )
                gs1.update(bottom=0.13, top=0.97, left=0.1, right=0.99, wspace=0.4, hspace=0.6)

                plt.figure('construct+test')
                plt.clf()
                gs2 = matplotlib.gridspec.GridSpec(n, 3,
                   width_ratios=[1,1,1],
                   height_ratios=[1]*n
                   )
                gs2.update(bottom=0.13, top=0.97, left=0.1, right=0.99, wspace=0.4, hspace=0.6)

                t_in, t_test, t_basis, x  = results.get('t_in'), results.get('t_test'), results.get('t_basis'), results.get('x')
                filtered_est_v, filtered_test_v, filtered_basis_v, uest, utest, ubasis = results.get('filtered_est_v'), results.get('filtered_test_v'), results.get('filtered_basis_v'), results.get('uest'), results.get('utest'), results.get('ubasis')
                est, test, basis = results.get('est'), results.get('test'), results.get('basis')

                for i in range(n):

                    filtered_est = filtered_est_v[:,i]
                    filtered_test = filtered_test_v[:,i]
                    filtered_basis = filtered_basis_v[:,i]

                    uest1 = uest[:,i]
                    utest1 = utest[:,i]
                    ubasis1 = ubasis[:,i]

                    x1 = x[i]

                    if False:
                        if not isnan(sum(basis_error)):
                            plt.figure('weights')
                            ax = plt.subplot(gs1[i,0])
                            ax.set_yscale('log')

                            bins = np.arange(-80,80,0.1)
                            print shape(x)
                            print x
                            n, bins, patches = ax.hist(x, bins, histtype='bar')

                            plt.savefig('./figs/dump/' + filename + '_basis_weights' + imgf, dpi = 300) #, transparent=True


                    plt.figure('construct+test')
                    ax_construct = plt.subplot(gs2[i,0])
                    ax_construct.plot(t_in, filtered_est/max(abs(filtered_basis)), 'k--')
                    ax_construct.plot(t_in, est, 'k:')
                    ax_construct.plot(t_in, uest1/max(abs(filtered_basis)), color = "b")
                    #ax_construct.axis(ymin=-1.2, ymax=1.2)


                    ax_test = plt.subplot(gs2[i,1])
                    ax_test.plot(t_test, filtered_test/max(abs(filtered_basis)), 'k--')
                    ax_test.plot(t_test, test, 'b:')
                    ax_test.plot(t_test, utest1/max(abs(filtered_basis)), color = "r")
                    #ax_test.axis(ymin=-1.2, ymax=1.2)
                    plt.title("Basis error: " + str(basis_error[i]) + "vaf: " + str(basis_vaf[i]) )
                    plt.savefig('./figs/dump/' + filename + '_basis' + imgf, dpi = 300) #, transparent=True


                    ax_test = plt.subplot(gs2[i,2])
                    ax_test.plot(t_basis, filtered_basis/max(abs(filtered_basis)), 'k--')
                    ax_test.plot(t_basis, basis, 'b:')
                    ax_test.plot(t_basis, ubasis1/max(abs(filtered_basis)), color = "r")
                   # ax_test.axis(ymin=-1.2, ymax=1.2)


                plt.savefig('./figs/dump/' + filename + '_basis' + imgf, dpi = 300) #, transparent=True


            elif ("_updown_" in do):

                dt2 = times2[1]-times2[0]
                t_basis = times2[int(t_startstop[0]/dt2):int(t_startstop[1]/dt2)]

                sig = np.array(signals[1])
                basis_in = np.array(sig).T[int((t_startstop[0])/dt2):int((t_startstop[1])/dt2),::downsample]

                results = {'t_basis':t_basis, 'basis_in':basis_in[:,0:100]}

                plt.figure('signals')

                ii=0
                #basis_in = basis_in[:,0:100]
                li = np.shape(basis_in)[1]
                for i in range(li):
                    basis_in1 = basis_in[:,i]
                    basis_in2 = basis_in[:,i]-basis_in[0,i]
                    rate_vec1 = [basis_in1[(9-4)/dt2], basis_in1[(14-4)/dt2], basis_in1[(19-4)/dt2], basis_in1[(24-4)/dt2], basis_in1[(29-4)/dt2]]
                    rate_vec2 = [basis_in2[(9-4)/dt2], basis_in2[(14-4)/dt2], basis_in2[(19-4)/dt2], basis_in2[(24-4)/dt2], basis_in2[(29-4)/dt2]]
                    pos_vec = [-20, -10, 0, 10, 20]

                    ax1 = plt.subplot(4,2,1)
                    ax2 = plt.subplot(4,2,2)

                    if rate_vec2[0]>0 and rate_vec2[4]>0:
                        ax1 = plt.subplot(4,2,1)
                        ax2 = plt.subplot(4,2,2)
                    if rate_vec2[0]>0 and rate_vec2[4]<0:
                        ax1 = plt.subplot(4,2,3)
                        ax2 = plt.subplot(4,2,4)
                    if rate_vec2[0]<0 and rate_vec2[4]>0:
                        ax1 = plt.subplot(4,2,5)
                        ax2 = plt.subplot(4,2,6)
                    if rate_vec2[0]<0 and rate_vec2[4]<0:
                        ax1 = plt.subplot(4,2,7)
                        ax2 = plt.subplot(4,2,8)

                    ax1.plot(t_basis, basis_in2, color=cm.jet(1.*i/li))
                    ax2.plot(pos_vec, rate_vec1, '*-', color=cm.jet(1.*i/li))
                    #plt.plot(t_basis, basis_in1, color=cm.jet(1.*i/li))

                plt.savefig('./figs/dump/' + filename + '_signals' + imgf, dpi = 300) #, transparent=True


    if do_run: barrier(use_mpi, use_pc)

    if ifun is False:
        if do_run:
            pop.delall()
        #del pop
        pop = None


    return pca_var, pca_start, pca_max, mean_spike_freq, basis_error, basis_vaf, signals, times2, x


def lyap(params0, runs = 1):

    params = params0.copy() #copy.deepcopy(params0)

    delta = 0
    len_train = 5
    len_test = 1
    t_analysis_delay = 1
    t_analysis_stop = 1

    dt = params['dt']
    use_mpi = params['use_mpi']
    use_pc = params['use_pc']
    myid = params['myid']
    data_dir = params['data_dir']

    if use_pc:
        imgf = ".svg"
    else:
        imgf = ".png"

    params['t_analysis_delay'] = t_analysis_delay
    params['t_analysis_stop'] = t_analysis_stop

    # cnoise height
    istep0 = np.array([1])
    istep1 = np.zeros(len(istep0))
    istep = [istep0,istep1]
    istep_sigma = [0, 0]

    itest0 = np.array([1])
    itest1 = np.zeros(len(itest0))
    itest = [itest0,itest1]
    itest_sigma = [0, 0]

    params['istep'] = istep
    params['istep_sigma'] = istep_sigma
    params['itest'] = itest
    params['itest_sigma'] = itest_sigma

    # cnoise time, start stop vectors!
    tstep = np.array([0,len_train]) + delta
    ttest = tstep[-1] + np.array([0,len_test])
    tstop = ttest[-1]

    params['tstep'] = tstep
    params['ttest'] = ttest
    params['tstop'] = tstop

    t_plot_delay = 0.1
    t_plot_stop = tstop

    params['t_plot_delay'] = t_plot_delay
    params['t_plot_stop'] = t_plot_stop

    uselen=5500
    eucd_v = np.zeros((uselen,runs))

    pickle_prefix = params['pickle_prefix']
    do = params['do']

    for i in range(runs):

        #print do
        #print pickle_prefix

        params['seed0'] = i+1
        params['do'] = do + "lyap0_"
        params['pickle_prefix'] =  pickle_prefix + "_lyap0"
        pca_var, pca_start, pca_max, mean_spike_freq, basis_error, basis_vaf, signals0, t, _ = randomnet(params)
        params['do'] = do + "lyap1_"
        params['pickle_prefix'] = pickle_prefix + "_lyap1"
        pca_var, pca_start, pca_max, mean_spike_freq, basis_error, basis_vaf, signals1, t, _ = randomnet(params)

        if (use_mpi is False) or (myid == 0):
            eucd = np.sqrt(np.sum((np.array(signals0[0])-np.array(signals1[0]))**2, axis=0))[0:uselen] # Euclidian difference
            eucd_v[:,i] = eucd

    ly = None

    if (use_mpi is False) or (myid == 0):
        eucd_m = np.mean(eucd_v, axis = 1)
        dt2 = 1e-3
        Dt = 2*s
        h0 = mean(eucd_m[((2.01)/dt2):(2.11/dt2)])
        ht = mean(eucd_m[((2.01+Dt)/dt2):((2.11+Dt)/dt2)])

        ly = np.log(ht/h0)/Dt

        print h0, ht, " Lyapunov = ", ly

        filepath = data_dir + pickle_prefix + "_lyap.hdf5"
        lyd = {}
        lyd['ly'] = ly
        rw_hdf5(filepath, lyd)

        plt.figure("eucd")
        subplot(2,1,2)
        plt.plot(t[0:uselen], eucd_m)
        plt.savefig("./figs/dump/" + pickle_prefix + "_eucd" + imgf, dpi = 300)  # save it
        plt.clf()


    return ly
