# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 17:01:25 2013

@author: chris

"""

from __future__ import with_statement
from __future__ import division

import sys
argv = sys.argv

use_mpi = False
if "-nompi" not in argv:
    use_mpi = True

if use_mpi:

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        myid = comm.rank
        psize = comm.size

    except ImportError:
        use_mpi = False
        myid = 0

else:
    use_mpi = False
    myid = 0

imgf = ".png"
use_pc = False

import sys
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-o', action='store', dest='opt')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--norun', action='store_true')
parser.add_argument('--noconst', action='store_true')
parser.add_argument('--noqual', action='store_true')
parser.add_argument('--pickle', action='store_true')
parser.add_argument('-nompi', action='store_true')
pars = parser.parse_args()

do_run = 1
if pars.norun:  # do not run again use pickled files!
    print "- Not running, using saved files"
    do_run = 0

do_run_constr = 1
if pars.noconst:  # do not run again use pickled files!
    print "- Not construction"
    do_run_constr = 0

do_plot = 0
if pars.plot:  # do not plot to windows
    if myid == 0: print "- Plotting"
    do_plot = 1

use_h5 = True
if pars.pickle:  # do not plot to windows
    if myid == 0: print "- Use pickle"
    use_h5 = False

opt = pars.opt

import matplotlib
if myid == 0:
    if do_plot == 1:
        matplotlib.use('TkAgg', warn=True)
    else:
        matplotlib.use('Agg', warn=True)
else:
    matplotlib.use('Agg', warn=True)


use_c = False
export = False
export_m = False

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import random as rnd

from matplotlib.font_manager import FontProperties
font0 = FontProperties()

if do_run == 0:
    pass
else:
    from Randomnet import *
    
from Stimhelp import *
from Plotter import *
from units import *

from itertools import izip

try:
    import cPickle as pickle
except:
    import pickle

import gzip
import h5py
import math

import scipy
from scipy import io
    
from scipy.optimize import fmin, leastsq, anneal, brute, fmin_bfgs, fmin_l_bfgs_b, fmin_slsqp, fmin_tnc, basinhopping


def broadcast(vec, root = 0):
    if use_mpi:
        vec = comm.bcast(vec, root=0)
    return vec

dt = 0.1*ms
#dt = 0.025*ms

plot_all = 3
plot_all = 1
runs = 1

data_dir = "./data/"

if opt == "ifun":
    do_vec = np.array([
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.4_ihsigma0.1_lentrain10_0wv1.4_ax1_run_",
                      ])

# FIGURE 2
if opt == "fig1":
    do_run_constr = 0
    do_vec = np.array([

                       #"randomnet_ifun_pca_grc_0N1000_0tau1v100_amod0.1_alpha10_0conv0.4_ihsigma0.1_lentrain10_0wendv2.0_pcaplot_first_colgrlight_pos2_",
                       #"randomnet_ifun_pca_grc_0N1000_0tau1v100_amod0.1_alpha10_0conv0.4_ihsigma0.1_lentrain10_0wendv0.6_pcaplot_pos3_",
                       #"randomnet_ifun_pca_grc_0N1000_0tau1v100_amod0.1_alpha10_0conv0.4_ihsigma0.1_lentrain10_0wendv0.6_noinv_pcaplot_pos4_",

                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.4_ihsigma0.1_lentrain10_0wendv0.01_sigplot_specy1_first_pos2_",
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.4_ihsigma0.1_lentrain10_0wendv1.4_colgrlight_sigplot_first_pos3_",
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.4_ihsigma0.1_lentrain10_0wendv3.0_sigplot_first_pos4_",
                       #"randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.4_ihsigma0.1_lentrain10_0wendv1.4_noinv_sigplot_end_pos4_",

                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.4_ihsigma0.1_lentrain10_0wendv0.01_dotted_colgrlight_first_runplot_pos1_", # _l1r0.5 #
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.4_ihsigma0.1_lentrain10_0wendv1.4_runplot_pos1_", # _l1r0.5
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.4_ihsigma0.1_lentrain10_0wendv3.0_colgrlight_runplot_end_pos1_", #_l1r0.5

                       #"randomnet_ifun_pca_grc_0N1000_0tau1v50_amod0.1_alpha10_0conv0.4_ihsigma0.1_lentrain10_0wendv2.5_pcaplot_colgrlight_pos3_",
                       #"randomnet_ifun_pca_grc_0N1000_0tau1v50_amod0.1_alpha10_0conv0.4_ihsigma0.1_lentrain10_0wendv1.5_pcaplot_end_pos2_",
                       #"randomnet_ifun_pca_grc_0N1000_0tau1v50_amod0.1_alpha10_0conv0.4_ihsigma0.1_lentrain10_0wendv1.5_noinv_pcaplot_dotted_end_pos3_",
                       #"randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha10_0conv0.4_ihsigma0.1_lentrain10_0wendv2.5_colgrlight_runplot_pos1_",
                       #"randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha10_0conv0.4_ihsigma0.1_lentrain10_0wendv1.5_runplot_end_pos1_",


                      ])
    data_dir = "./publish/closedloop/data/"
    #export_m = "./export"



# FIGURE 3
if opt == "fig2":
    #do_run_constr = 0
    do_vec = np.array([
                       #"randomnet_ifun_cnoise_grc_0N1000_0tau1v10_amod0.1_alpha0.0001_0conv0.4_ihsigma0.1_lentrain10_fitlassopos_vec0w_ax1_first_lyline_colgrlight_continue_usemean_",
                       #"randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_0conv0.4_ihsigma0.1_lentrain10_fitlassopos_vec0w_ax2_lyline_colgrlight_continue_usemean_",
                       #"randomnet_ifun_cnoise_grc_0N1000_0tau1v100_amod0.1_alpha0.0001_0conv0.4_ihsigma0.1_lentrain10_fitlassopos_vec0w_ax3_lyline_colgrlight_continue_usemean_",

                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_0conv0.4_ihsigma0.1_lentrain10_fitlasso_vec0w_ax2_lyline_continue_usemean_",
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v10_amod0.1_alpha0.0001_0conv0.4_ihsigma0.1_lentrain10_fitlasso_vec0w_ax1_first_lyline_continue_usemean_",
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v100_amod0.1_alpha0.0001_0conv0.4_ihsigma0.1_lentrain10_fitlasso_vec0w_ax3_lyline_continue_usemean_",
                      ])
    data_dir = "./publish/closedloop/data/"
    #export_m = "./export"
    runs = 10

# check for best alpha
if opt == "fig2alpha":
    #do_run_constr = 0
    do_vec = np.array([
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0000001_0conv0.4_ihsigma0.1_lentrain10_fitlasso_vec0w_ax2_relalpha_dotted_lyline_continue_usemean_",
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.000001_0conv0.4_ihsigma0.1_lentrain10_fitlasso_vec0w_ax2_relalpha_dashed_lyline_continue_usemean_",
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.00001_0conv0.4_ihsigma0.1_lentrain10_fitlasso_vec0w_ax2_relalpha_dashdot_lyline_continue_usemean_",
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_0conv0.4_ihsigma0.1_lentrain10_fitlasso_vec0w_ax2_relalpha_lyline_continue_usemean_",
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.001_0conv0.4_ihsigma0.1_lentrain10_fitlasso_vec0w_ax2_relalpha_colgrlight_lyline_continue_usemean_",
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.01_0conv0.4_ihsigma0.1_lentrain10_fitlasso_vec0w_ax2_relalpha_colgrlight_dotted_lyline_continue_usemean_",
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.1_0conv0.4_ihsigma0.1_lentrain10_fitlasso_vec0w_ax2_relalpha_colgrlight_dashed_lyline_continue_usemean_",
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha1_0conv0.4_ihsigma0.1_lentrain10_fitlasso_vec0w_ax2_relalpha_colgrlight_dashdot_lyline_continue_usemean_", # _relalpha
                      ])
    data_dir = "./publish/closedloop/data/"
    #export_m = "./export"
    runs = 10



# FIGURE 4
if opt == "fig3":
    do_vec = np.array([
                       #"randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.4_ihsigma0.1_lentrain10_vec0w_ax1_lyline_specy1_colgrlight_first_continue_usemean_",
                       #"randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.4_ihsigma0.1_lentrain10_noinv_vec0w_ax1_lyline_specy1_first_continue_usemean_",

                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.4_ihsigma0.1_lentrain10_vec0w_ax1_lyline_specy1_colgrlight_first_continue_usemean_",
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_anoise0.05_alpha0.0001_fitlasso_0conv0.4_ihsigma0.1_lentrain10_vec0w_ax1_lyline_specy1_first_continue_usemean_",

                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.4_ihsigma0.1_lentrain10_vec0w_ax2_lyline_colgrlight_specy1_continue_usemean_",
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.4_ihsigma2_lentrain10_vec0w_ax2_lyline_specy1_continue_usemean_",
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.4_0varw2_ihsigma0.1_lentrain10_vec0w_ax2_lyline_dotted_specy1_relyap_continue_usemean_",

                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.4_ihsigma0.1_lentrain10_normp_vec0w_specx1_ax3_lyline_specy1_colgrlight_continue_usemean_",
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.04_ihsigma0.1_lentrain10_normp_vec0w_specx1_ax3_lyline_specy1_dotted_continue_usemean_",
                       "randomnet_ifun_cnoise_grc_0N10000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.04_ihsigma0.1_lentrain10_normp_vec0w_specx1_ax3_lyline_specy1_continue_usemean_",

                       ])
    data_dir = "./publish/closedloop/data/"
    #export_m = "./export"
    runs = 10


# FIGURE 5
if opt == "fig3b":

    do_run_constr = 0
    do_vec = np.array([

                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.4_ihsigma0.1_lentrain10_0wendv1.4_noinv_sigplot_first_end_pos1_",

                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.4_ihsigma0.1_lentrain10_vec0w_ax2_lyline_specy1_colgrlight_first_noly_continue_usemean_",
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.4_ihsigma0.1_lentrain10_noinv_vec0w_ax2_lyline_specy1_first_noly_continue_usemean_",

                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlassopos_0conv0.4_ihsigma0.1_lentrain10_vec0w_ax3_lyline_specy1_first_colgrlight_continue_usemean_",
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlassopos_0conv0.4_ihsigma0.1_lentrain10_noinv_vec0w_ax3_lyline_specy1_first_continue_usemean_",

                       ])
    data_dir = "./publish/closedloop/data/"
    #export_m = "./export"
    runs = 10


# FIGURE 6
if opt == "fig4lruntest":
    do_vec = np.array([
                       "randomnet_ifun2_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv0.1_ihsigma0.1_lentrain10_vec0w_ax1_first_lyline_title1_continue_usemean_",
                      ])

    #data_dir = "./publish/closedloop/data/"
    #export_m = "./export"
    runs = 3
    

if opt == "fig4l":
    do_vec = np.array([

                       #"randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.4_ihsigma0_lentrain10_vec0w_ax1_first_lyline_colgrlight_dotted_continue_usemean_",
                       #"randomnet_ifun2_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv0.1_ihsigma0_lentrain10_vec0w_ax1_first_lyline_title1_dotted_continue_usemean_",

                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.4_ihsigma0.1_lentrain10_vec0w_ax1_first_lyline_colgrlight_continue_usemean_",
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.4_ihsigma0.1_lentrain10_vec0w_ax2_lyline_colgrlight_continue_usemean_",
                       "randomnet_ifun_cnoise_grc_0N1000_0tau1v50_amod0.1_alpha0.0001_fitlasso_0conv0.4_ihsigma0.1_lentrain10_vec0w_ax3_lyline_colgrlight_continue_usemean_",

                       "randomnet_ifun2_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv0.1_ihsigma0.1_lentrain10_vec0w_ax1_first_lyline_title1_continue_usemean_",
                       "randomnet_ifun2_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v50_amod0.1_alpha0.0001_fitlasso_1wv0.002_ihsigma0.1_lentrain10_vec0w_ax2_lyline_title1_continue_usemean_",
                       "randomnet_ifun2_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v100_amod0.1_alpha0.0001_fitlasso_1wv0.001_ihsigma0.1_lentrain10_vec0w_ax3_lyline_title1_continue_usemean_",


                       "randomnet_ifun2_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv0.1_ihsigma0.1_lentrain10_vec0w_specx2_ax4_noly_first_lyline_continue_usemean_",
                       "randomnet_ifun2_cnoise_grc_0N10000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv0.1_ihsigma0.1_lentrain10_vec0w_specx2_ax4_noly_first_lyline_dotted_continue_usemean_",

                       "randomnet_ifun2_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv0.1_1conv10_ihsigma0.1_lentrain10_vec0w_specx2_ax4_noly_first_lyline_colgrlight_title2_continue_usemean_",

                       "randomnet_ifun2_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv0.1_ihsigma0.1_lentrain10_vec0w_ax5_lyline_specx5_noly_continue_usemean_",
                       "randomnet_ifun2_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv0.1_0varw4_ihsigma0.1_lentrain10_vec0w_ax5_noly_specx6_dotted_continue_usemean_",
                       "randomnet_ifun2_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv0.1_1varw4_ihsigma0.1_lentrain10_vec0w_ax5_lyline_specx5_noly_colgrlight_title3_continue_usemean_",

                      ])

    data_dir = "./publish/closedloop/data/"
    #export_m = "./export"
    runs = 10


#FIGURE 7
if opt == "fig51l":
    do_vec = np.array([
                       "randomnet_ifun2_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv0.1_ihsigma0.1_lentrain10_vec0w_first_end_specx3_ax1_colgrlight_continue_usemean_",
                       "randomnet_ifun2_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv50_ihsigma0.1_lentrain10_vec0w_end_specx3_ax2_colgrlight_lyline_continue_usemean_",

                       "randomnet_ifun2_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv0.1_ihsigma0.1_2ndih1_lentrain10_vec0w_first_end_specx3_ax1_continue_usemean_",
                       "randomnet_ifun2_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv50_ihsigma0.1_2ndih1_lentrain10_vec0w_end_specx3_ax2_title5_continue_usemean_",

                      ])
    data_dir = "./publish/closedloop/data/"
    #export_m = "./export"
    runs = 10


#FIGURE 8
if opt == "fig52l":
    do_vec = np.array([
                       "randomnet_ifun2_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv0.1_ihsigma0.1_lentrain10_vec0w_first_end_ax3_colgrlight_lyline_onlyslow_continue_usemean_",
                       #"randomnet_ifun2_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_2tau1v50_amod0.1_alpha0.0001_fitlasso_1wv0.1_2wv0.002_2varw2_ihsigma0.1_lentrain10_vec0w_first_end_ax3_continue_usemean_",
                       #"randomnet_ifun2_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_2tau1v50_amod0.1_alpha0.0001_fitlasso_1wv0.1_2wv0.002_2varw0.1_ihsigma0.1_lentrain10_vec0w_first_end_ax3_dotted_continue_usemean_",
                       #"randomnet_ifun2b_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_2tau1v50_amod0.1_alpha0.0001_fitlasso_1wv0.1_2wv0.004_2probw0.5_ihsigma0.1_lentrain10_vec0w_first_end_ax3_lyline_continue_usemean_",
                       #"randomnet_ifun2b_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_2tau1v50_amod0.1_alpha0.0001_fitlasso_1wv0.1_2wv0.002_ihsigma0.1_lentrain10_vec0w_first_end_ax3_dotted_lyline_continue_usemean_",

                       "randomnet_ifun2b_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_2tau1v50_amod0.1_alpha0.0001_fitlasso_1wv0.1_2wv0.003_ihsigma0.1_lentrain10_vec0w_first_end_ax3_dotted_lyline_onlyslow_continue_usemean_",
                       "randomnet_ifun2b_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_2tau1v50_amod0.1_alpha0.0001_fitlasso_1wv0.1_2wv0.003_2varw0.1_ihsigma0.1_lentrain10_vec0w_first_end_ax3_dashed_lyline_onlyslow_continue_usemean_",
                       "randomnet_ifun2b_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_2tau1v50_amod0.1_alpha0.0001_fitlasso_1wv0.1_2wv0.003_2varw0.1_2probw0.5_ihsigma0.1_lentrain10_vec0w_first_end_ax3_lyline_onlyslow_continue_usemean_",


                       #"randomnet_ifun2_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv0.1_ihsigma0.1_lentrain10_vec0w_noinv_title6_end_ax4_colgrlight_continue_usemean_",
                       #"randomnet_ifun2_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_2tau1v50_amod0.1_alpha0.0001_fitlasso_1wv0.1_2wv0.002_2varw2_ihsigma0.1_lentrain10_vec0w_noinv_title6_end_ax4_continue_usemean_",
                       "randomnet_ifun2b_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv0.1_ihsigma0.1_lentrain10_vec0w_noinv_title6_end_ax4_colgrlight_lyline_onlyslow_continue_usemean_",
                       #"randomnet_ifun2b_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_2tau1v50_amod0.1_alpha0.0001_fitlasso_1wv0.1_2wv0.004_2probw0.5_ihsigma0.1_noinv_lentrain10_vec0w_end_ax4_lyline_continue_usemean_",
                       "randomnet_ifun2b_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_2tau1v50_amod0.1_alpha0.0001_fitlasso_1wv0.1_2wv0.003_2varw0.1_ihsigma0.1_noinv_lentrain10_vec0w_end_ax4_dashed_lyline_onlyslow_continue_usemean_",
                       "randomnet_ifun2b_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_2tau1v50_amod0.1_alpha0.0001_fitlasso_1wv0.1_2wv0.003_2probw0.5_2varw0.1_ihsigma0.1_noinv_lentrain10_vec0w_end_ax4_lyline_onlyslow_continue_usemean_",

                      ])
    data_dir = "./publish/closedloop/data/"
    #export_m = "./export"
    runs = 10


# FIGURE 9
# NOTE: The first line with "_runre1_" runs the model with teacher forcing only!
#       This is needed to get the proper weights for the Lyapunov estimation!
#       When running the model form scratch uncomment these lines in "figRe6"
 
if opt == "figRe6runtest":
    do_vec = np.array([
                       "randomnet_ifun2re_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv0.1_ihsigma0.1_lentrain10_refilt2_runre1_exprecx_norunly_vec0w_ax1_specx21_continue_usemean_",
                       "randomnet_ifun2re_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv0.1_ihsigma0.1_lentrain10_refilt2_runre2_lyline_vec0w_first_end_ax1_specx21_continue_usemean_",
                      ])
    #data_dir = "./publish/closedloop/data/"
    #export_m = "./export"
    runs = 10
    

if opt == "figRe6":
    do_vec = np.array([
                       "randomnet_ifun2_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv0.1_ihsigma0.1_lentrain10_vec0w_first_end_ax1_specx21_colgrlight_lyline_continue_usemean_",
                       #"randomnet_ifun2_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_l1r0.5_fitelastic_1wv0.1_ihsigma0.1_lentrain10_vec0w_first_end_ax1_specx21_colgrlight_dotted_lyline_continue_usemean_",

                       #"randomnet_ifun2re_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv0.1_ihsigma0.1_lentrain10_refilt2_runre1_exprecx_norunly_vec0w_ax1_specx21_continue_usemean_",
                       "randomnet_ifun2re_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv0.1_ihsigma0.1_lentrain10_refilt2_runre2_lyline_vec0w_first_end_ax1_specx21_continue_usemean_",

                       #"randomnet_ifun2re_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_noinv_fitlasso_1wv0.1_ihsigma0.1_lentrain10_refilt2_runre1_exprecx_norunly_vec0w_ax1_specx21_continue_usemean_",
                       #"randomnet_ifun2re_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_noinv_fitlasso_1wv0.1_ihsigma0.1_lentrain10_refilt2_runre2_vec0w_first_end_ax1_specx21_dotted_continue_usemean_",


                       "randomnet_ifun2_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_noinv_fitlasso_1wv0.1_ihsigma0.1_lentrain10_vec0w_first_ax2_specx22_colgrlight_continue_usemean_",

                       #"randomnet_ifun2re_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv0.1_ihsigma0.1_lentrain10_refilt2_runre1_exprecx_norunly_vec0w_ax2_specx22_continue_usemean_",
                       ##"randomnet_ifun2re_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_fitlasso_1wv0.1_ihsigma0.1_lentrain10_refilt2_runre2_vec0w_first_end_ax2_specx22_colgrlight_continue_usemean_", # _relyap, _prec1.0, _readdlyap_

                       #"randomnet_ifun2re_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_noinv_fitlasso_1wv0.1_ihsigma0.1_lentrain10_refilt2_runre1_exprecx_norunly_vec0w_ax2_specx22_continue_usemean_",
                       "randomnet_ifun2re_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_noinv_fitlasso_1wv0.1_ihsigma0.1_lentrain10_refilt2_runre2_vec0w_first_end_ax2_specx22_dots_continue_usemean_",

                       #"randomnet_ifun2re_cnoise_grc_0N1000_1N100_0tau1v50_1tau1v1_amod0.1_alpha0.0001_noinv_fitlassopos_1wv0.1_ihsigma0.1_lentrain10_refilt2_runre2_vec0w_first_end_ax3_specx21_lyline_continue_usemean_",

                      ])
    data_dir = "./publish/closedloop/data/"
    #export_m = "./export"
    runs = 10



# SET DEFAULT VALUES FOR THIS PLOT
fig_size =  [11.7, 8.3]
params = {  'backend': 'ps',
            'axes.labelsize': 9,
            'axes.linewidth' : 0.5,
            'title.fontsize': 8,
            'text.fontsize': 9,
            'legend.borderpad': 0.2,
            'legend.fontsize': 8,
            'legend.linewidth': 0.1,
            'legend.loc': 'best', # 'lower right'
            'legend.ncol': 4,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'text.usetex': False,
            'font.family': 'sans-serif',
            'font.serif': 'Arial',
            'pdf.fonttype' : 42,
            'ps.fonttype' : 42,
            'ps.useafm': True,
            'pdf.use14corefonts': True,
            'text.usetex': False,
            'figure.figsize': fig_size}
rcParams.update(params)


b1 = '#1F78B4' #377EB8
b2 = '#A6CEE3'
g1 = '#33A02C' #4DAF4A
g2 = '#B2DF8A'
r1 = '#E31A1C' #E41A1C
r2 = '#FB9A99'
o1 = '#FF7F00' #FF7F00
o2 = '#FDBF6F'
p1 = '#6A3D9A' #984EA3
p2 = '#CAB2D6'

ye1 = '#FFFF33'
br1 = '#A65628'
pi1 = '#F781BF'
gr1 = '#999999'
k1 = '#000000'

color_v = []

if myid == 0:

    if "fig1" in opt:

        fig_size =  [4.86,6.00] # 1.5-Column 6.83

        params = {'backend': 'ps',
          'axes.labelsize': 8,
          'axes.linewidth' : 0.5,
          'title.fontsize': 8,
          'text.fontsize': 10,
          'font.size':10,
          'axes.titlesize':8,
          'legend.fontsize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'legend.borderpad': 0.2,
          'legend.linewidth': 0.1,
          'legend.loc': 'best',
          'legend.ncol': 4,
          'figure.figsize': fig_size}
        rcParams.update(params)

        fig1 = plt.figure('vec')

        gs1 = matplotlib.gridspec.GridSpec(2, 3,
           width_ratios=[1,1,1],
           height_ratios=[1,1]
           )

        gs1.update(bottom=0.07, top=0.45, left=0.09, right=0.97, wspace=0.2, hspace=0.3)

        ax1 = plt.subplot(gs1[0,0])
        ax2 = plt.subplot(gs1[0,1])
        ax3 = plt.subplot(gs1[0,2])

        ax4 = plt.subplot(gs1[1,0])
        ax5 = plt.subplot(gs1[1,1])
        ax6 = plt.subplot(gs1[1,2])

        gs2 = matplotlib.gridspec.GridSpec(1, 3,
           width_ratios=[1,1,1],
           height_ratios=[1]
           )

        gs2.update(bottom=0.55, top=0.76, left=0.11, right=0.97, wspace=0.35, hspace=0.3)

        ax7 = plt.subplot(gs2[0,0])
        ax8 = plt.subplot(gs2[0,1])
        ax9 = plt.subplot(gs2[0,2])

        font = font0.copy()
        font.set_family('sans-serif')
        font.set_weight('bold')


        x1 = -0.12
        y1 = 1.2
        ax1.text(x1, y1, 'D1', transform=ax1.transAxes, fontsize=12, va='top', fontproperties=font)
        ax2.text(x1, y1, 'D2', transform=ax2.transAxes, fontsize=12, va='top', fontproperties=font)
        ax3.text(x1, y1, 'D3', transform=ax3.transAxes, fontsize=12, va='top', fontproperties=font)

        ax4.text(x1, y1, 'E1', transform=ax4.transAxes, fontsize=12, va='top', fontproperties=font)
        ax5.text(x1, y1, 'E2', transform=ax5.transAxes, fontsize=12, va='top', fontproperties=font)
        ax6.text(x1, y1, 'E3', transform=ax6.transAxes, fontsize=12, va='top', fontproperties=font)

        x1 = -0.12
        y1 = 1.15
        ax7.text(x1, y1, 'A', transform=ax7.transAxes, fontsize=12, va='top', fontproperties=font)
        ax8.text(x1, y1, 'B', transform=ax8.transAxes, fontsize=12, va='top', fontproperties=font)
        ax9.text(x1, y1, 'C', transform=ax9.transAxes, fontsize=12, va='top', fontproperties=font)

        d_out = 10
        d_down = 5
        linewidth = 1.5



    elif "fig2" in opt:

        fig_size =  [4.86,4] # 1.5-Column 6.83

        params = {'backend': 'ps',
          'axes.labelsize': 8,
          'axes.linewidth' : 0.5,
          'title.fontsize': 8,
          'text.fontsize': 10,
          'font.size':10,
          'axes.titlesize':8,
          'legend.fontsize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'legend.borderpad': 0.2,
          'legend.linewidth': 0.1,
          'legend.loc': 'best',
          'legend.ncol': 4,
          'figure.figsize': fig_size}
        rcParams.update(params)

        fig1 = plt.figure('vec')

        gs1 = matplotlib.gridspec.GridSpec(2, 3,
           width_ratios=[1,1,1],
           height_ratios=[0.75,1]
           )

        gs1.update(bottom=0.09, top=0.93, left=0.12, right=0.97, wspace=0.2, hspace=0.2)

        ax1 = plt.subplot(gs1[1,0])
        ax2 = plt.subplot(gs1[1,1])
        ax3 = plt.subplot(gs1[1,2])

        ax1b = plt.subplot(gs1[0,0])
        ax2b = plt.subplot(gs1[0,1])
        ax3b = plt.subplot(gs1[0,2])

        font = font0.copy()
        font.set_family('sans-serif')
        font.set_weight('bold')


        x1 = -0.15
        y1 = 1.14
        ax1.text(x1, y1-0.02, 'A2', transform=ax1.transAxes, fontsize=12, va='top', fontproperties=font)
        ax2.text(x1, y1-0.02, 'B2', transform=ax2.transAxes, fontsize=12, va='top', fontproperties=font)
        ax3.text(x1, y1-0.02, 'C2', transform=ax3.transAxes, fontsize=12, va='top', fontproperties=font)

        ax1b.text(x1, y1, 'A1', transform=ax1b.transAxes, fontsize=12, va='top', fontproperties=font)
        ax2b.text(x1, y1, 'B1', transform=ax2b.transAxes, fontsize=12, va='top', fontproperties=font)
        ax3b.text(x1, y1, 'C1', transform=ax3b.transAxes, fontsize=12, va='top', fontproperties=font)

        #ax4.text(x1, y1, 'D1', transform=ax4.transAxes, fontsize=12, va='top', fontproperties=font)
        #ax5.text(x1, y1, 'D2', transform=ax5.transAxes, fontsize=12, va='top', fontproperties=font)
        #ax6.text(x1, y1, 'D3', transform=ax6.transAxes, fontsize=12, va='top', fontproperties=font)

        #ax7.text(x1, y1, 'E1', transform=ax7.transAxes, fontsize=12, va='top', fontproperties=font)
        #ax8.text(x1, y1, 'E2', transform=ax8.transAxes, fontsize=12, va='top', fontproperties=font)
        #ax9.text(x1, y1, 'E3', transform=ax9.transAxes, fontsize=12, va='top', fontproperties=font)

        d_out = 10
        d_down = 5
        linewidth = 1.5

    elif "fig3b" in opt:

        fig_size =  [8.3*0.3937,6] # 1.5-Column 6.83

        params = {'backend': 'ps',
          'axes.labelsize': 8,
          'axes.linewidth' : 0.5,
          'title.fontsize': 8,
          'text.fontsize': 10,
          'font.size':10,
          'axes.titlesize':8,
          'legend.fontsize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'legend.borderpad': 0.2,
          'legend.linewidth': 0.1,
          'legend.loc': 'best',
          'legend.ncol': 4,
          'figure.figsize': fig_size}
        rcParams.update(params)

        fig1 = plt.figure('vec')

        gs0 = matplotlib.gridspec.GridSpec(1, 1,
           width_ratios=[1],
           height_ratios=[1]
           )

        gs0.update(bottom=0.75, top=0.94, left=0.17, right=0.95, wspace=0.2, hspace=0.4)

        ax1 = plt.subplot(gs0[0,0])

        gs1 = matplotlib.gridspec.GridSpec(3, 1,
           width_ratios=[1],
           height_ratios=[1,1,0.75]
           )

        gs1.update(bottom=0.07, top=0.65, left=0.17, right=0.95, wspace=0.2, hspace=0.4)

        ax2 = plt.subplot(gs1[0,0])
        ax3 = plt.subplot(gs1[1,0])
        ax3b = plt.subplot(gs1[2,0])

        font = font0.copy()
        font.set_family('sans-serif')
        font.set_weight('bold')

        x1 = -0.03
        y1 = 1.25
        ax1.text(x1, y1, 'A', transform=ax1.transAxes, fontsize=12, va='top', fontproperties=font)
        ax2.text(x1, y1, 'B1', transform=ax2.transAxes, fontsize=12, va='top', fontproperties=font)
        ax3.text(x1, y1, 'B2', transform=ax3.transAxes, fontsize=12, va='top', fontproperties=font)
        ax3b.text(x1, y1, 'B3', transform=ax3b.transAxes, fontsize=12, va='top', fontproperties=font)

        d_out = 10
        d_down = 5
        linewidth = 1.5


    elif "fig3" in opt:

        fig_size =  [17.35*0.3937,3.5] # 1.5-Column 6.83

        params = {'backend': 'ps',
          'axes.labelsize': 8,
          'axes.linewidth' : 0.5,
          'title.fontsize': 8,
          'text.fontsize': 10,
          'font.size':10,
          'axes.titlesize':8,
          'legend.fontsize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'legend.borderpad': 0.2,
          'legend.linewidth': 0.1,
          'legend.loc': 'best',
          'legend.ncol': 4,
          'figure.figsize': fig_size}
        rcParams.update(params)

        fig1 = plt.figure('vec')

        gs1 = matplotlib.gridspec.GridSpec(2, 3,
           width_ratios=[1,1,1],
           height_ratios=[1,0.75]
           )

        gs1.update(bottom=0.11, top=0.90, left=0.08, right=0.97, wspace=0.2, hspace=0.2)

        ax1 = plt.subplot(gs1[0,0])
        ax2 = plt.subplot(gs1[0,1])
        ax1b = plt.subplot(gs1[1,0])
        ax2b = plt.subplot(gs1[1,1])
        ax3 = plt.subplot(gs1[0,2])
        ax3b = plt.subplot(gs1[1,2])

        font = font0.copy()
        font.set_family('sans-serif')
        font.set_weight('bold')

        x1 = -0.13
        y1 = 1.18
        ax1.text(x1, y1, 'A1', transform=ax1.transAxes, fontsize=12, va='top', fontproperties=font)
        ax2.text(x1, y1, 'B1', transform=ax2.transAxes, fontsize=12, va='top', fontproperties=font)
        ax1b.text(x1, y1, 'A2', transform=ax1b.transAxes, fontsize=12, va='top', fontproperties=font)
        ax2b.text(x1, y1, 'B2', transform=ax2b.transAxes, fontsize=12, va='top', fontproperties=font)
        ax3.text(x1, y1, 'C1', transform=ax3.transAxes, fontsize=12, va='top', fontproperties=font)
        ax3b.text(x1, y1, 'C2', transform=ax3b.transAxes, fontsize=12, va='top', fontproperties=font)

        d_out = 10
        d_down = 5
        linewidth = 1.5

    elif "fig3x" in opt:

        fig_size =  [4.86,6] # 1.5-Column 6.83

        params = {'backend': 'ps',
          'axes.labelsize': 8,
          'axes.linewidth' : 0.5,
          'title.fontsize': 8,
          'text.fontsize': 10,
          'font.size':10,
          'axes.titlesize':8,
          'legend.fontsize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'legend.borderpad': 0.2,
          'legend.linewidth': 0.1,
          'legend.loc': 'best',
          'legend.ncol': 4,
          'figure.figsize': fig_size}
        rcParams.update(params)

        fig1 = plt.figure('vec')

        gs1 = matplotlib.gridspec.GridSpec(2, 2,
           width_ratios=[1,1],
           height_ratios=[1,0.75]
           )

        gs1.update(bottom=0.55, top=0.95, left=0.12, right=0.97, wspace=0.2, hspace=0.3)

        ax1 = plt.subplot(gs1[0,0])
        ax2 = plt.subplot(gs1[0,1])
        ax1b = plt.subplot(gs1[1,0])
        ax2b = plt.subplot(gs1[1,1])


        gs2 = matplotlib.gridspec.GridSpec(2, 2,
           width_ratios=[1,1],
           height_ratios=[1,0.75]
           )

        gs2.update(bottom=0.06, top=0.45, left=0.12, right=0.97, wspace=0.2, hspace=0.3)

        ax3 = plt.subplot(gs2[0,0])
        ax4 = plt.subplot(gs2[0,1])
        ax3b = plt.subplot(gs2[1,0])
        ax4b = plt.subplot(gs2[1,1])


        font = font0.copy()
        font.set_family('sans-serif')
        font.set_weight('bold')

        x1 = -0.12
        y1 = 1.18
        ax1.text(x1, y1, 'A1', transform=ax1.transAxes, fontsize=12, va='top', fontproperties=font)
        ax2.text(x1, y1, 'B1', transform=ax2.transAxes, fontsize=12, va='top', fontproperties=font)
        ax1b.text(x1, y1, 'A2', transform=ax1b.transAxes, fontsize=12, va='top', fontproperties=font)
        ax2b.text(x1, y1, 'B2', transform=ax2b.transAxes, fontsize=12, va='top', fontproperties=font)

        ax3.text(x1, y1, 'C1', transform=ax3.transAxes, fontsize=12, va='top', fontproperties=font)
        ax4.text(x1, y1, 'D1', transform=ax4.transAxes, fontsize=12, va='top', fontproperties=font)
        ax3b.text(x1, y1, 'C2', transform=ax3b.transAxes, fontsize=12, va='top', fontproperties=font)
        ax4b.text(x1, y1, 'D2', transform=ax4b.transAxes, fontsize=12, va='top', fontproperties=font)

        d_out = 10
        d_down = 5
        linewidth = 1.5


    elif "fig51" in opt:

        fig_size =  [4.86,3.75] # 1.5-Column 6.83

        params = {'backend': 'ps',
          'axes.labelsize': 8,
          'axes.linewidth' : 0.5,
          'title.fontsize': 8,
          'text.fontsize': 10,
          'font.size':10,
          'axes.titlesize':8,
          'legend.fontsize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'legend.borderpad': 0.2,
          'legend.linewidth': 0.1,
          'legend.loc': 'best',
          'legend.ncol': 4,
          'figure.figsize': fig_size}
        rcParams.update(params)

        fig1 = plt.figure('vec')

        gs1 = matplotlib.gridspec.GridSpec(2, 2,
           width_ratios=[1,1],
           height_ratios=[1,0.75]
           )

        gs1.update(bottom=0.10, top=0.63, left=0.12, right=0.97, wspace=0.15, hspace=0.25)

        ax1 = plt.subplot(gs1[0,0])
        ax2 = plt.subplot(gs1[0,1])

        ax1b = plt.subplot(gs1[1,0])
        ax2b = plt.subplot(gs1[1,1])


        font = font0.copy()
        font.set_family('sans-serif')
        font.set_weight('bold')

        x1 = -0.09
        y1 = 1.17
        ax1.text(x1, y1, 'A1', transform=ax1.transAxes, fontsize=12, va='top', fontproperties=font)
        ax2.text(x1, y1, 'B1', transform=ax2.transAxes, fontsize=12, va='top', fontproperties=font)

        ax1b.text(x1, y1+0.04, 'A2', transform=ax1b.transAxes, fontsize=12, va='top', fontproperties=font)
        ax2b.text(x1, y1+0.04, 'B2', transform=ax2b.transAxes, fontsize=12, va='top', fontproperties=font)


        d_out = 10
        d_down = 5
        linewidth = 1.5


    elif "fig52" in opt:

        fig_size =  [4.86,3.75] # 1.5-Column 6.83

        params = {'backend': 'ps',
          'axes.labelsize': 8,
          'axes.linewidth' : 0.5,
          'title.fontsize': 8,
          'text.fontsize': 10,
          'font.size':10,
          'axes.titlesize':8,
          'legend.fontsize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'legend.borderpad': 0.2,
          'legend.linewidth': 0.1,
          'legend.loc': 'best',
          'legend.ncol': 4,
          'figure.figsize': fig_size}
        rcParams.update(params)

        fig1 = plt.figure('vec')

        gs2 = matplotlib.gridspec.GridSpec(2,2 ,
           width_ratios=[1,1],
           height_ratios=[1,0.75]
           )

        gs2.update(bottom=0.10, top=0.63, left=0.12, right=0.97, wspace=0.15, hspace=0.25)

        ax3 = plt.subplot(gs2[0,0])
        ax4 = plt.subplot(gs2[0,1])

        ax3b = plt.subplot(gs2[1,0])
        ax4b = plt.subplot(gs2[1,1])

        font = font0.copy()
        font.set_family('sans-serif')
        font.set_weight('bold')

        x1 = -0.09
        y1 = 1.16

        ax3.text(x1, y1, 'A1', transform=ax3.transAxes, fontsize=12, va='top', fontproperties=font)
        ax3b.text(x1, y1+0.04, 'A2', transform=ax3b.transAxes, fontsize=12, va='top', fontproperties=font)

        ax4.text(x1, y1, 'B1', transform=ax4.transAxes, fontsize=12, va='top', fontproperties=font)
        ax4b.text(x1, y1+0.04, 'B2', transform=ax4b.transAxes, fontsize=12, va='top', fontproperties=font)


        d_out = 10
        d_down = 5
        linewidth = 1.5


    elif "figRe6" in opt:

        fig_size =  [4.86,3.75] # 1.5-Column 6.83

        params = {'backend': 'ps',
          'axes.labelsize': 8,
          'axes.linewidth' : 0.5,
          'title.fontsize': 8,
          'text.fontsize': 10,
          'font.size':10,
          'axes.titlesize':8,
          'legend.fontsize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'legend.borderpad': 0.2,
          'legend.linewidth': 0.1,
          'legend.loc': 'best',
          'legend.ncol': 4,
          'figure.figsize': fig_size}
        rcParams.update(params)

        fig1 = plt.figure('vec')

        gs1 = matplotlib.gridspec.GridSpec(2, 2,
           width_ratios=[1,1],
           height_ratios=[1,0.75]
           )

        gs1.update(bottom=0.10, top=0.63, left=0.12, right=0.97, wspace=0.25, hspace=0.25)

        ax1 = plt.subplot(gs1[0,0])
        ax2 = plt.subplot(gs1[0,1])

        ax1b = plt.subplot(gs1[1,0])
        ax2b = plt.subplot(gs1[1,1])


        font = font0.copy()
        font.set_family('sans-serif')
        font.set_weight('bold')

        x1 = -0.09
        y1 = 1.17
        ax1.text(x1, y1, 'A1', transform=ax1.transAxes, fontsize=12, va='top', fontproperties=font)
        ax2.text(x1, y1, 'B1', transform=ax2.transAxes, fontsize=12, va='top', fontproperties=font)

        ax1b.text(x1, y1+0.04, 'A2', transform=ax1b.transAxes, fontsize=12, va='top', fontproperties=font)
        ax2b.text(x1, y1+0.04, 'B2', transform=ax2b.transAxes, fontsize=12, va='top', fontproperties=font)


        d_out = 10
        d_down = 5
        linewidth = 1.5


    elif "fig5" in opt:

        fig_size =  [4.86,7.5] # 1.5-Column 6.83

        params = {'backend': 'ps',
          'axes.labelsize': 8,
          'axes.linewidth' : 0.5,
          'title.fontsize': 8,
          'text.fontsize': 10,
          'font.size':10,
          'axes.titlesize':8,
          'legend.fontsize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'legend.borderpad': 0.2,
          'legend.linewidth': 0.1,
          'legend.loc': 'best',
          'legend.ncol': 4,
          'figure.figsize': fig_size}
        rcParams.update(params)

        fig1 = plt.figure('vec')

        gs1 = matplotlib.gridspec.GridSpec(2, 2,
           width_ratios=[1,1],
           height_ratios=[1,0.75]
           )

        gs1.update(bottom=0.55, top=0.82, left=0.12, right=0.97, wspace=0.15, hspace=0.25)

        ax1 = plt.subplot(gs1[0,0])
        ax2 = plt.subplot(gs1[0,1])

        ax1b = plt.subplot(gs1[1,0])
        ax2b = plt.subplot(gs1[1,1])

        gs2 = matplotlib.gridspec.GridSpec(2,2 ,
           width_ratios=[1,1],
           height_ratios=[1,0.75]
           )

        gs2.update(bottom=0.05, top=0.32, left=0.12, right=0.97, wspace=0.15, hspace=0.25)

        ax3 = plt.subplot(gs2[0,0])
        ax4 = plt.subplot(gs2[0,1])

        ax3b = plt.subplot(gs2[1,0])
        ax4b = plt.subplot(gs2[1,1])

        font = font0.copy()
        font.set_family('sans-serif')
        font.set_weight('bold')

        x1 = -0.09
        y1 = 1.16
        ax1.text(x1, y1, 'A1', transform=ax1.transAxes, fontsize=12, va='top', fontproperties=font)
        ax2.text(x1, y1, 'B1', transform=ax2.transAxes, fontsize=12, va='top', fontproperties=font)
        ax3.text(x1, y1, 'C1', transform=ax3.transAxes, fontsize=12, va='top', fontproperties=font)

        ax1b.text(x1, y1+0.04, 'A2', transform=ax1b.transAxes, fontsize=12, va='top', fontproperties=font)
        ax2b.text(x1, y1+0.04, 'B2', transform=ax2b.transAxes, fontsize=12, va='top', fontproperties=font)
        ax3b.text(x1, y1+0.04, 'C2', transform=ax3b.transAxes, fontsize=12, va='top', fontproperties=font)

        ax4.text(x1, y1, 'D1', transform=ax4.transAxes, fontsize=12, va='top', fontproperties=font)
        ax4b.text(x1, y1+0.04, 'D2', transform=ax4b.transAxes, fontsize=12, va='top', fontproperties=font)


        d_out = 10
        d_down = 5
        linewidth = 1.5


    elif "fig4" in opt:

        fig_size =  [4.86,6] # 1.5-Column 6.83

        params = {'backend': 'ps',
          'axes.labelsize': 8,
          'axes.linewidth' : 0.5,
          'title.fontsize': 8,
          'text.fontsize': 10,
          'font.size':10,
          'axes.titlesize':8,
          'legend.fontsize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'legend.borderpad': 0.2,
          'legend.linewidth': 0.1,
          'legend.loc': 'best',
          'legend.ncol': 4,
          'figure.figsize': fig_size}
        rcParams.update(params)

        fig1 = plt.figure('vec')

        gs1 = matplotlib.gridspec.GridSpec(2, 3,
           width_ratios=[1,1,1],
           height_ratios=[1,1]
           )

        gs1.update(bottom=0.32, top=0.75, left=0.12, right=0.97, wspace=0.2, hspace=0.2)

        ax1 = plt.subplot(gs1[0,0])
        ax2 = plt.subplot(gs1[0,1])
        ax3 = plt.subplot(gs1[0,2])

        ax1b = plt.subplot(gs1[1,0])
        ax2b = plt.subplot(gs1[1,1])
        ax3b = plt.subplot(gs1[1,2])

        gs2 = matplotlib.gridspec.GridSpec(1, 2,
           width_ratios=[1,1],
           height_ratios=[1]
           )

        gs2.update(bottom=0.06, top=0.23, left=0.12, right=0.97, wspace=0.15, hspace=0.6)

        ax4 = plt.subplot(gs2[0,0])
        ax5 = plt.subplot(gs2[0,1])
        #ax5b = plt.subplot(gs2[0,2])

        font = font0.copy()
        font.set_family('sans-serif')
        font.set_weight('bold')

        x1 = -0.12
        y1 = 1.15
        ax1.text(x1, y1, 'A1', transform=ax1.transAxes, fontsize=12, va='top', fontproperties=font)
        ax2.text(x1, y1, 'B1', transform=ax2.transAxes, fontsize=12, va='top', fontproperties=font)
        ax3.text(x1, y1, 'C1', transform=ax3.transAxes, fontsize=12, va='top', fontproperties=font)

        ax1b.text(x1, y1, 'A2', transform=ax1b.transAxes, fontsize=12, va='top', fontproperties=font)
        ax2b.text(x1, y1, 'B2', transform=ax2b.transAxes, fontsize=12, va='top', fontproperties=font)
        ax3b.text(x1, y1, 'C2', transform=ax3b.transAxes, fontsize=12, va='top', fontproperties=font)

        x1 = -0.10
        y1 = 1.18
        ax4.text(x1, y1, 'D', transform=ax4.transAxes, fontsize=12, va='top', fontproperties=font)
        ax5.text(x1, y1, 'E', transform=ax5.transAxes, fontsize=12, va='top', fontproperties=font)
        #ax5b.text(x1, y1, 'E2', transform=ax5b.transAxes, fontsize=12, va='top', fontproperties=font)

        d_out = 10
        d_down = 5
        linewidth = 1.5

    else:

        fig1 = plt.figure('vec')

        gs = matplotlib.gridspec.GridSpec(2, 4,
           width_ratios=[1,1,1,1],
           height_ratios=[1,1]
           )

        #gs.update(bottom=0.08, top=0.90, left=0.08, right=0.97, wspace=0.4, hspace=0.6)

        ax1 = plt.subplot(gs[0,0])
        ax2 = plt.subplot(gs[0,1])
        ax3 = plt.subplot(gs[0,2])
        ax4 = plt.subplot(gs[0,3])

        ax1b = plt.subplot(gs[1,0])
        ax2b = plt.subplot(gs[1,1])
        ax3b = plt.subplot(gs[1,2])
        ax4b = plt.subplot(gs[1,3])

        font = font0.copy()
        font.set_family('sans-serif')
        font.set_weight('bold')

        d_out = 10
        d_down = 5
        linewidth = 1.5



for d, do in enumerate(do_vec):

    # Isope and Barbour 2002: (also: 1 ms and 11 ms at 32°C)
    Q10_channel = 2.4
    Q10 = Q10_channel**((37-32)/10.)
    tau1_pj = 1.2*ms/Q10
    tau2_pj = 13.9*ms/Q10

    bin_width = 0.1*ms

    temperature = 37

    np.random.seed(444)

    delta = 2

    #set to True for artefact test:
    nulltest = False

    fluct_s = [0,0]
    fluct_tau = 10*ms
    factor_celltype = [1,1]
    factor_recurrent = [[0,0],[0,0]]
    stdp = []

    tdur = 0.05
    istart = 0; istop = 0.2; di = 0.01

    dumpsave = 1

    give_freq = True
    amod = [1,1]

    tau1_ex=[]
    tau2_ex=[]
    n_syn_ex = []
    syn_max_mf = []
    g_syn_ex_s = []
    g_syn_ex = []
    mglufac_ex = []
    anoise = []
    noise_a = []
    noise_out = 0
    noise_syn = []
    noise_syn_tau = []
    syn_ex_dist = []
    tau1_basis = np.array([0,0,0])
    tau2_basis = np.array([0.01,0.1,0.5])
    max_weight = 1e9

    len_train = 3
    len_test = 3

    seed0 = 1

    fit_restart = True
    fit_restart_mean = True
    restart_lyap = False

    t_analysis_delay = 0
    t_analysis_stop = 0


    istep = []
    istep_sigma = []
    itest = []
    itest_sigma = []

    tstep = []
    ttest = []
    tstop = []

    t_plot_delay = []
    t_plot_stop = []

    N = [0]

    lymax = False
    lymin = False
    downsample = 1
    max_freq = 1e10
    ridge_alpha = 1

    prefix = "randomnet"

    if "_continue_" in do:
        fit_restart = False

    if "_usemean_" in do:
        fit_restart_mean = False

    readd_lyap = False
    if "_readdlyap_" in do:
        readd_lyap = True

    if "_ifun_" in do:

        dt = 1*ms
        cellimport = ["import cells.ifun._ifun as ifun"]
        celltype = ["ifun"]
        cell_exe = [""]

        conntype=[{'type':'e2inh', 'src':0, 'tgt':0, 'w':0.1, 'var':0, 'tau1':100*ms, 'tau2':0*ms, 'conv':0.5}
                 ]

        N = [500]

        amod = [0.01]
        ihold = [1]
        ihold_sigma = [0]

        factor_celltype = [[1,0,0.5]] #[:][0] mean height of modulation, [:][1] variance of modulation, [:][2] probability that modulation is inverse (NOW if >0, 50% inverse input!)

        anoise = [0]

        len_train = 100
        len_test = 10

        lymax=17

        if "_0tau1v10_" in do:
            xmin = 0; xmax=40; vec0w = np.arange(0,40.2,0.2)
        if "_0tau1v50_" in do:
            xmin = 0; xmax=4; vec0w = np.arange(0,4.02,0.02)
        if "_0tau1v100_" in do:

            if "_dt" in do:
                xmin = 0; xmax=1; vec0w = np.arange(0,1.02,0.02)
            else:
                xmin = 0; xmax=4; vec0w = np.arange(0,4.02,0.02)

        if "_0conv0.04_" in do:
            if "_0tau1v10_" in do:
                xmin = 0; xmax=40; vec0w = np.arange(0,40.2,0.2)
            if "_0tau1v50_" in do:
                xmin = 0; xmax=4; vec0w = np.arange(0,4.02,0.02)
            if "_0tau1v100_" in do:
                xmin = 0; xmax=4; vec0w = np.arange(0,4.02,0.02)

        if "_specx1_" in do:
            if "_0tau1v10_" in do:
                xmin = 0; xmax=80; vec0w = np.arange(0,80,1)
            if "_0tau1v50_" in do:
                xmin = 0; xmax=14; vec0w = np.arange(0,14.1,0.1)
            if "_0tau1v100_" in do:
                xmin = 0; xmax=14; vec0w = np.arange(0,14.1,0.1)

            if ("_0N10000_" in do):
                if "_0tau1v10_" in do:
                    xmin = 0; xmax=80; vec0w = np.arange(0,80,2)
                if "_0tau1v50_" in do:
                    xmin = 0; xmax=14; vec0w = np.arange(0,14.5,0.5)
                if "_0tau1v100_" in do:
                    xmin = 0; xmax=14; vec0w = np.arange(0,14.5,0.5)

        if ("_normp_" in do):
            xmin = 0; xmax = 1
            if "_0conv0.04_" in do:
                vec0w = np.arange(0,25.2,0.2)
                if ("_0N10000_" in do):
                    vec0w = np.arange(0,25.2,0.2)
            elif "_0conv0.4_" in do:
                vec0w = np.arange(0,2.62,0.02)



        if "_specy1_" in do:
            lymax = 4.2

        noise_out = 0

        prefix = prefix + "_ifun"

        use_c = True
        use_mpi = False



    if "_ifun2_" in do:

        dt = 1*ms
        cellimport = ["import cells.ifun._ifun2 as ifun2"]
        celltype = ["ifun2"]
        cell_exe = [""]

        conntype=[{'type':'e2inh', 'src':1, 'tgt':0, 'w':0.1, 'var':0, 'tau1':100*ms, 'tau2':0*ms, 'conv':4}, #0.0005
                  {'type':'e2ex', 'src':0, 'tgt':1, 'w':1, 'var':0, 'tau1':1*ms, 'tau2':0*ms, 'conv':100},
                  {'type':'e2m', 'src':0, 'tgt':1, 'w':0, 'var':0, 'tau1':50*ms, 'tau2':0*ms, 'conv':0}
                 ]

        N = [500,50]

        amod = [0.01,0]
        ihold = [1,0]
        ihold_sigma = [0,0]

        factor_celltype = [[1,0,0.5],[1,0,0.5]] #[:][0] mean height of modulation, [:][1] variance of modulation, [:][2] probability that modulation is inverse (NOW if >0, 50% inverse input!)

        anoise = [0,0]

        len_train = 100
        len_test = 10

        lymax = 7

        xmin = 0; xmax=4 ; vec0w = np.arange(0,4.02,0.02)

        if "_0tau1v100_1tau1v1_" in do:
            xmin = 0; xmax=4; vec0w = np.arange(0,4.02,0.02)
        if "_0tau1v50_1tau1v50_" in do:
            xmin = 0; xmax=4; vec0w = np.arange(0,4.02,0.02)
        if "_0tau1v100_1tau1v100_" in do:
            xmin = 0; xmax=4; vec0w = np.arange(0,4.02,0.02)
        if "_0tau1v100_1tau1v50_" in do:
            xmin = 0; xmax=4; vec0w = np.arange(0,4.02,0.02)
        if "_0tau1v100_1tau1v10_" in do:
            xmin = 0; xmax=4; vec0w = np.arange(0,4.02,0.02)

        if "_specx7_" in do:
            xmin = 0; xmax=4 ; vec0w = np.arange(0,4.02,0.02)

        if "_specx2_" in do:
            xmin = 0; xmax=2; vec0w = np.arange(0,2.02,0.02)

        if "_specx21_" in do:
            xmin = 0; xmax=2; vec0w = np.arange(0,2.01,0.01)

        if "_specx22_" in do:
            xmin = 0; xmax=0.01; vec0w = np.arange(0,0.01002,0.00002)

        if "_specx3_" in do:
            xmin = 0; xmax=0.02; vec0w = np.arange(0,0.0202,0.0002)

        if "_specx5_" in do:
            xmin = 0; xmax=6 ; vec0w = np.arange(0,6.02,0.02)

        if "_specx6_" in do:
            xmin = 0; xmax=6 ; vec0w = np.arange(0,6.1,0.1)

        if ("_specx4_" in do):
            xmin = 0.014; xmax=0.016; vec0w = np.arange(0.014,0.016,0.00001)

            lymax = None
            lymin = None

        if "_specx8_" in do:
            xmin = 0; xmax=0.1 ; vec0w = np.arange(0,0.101,0.001)
            lymax = 7


        noise_out = 0

        prefix = prefix + "_ifun2"

        use_c = True
        use_mpi = False


    delay_recurrent = False
    teacher_forcing = False
    if "_ifun2re_" in do:

        dt = 1*ms
        cellimport = ["import cells.ifun._ifun2re as ifun2re"]
        celltype = ["ifun2re"]
        cell_exe = [""]

        conntype=[{'type':'e2inh', 'src':1, 'tgt':0, 'w':0.1, 'var':0, 'tau1':100*ms, 'tau2':0*ms, 'conv':4}, #0.0005
                  {'type':'e2ex', 'src':0, 'tgt':1, 'w':1, 'var':0, 'tau1':1*ms, 'tau2':0*ms, 'conv':100},
                  {'type':'e2m', 'src':0, 'tgt':1, 'w':0, 'var':0, 'tau1':50*ms, 'tau2':0*ms, 'conv':0}
                 ]

        N = [500,50]

        amod = [0.01,0]
        ihold = [1,0]
        ihold_sigma = [0,0]

        factor_celltype = [[1,0,0.5],[1,0,0.5]] #[:][0] mean height of modulation, [:][1] variance of modulation, [:][2] probability that modulation is inverse (NOW if >0, 50% inverse input!)
        factor_recurrent = [[0.2,0.5,1e-4,0.1],[0.2,0.5,1e-4,0.1]]
        delay_recurrent = 1e-3
        teacher_forcing = True

        anoise = [0,0]

        len_train = 100
        len_test = 10

        lymax = 7

        xmin = 0; xmax=4 ; vec0w = np.arange(0,4.02,0.02)

        if "_0tau1v100_1tau1v1_" in do:
            xmin = 0; xmax=4; vec0w = np.arange(0,4.02,0.02)
        if "_0tau1v50_1tau1v50_" in do:
            xmin = 0; xmax=4; vec0w = np.arange(0,4.02,0.02)
        if "_0tau1v100_1tau1v100_" in do:
            xmin = 0; xmax=4; vec0w = np.arange(0,4.02,0.02)
        if "_0tau1v100_1tau1v50_" in do:
            xmin = 0; xmax=4; vec0w = np.arange(0,4.02,0.02)
        if "_0tau1v100_1tau1v10_" in do:
            xmin = 0; xmax=4; vec0w = np.arange(0,4.02,0.02)

        if "_specx7_" in do:
            xmin = 0; xmax=4 ; vec0w = np.arange(0,4.02,0.02)

        if "_specx2_" in do:
            xmin = 0; xmax=2; vec0w = np.arange(0,2.02,0.02)

        if "_specx21_" in do:
            xmin = 0; xmax=2; vec0w = np.arange(0,2.01,0.01)
            lymax = 3

        if "_specx22_" in do:
            xmin = 0; xmax=0.01; vec0w = np.arange(0,0.01002,0.00002)
            lymax = 0.5
            lymin = -3

        if "_specx3_" in do:
            xmin = 0; xmax=0.02; vec0w = np.arange(0,0.0202,0.0002)

        if "_specx5_" in do:
            xmin = 0; xmax=6 ; vec0w = np.arange(0,6.02,0.02)

        if "_specx6_" in do:
            xmin = 0; xmax=6 ; vec0w = np.arange(0,6.1,0.1)

        if ("_specx4_" in do):
            xmin = 0.014; xmax=0.016; vec0w = np.arange(0.014,0.016,0.00001)

            lymax = None
            lymin = None

        if "_specx8_" in do:
            xmin = 0; xmax=0.1 ; vec0w = np.arange(0,0.101,0.001)
            lymax = 7


        noise_out = 0

        prefix = prefix + "_ifun2re"

        use_c = True
        use_mpi = False


    if "_ifun2b_" in do:

        dt = 1*ms
        cellimport = ["import cells.ifun._ifun2b as ifun2b"]
        celltype = ["ifun2b"]
        cell_exe = [""]

        conntype=[{'type':'e2inh', 'src':1, 'tgt':0, 'w':0.1, 'var':0, 'tau1':100*ms, 'tau2':0*ms, 'conv':4}, #0.0005
                  {'type':'e2ex', 'src':0, 'tgt':1, 'w':1, 'var':0, 'tau1':1*ms, 'tau2':0*ms, 'conv':100},
                  {'type':'e2m', 'src':0, 'tgt':1, 'w':0, 'var':0, 'tau1':50*ms, 'tau2':0*ms, 'conv':0, 'prob':1}
                 ]

        N = [500,50]

        amod = [0.01,0]
        ihold = [1,0]
        ihold_sigma = [0,0]

        factor_celltype = [[1,0,0.5,0],[1,0,0.5,0]] #[:][0] mean height of modulation, [:][1] variance of modulation, [:][2] probability that modulation is inverse (NOW if >0, 50% inverse input!)

        anoise = [0,0]

        len_train = 100
        len_test = 10

        lymax = 7

        xmin = 0; xmax=4 ; vec0w = np.arange(0,4.02,0.02)

        if "_0tau1v100_1tau1v1_" in do:
            xmin = 0; xmax=4; vec0w = np.arange(0,4.02,0.02)
        if "_0tau1v50_1tau1v50_" in do:
            xmin = 0; xmax=4; vec0w = np.arange(0,4.02,0.02)
        if "_0tau1v100_1tau1v100_" in do:
            xmin = 0; xmax=4; vec0w = np.arange(0,4.02,0.02)
        if "_0tau1v100_1tau1v50_" in do:
            xmin = 0; xmax=4; vec0w = np.arange(0,4.02,0.02)
        if "_0tau1v100_1tau1v10_" in do:
            xmin = 0; xmax=4; vec0w = np.arange(0,4.02,0.02)

        if "_specx7_" in do:
            xmin = 0; xmax=4 ; vec0w = np.arange(0,4.02,0.02)

        if "_specx2_" in do:
            xmin = 0; xmax=2; vec0w = np.arange(0,2.02,0.02)

        if "_specx3_" in do:
            xmin = 0; xmax=0.02; vec0w = np.arange(0,0.0202,0.0002)

        if "_specx5_" in do:
            xmin = 0; xmax=6 ; vec0w = np.arange(0,6.02,0.02)

        if "_specx6_" in do:
            xmin = 0; xmax=6 ; vec0w = np.arange(0,6.1,0.1)

        if ("_specx4_" in do):
            xmin = 0.014; xmax=0.016; vec0w = np.arange(0.014,0.016,0.00001)

            lymax = None
            lymin = None


        noise_out = 0

        prefix = prefix + "_ifun2b"

        use_c = True
        use_mpi = False


    mfratio = 0.5


    if "tau" in do:
        split = str(do).split("tau")
        for i in range(len(split)-1):
            iline = split[i][-1]
            split2 = split[i+1].split("v")
            inum = split2[0]
            val = split2[1].split("_")[0]
            prefix = prefix + "_" + iline + "tau" + inum + "v" + val
            conntype[int(iline)]['tau'+inum] = float(val)*ms

    if "wv" in do:
        split = str(do).split("wv")
        for i in range(len(split)-1):
            iline = split[i][-1]
            val = split[i+1].split("_")[0]
            prefix = prefix + "_" + iline + "wv" + val
            conntype[int(iline)]['w'] = float(val)

    if "varw" in do:
        split = str(do).split("varw")
        for i in range(len(split)-1):
            iline = split[i][-1]
            val = split[i+1].split("_")[0]
            prefix = prefix + "_" + iline + "varw" + val
            conntype[int(iline)]['var'] = float(val)

    if "probw" in do:
        split = str(do).split("probw")
        for i in range(len(split)-1):
            iline = split[i][-1]
            val = split[i+1].split("_")[0]
            prefix = prefix + "_" + iline + "probw" + val
            conntype[int(iline)]['prob'] = float(val)

    if "conv" in do:
        split = str(do).split("conv")
        for i in range(len(split)-1):
            iline = split[i][-1]
            val = split[i+1].split("_")[0]
            prefix = prefix + "_" + iline + "conv" + val
            if float(val) < 1:
                conntype[int(iline)]['conv'] = float(val)
            else:
                conntype[int(iline)]['conv'] = int(val)


    if "N" in do:
        split = str(do).split("N")
        for i in range(len(split)-1):
            iline = split[i][-1]
            val = split[i+1].split("_")[0]
            prefix = prefix + "_" + iline + "N" + val
            N[int(iline)] = int(val)


    if "_ihsigma" in do:
        ihsigma = str(do).split("_ihsigma")[1].split("_")[0]
        prefix = prefix + "_ihsigma" + ihsigma
        ihold_sigma = [float(ihsigma)]*len(N)

    if "_runs" in do:
        runs = str(do).split("_runs")[1].split("_")[0]
        runs = int(runs)

    if "_anoise" in do:
        split = str(do).split("anoise")
        val = split[1].split("_")[0]
        prefix = prefix + "_" + "anoise" + val
        anoise = [val]*len(N)

    scaleih = 1
    if "_1stih" in do:
        ndih = str(do).split("_1stih")[1].split("_")[0]
        prefix = prefix + "_1stih" + ndih
        scaleih = float(ndih)

    if "_2ndih" in do:
        ndih = str(do).split("_2ndih")[1].split("_")[0]
        prefix = prefix + "_2ndih" + ndih
        ihold[1] = float(ndih)*ihold[0]

    if "_0ih" in do:
        ndih = str(do).split("_0ih")[1].split("_")[0]
        prefix = prefix + "_0ih" + ndih
        ihold[0] = float(ndih)

    if "_dt" in do:
        sdt = str(do).split("_dt")[1].split("_")[0]
        prefix = prefix + "_dt" + sdt
        dt = float(sdt) * ms

    if "_noinv_" in do:
        for ii in range(len(factor_celltype)):
            factor_celltype[ii][2] = 0
        #factor_celltype = [[1,0,0]]*len(N)
        #amod = [0.01]
        prefix = prefix + "_noinv"

    if "_noinp" in do:
        noinp = str(do).split("_noinp")[1].split("_")[0]
        prefix = prefix + "_noinp" + noinp
        factor_celltype[0][3] = float(noinp)


    if "_mfrat" in do:
        mfrat = str(do).split("_mfrat")[1].split("_")[0]
        prefix = prefix + "_mfrat" + mfrat
        mfratio = float(mfrat)

    if "_ubc_" in do:
        prefix = prefix + "_ubc"

    if "_goc_" in do:
        prefix = prefix + "_goc"

    if "_2vec" in do:
        prefix = prefix + "_2vec"

    # rest here does not matter for lyapunov!
    prefix_ly = prefix[:]

    if "_ds" in do:
        ds = str(do).split("_ds")[1].split("_")[0]
        downsample = float(ds)
        prefix = prefix + "_ds" + str(int(ds))

    if "_lentrain" in do:
        lentrain = str(do).split("_lentrain")[1].split("_")[0]
        prefix = prefix + "_lentrain" + lentrain
        len_train = int(lentrain)

    if "_alpha" in do:
        al = str(do).split("_alpha")[1].split("_")[0]
        prefix = prefix + "_alpha" + al
        ridge_alpha = float(al)

    fitter = "ridge"
    if "_fit" in do:
        al = str(do).split("_fit")[1].split("_")[0]
        prefix = prefix + "_fit" + al
        fitter = al

    if "_noiseav" in do:
        split = str(do).split("noiseav")
        val = split[1].split("_")[0]
        prefix = prefix + "_" + "noiseav" + val
        noise_out = float(val)

    if "_amod" in do:
        a = str(do).split("_amod")[1].split("_")[0]
        prefix = prefix + "_amod" + a
        amod = [float(a)]*len(N)

    if "_2nda0" in do:
        prefix = prefix + "_2nda0"
        amod[1] = 0

    if "_prec" in do:
        a = str(do).split("_prec")[1].split("_")[0]
        prefix = prefix + "_prec" + a
        factor_recurrent[:][1] = float(a)
        print factor_recurrent

    normp = 1
    if ("_normp_" in do):
        normp = conntype[0]['conv']

    l1r = 1
    if "_l1r" in do:
        l1r0 = str(do).split("_l1r")[1].split("_")[0]
        l1r = float(l1r0)
        prefix = prefix + "_l1r" + l1r0

    ax = 0
    if "_ax" in do:
        ax = str(do).split("_ax")[1].split("_")[0]
        if myid == 0:
            exec("ax01=ax"+ax)
            if "_noly" not in do:
                exec("ax02=ax"+ax+'b')
            if "_fstd" in do:
                exec("ax02=ax"+ax+'b')

    pos = 1
    if "_pos" in do:
        pos = int(str(do).split("_pos")[1].split("_")[0][0])

    color = b1
    if "_color" in do:
        nc = str(do).split("_color")[1].split("_")[0]
        #exec 'color_vec = (np.array(['+ str(nc) +']), np.array(['+ str(nc) +']))'
        exec 'color = ' + str(nc)

    colgr = [b1,g1,r1]
    color_ly = "black"
    color_ly2 = o1
    if "_colgr" in do:
        nc = str(do).split("_colgr")[1].split("_")[0]
        if nc == "light":
            colgr = [b2,g2,r2]
            color_ly = "gray"
            color_ly2 = o2


    if "_4thb" in do:
        tau1_basis = np.array([0,0,0,0])
        tau2_basis = np.array([0.01,0.1,0.5,0.005])
        colgr = [b1,g1,r1,o1]
        prefix = prefix + "_4thb"
        if "_colgr" in do:
            nc = str(do).split("_colgr")[1].split("_")[0]
            if nc == "light":
                colgr = [b2,g2,r2,o2]

    if "_1b" in do:
        tau1_basis = np.array([0])
        tau2_basis = np.array([0.5])
        colgr = [r1]
        prefix = prefix + "_1b"

    if "_slowb" in do:
        tau2_basis = np.array([0.01,0.5,1.0])
        prefix = prefix + "_slowb"

    recurrent_filter = 2
    if "_refilt" in do:
        recurrent_filter = [int(str(do).split("_refilt")[1].split("_")[0])]
        prefix = prefix + "_refilt" + str(recurrent_filter)

    if "_reall" in do:
        recurrent_filter = range(len(tau2_basis))
        prefix = prefix + "_reall"

    if "_neghb" in do:
        tau1_basis = np.array([0,0,0])
        tau2_basis = np.array([-0.01,-0.1,-0.5])
        prefix = prefix + "_neghb"

    if "_4bthb" in do:
        tau1_basis = np.array([0,0,0,0])
        tau2_basis = np.array([0.01,0.1,0.5,-0.05])
        colgr = [b1,g1,r1,o1]
        prefix = prefix + "_4bthb"
        if "_colgr" in do:
            nc = str(do).split("_colgr")[1].split("_")[0]
            if nc == "light":
                colgr = [b2,g2,r2,o2]

    run_recurrent_filter = 1
    if "_runre" in do:
        run_recurrent_filter = int(str(do).split("_runre")[1].split("_")[0])
        prefix_ly = prefix
        prefix = prefix + "_runre" + str(run_recurrent_filter)

    exprecx = False
    if "_exprecx" in do:
        exprecx = True

    # DEFINE ANALYSIS METHOD


    if "_pca_" in do:

        delta = 3
        tdur = 0.1

        tstep = np.array([delta])

        istep0 = np.array([1])
        #istep1 = np.array([1])
        istep1 = np.zeros(len(istep0))
        istep = [istep0,istep1]
        istep_sigma = [0, 0]

        itest0 = np.array([])
        itest1 = np.array([])
        itest = [itest0,itest1]
        itest_sigma = [0, 0]
        ttest = np.array([])
        tstop = tstep[-1]+delta

        t_analysis_delay = 0.5
        t_analysis_stop = 1

        prefix = prefix + "_pca"

        t_plot_delay = 0.1
        t_plot_stop = tstop


    elif "_silentpca_" in do:

        delta = 3

        tstep = np.array([delta])

        istep0 = np.array([0])
        istep1 = np.zeros(len(istep0))
        istep = [istep0,istep1]
        istep_sigma = [0, 0]

        itest0 = np.array([])
        itest1 = np.array([])
        itest = [itest0,itest1]
        itest_sigma = [0, 0]

        ttest = np.array([])
        tstop = tstep[-1]+delta

        t_analysis_delay = 1
        delta = 4

        # pulse height
        istep0 = np.array([1, 0.6, 0.4, 0.8, 0.5, 0.3, 1, 0.7, 0.5, 0.9, 0.4, 1, 0.7, 0.6, 0.9, 0.6, 1, 0.8, 0.4, 0.5, 1, 0.5, 0.7, 1, 0.4, 0.6, 0.9])
        istep1 = np.zeros(len(istep0))
        istep = [istep0,istep1]
        istep_sigma = [0, 0]

        itest0 = np.array([1, 0.7, 1, 0.5])
        t_analysis_stop = 2

        prefix = prefix + "_silentpca"

        t_plot_delay = 0.1
        t_plot_stop = tstop


    elif "_basis_" in do:

        delta = 2  # delay from the start

        istep0 = np.array([1])
        istep1 = np.zeros(len(istep0))
        istep = [istep0,istep1]
        istep_sigma = [0, 0]

        itest0 = np.array([1])
        itest1 = np.zeros(len(itest0))
        itest = [itest0,itest1]
        itest_sigma = [0, 0]

        # pulse time
        tstep = np.arange(0,len(istep0)*3,3) + delta
        tstep = tstep + np.random.rand(len(tstep))*2

        ttest = tstep[-1] + np.arange(0,len(itest0)*3,3) + delta
        ttest = ttest + np.random.rand(len(ttest))*2

        tstop = ttest[-1] + delta

        t_analysis_delay = 0.5
        t_analysis_stop = 2

        prefix = prefix + "_basis"

        t_plot_delay = 0.1
        t_plot_stop = tstop


    elif "_noibas_" in do:

        delta = 2  # delay from the start

        istep0 = np.array([1])
        istep1 = np.zeros(len(istep0))
        istep = [istep0,istep1]
        istep_sigma = [0, 0]

        itest0 = np.array([1])
        itest1 = np.zeros(len(itest0))
        itest = [itest0,itest1]
        itest_sigma = [0, 0]

        # pulse time
        tstep = np.array([0,3]) + delta

        ttest = tstep[-1] + np.arange(0,len(itest0)*3,3) + 1
        ttest = ttest + np.random.rand(len(ttest))*2

        tstop = ttest[-1] + delta

        t_analysis_delay = 0.5
        t_analysis_stop = 2

        prefix = prefix + "_noibas"

        t_plot_delay = 0.1
        t_plot_stop = tstop


    elif "_basnoi_" in do:

        delta = 2  # delay from the start

        istep0 = np.array([1])
        istep1 = np.zeros(len(istep0))
        istep = [istep0,istep1]
        istep_sigma = [0, 0]

        itest0 = np.array([1])
        itest1 = np.zeros(len(itest0))
        itest = [itest0,itest1]
        itest_sigma = [0, 0]

        # pulse time
        tstep = np.arange(0,len(istep0)*3,3) + delta
        tstep = tstep + np.random.rand(len(tstep))*2

        ttest = tstep[-1] + np.array([0,3]) + delta

        tstop = ttest[-1] + delta

        t_analysis_delay = 0.5
        t_analysis_stop = 2

        prefix = prefix + "_basnoi"

        t_plot_delay = 0.1
        t_plot_stop = tstop


    elif "_cnoise_" in do:

        delta = 4
        t_analysis_delay = -5
        t_analysis_stop = 9

        # cnoise height
        istep0 = np.array([1])
        istep1 = np.zeros(len(istep0))
        istep = [istep0,istep1]
        istep_sigma = [0, 0]

        itest0 = np.array([1])
        itest1 = np.zeros(len(itest0))
        itest = [itest0,itest1]
        itest_sigma = [0, 0]

        # cnoise time, start stop vectors!
        tstep = np.array([0,len_train]) + delta
        ttest = tstep[-1] + np.array([0,len_test])

        tstop = ttest[-1]

        prefix = prefix + "_cnoise"

        t_plot_delay = 0.1
        t_plot_stop = tstop

    elif "_updown_" in do:

        delta = 4
        t_analysis_delay = -5
        t_analysis_stop = 9

        # cnoise height
        istep0 = np.array([1])
        istep1 = np.zeros(len(istep0))
        istep = [istep0,istep1]
        istep_sigma = [0, 0]

        itest0 = np.array([1])
        itest1 = np.zeros(len(itest0))
        itest = [itest0,itest1]
        itest_sigma = [0, 0]

        # cnoise time, start stop vectors!
        tstep = np.array([0,len_train]) + delta
        ttest = tstep[-1] + np.array([0,len_test])

        tstop = ttest[-1]

        prefix = prefix + "_updown"

        t_plot_delay = 0.1
        t_plot_stop = tstop

    elif "_step_" in do:

        t_analysis_delay = 1
        t_analysis_stop = 5

        # cnoise height
        istep0 = np.array([1])
        istep1 = np.zeros(len(istep0))
        istep = [istep0,istep1]
        istep_sigma = [0, 0]

        itest0 = np.array([1])
        itest1 = np.zeros(len(itest0))
        itest = [itest0,itest1]
        itest_sigma = [0, 0]

        # cnoise time, start stop vectors!
        tstep = np.array([len_train, len_test])
        ttest = np.array([0,0])

        tstop = len_train

        prefix = prefix + "_step"

        t_plot_delay = 0.1
        t_plot_stop = tstop


    if "wendv" in do:
        split = str(do).split("wendv")
        for i in range(len(split)-1):
            iline = split[i][-1]
            val = split[i+1].split("_")[0]
            prefix = prefix + "_" + iline + "wv" + val
            conntype[int(iline)]['w'] = float(val)


    # PLOT OPTIONS

    linestyle = "-"
    if "_dotted_" in do: linestyle = ":"
    if "_dashed_" in do: linestyle = "--"
    if "_dashdot_" in do: linestyle = "-."

    output_dim = 5

    #bin_width = dt

    # syn_out time constant
    tau1_pj = 1*ms
    tau2_pj = 10*ms


    pickle_prefix = prefix
    #if "_runre" in do:
    #    prefix_ly = prefix

    params = {'cellimport':cellimport, 'celltype':celltype, 'cell_exe':cell_exe, 'N':N, 'temperature':temperature,
            'ihold':ihold, 'ihold_sigma':ihold_sigma, 'give_freq':give_freq, 'do_run_constr':do_run_constr, 'do_run':do_run,
            'pickle_prefix':pickle_prefix, 'istart':istart, 'istop':istop, 'di':di, 'dt':dt, 'istep':istep, 'istep_sigma':istep_sigma,
            'itest':itest, 'itest_sigma':itest_sigma, 'tstep':tstep, 'ttest':ttest, 'tdur':tdur, 'do':do, 'fluct_s':fluct_s,
            'fluct_tau':fluct_tau, 't_analysis_delay':t_analysis_delay, 'scaleih':scaleih, 'fitter':fitter,
            't_analysis_stop':t_analysis_stop, 'conntype':conntype, 'tstop':tstop, 'tau1_pj':tau1_pj, 'tau2_pj':tau2_pj,
            'output_dim':output_dim, 't_plot_delay':t_plot_delay, 't_plot_stop':t_plot_stop, 'factor_celltype':factor_celltype,
            'tau1_ex':tau1_ex, 'anoise':anoise, 'mfratio':mfratio, 'l1r':l1r,
            'tau2_ex':tau2_ex,'n_syn_ex':n_syn_ex,'syn_max_mf':syn_max_mf,'g_syn_ex_s':g_syn_ex_s,'g_syn_ex':g_syn_ex,
            'mglufac_ex':mglufac_ex, 'noise_out':noise_out, 'noise_a':noise_a,'noise_syn':noise_syn,'noise_syn_tau':noise_syn_tau,
            'syn_ex_dist':syn_ex_dist, 'tau1_basis':tau1_basis,'tau2_basis':tau2_basis,
            'max_weight':max_weight, 'stdp':stdp, 'plot_all':plot_all, 'use_mpi':use_mpi, 'use_h5':use_h5, 'use_pc':use_pc,
            'myid':myid, 'dumpsave':dumpsave, 'nulltest':nulltest, 'amod':amod, 'seed0':seed0, 'bin_width':bin_width, 'downsample':downsample,
            'factor_recurrent':factor_recurrent, 'recurrent_filter':recurrent_filter, 'run_recurrent_filter':run_recurrent_filter,
            'delay_recurrent':delay_recurrent, 'teacher_forcing':teacher_forcing, 'exprecx':exprecx,
            'max_freq':max_freq, 'export':export, 'ridge_alpha':ridge_alpha, 'data_dir':data_dir}


    if "_run_" in do: # only run once!
        print "Running:", pickle_prefix
        pca_var, pca_start, pca_max, mean_spike_freq, basis_error, basis_vaf, signals, times2, x = randomnet(params)

        mstdx = []
        for j, t2 in enumerate(tau2_basis):
            xabs = abs(x[:,j])
            xabs = xabs[np.where(xabs>0)] # remove zero entries
            mstdx.append([ mean(xabs), std(xabs), max(x[:,j]), min(x[:,j]), float(len(np.where(x[:,j]==0)[0]))/len(x[:,j])*100 ])

        print "_run_ finsihed!"
        #print mstdx
        #print x

    if "_pcaplot_" in do:

        simprop = pickle_prefix
        filename = slugify(simprop)
        filepath = data_dir + str(filename) + "_pca"

        if use_h5:
            filepath = filepath + '.hdf5'
        else:
            filepath = filepath + '.p'

        if do_run_constr or (os.path.isfile(filepath) is False):

            params['plot_all'] = 0
            params['dumpsave']  = 0
            pca_var, pca_start, pca_max, mean_spike_freq, basis_error, basis_vaf, signals, times2, _ = randomnet(params)

        if myid == 0:

            if use_h5:
                results = rw_hdf5(filepath, export = export_m)
            else:
                results = pickle.load( gzip.GzipFile( filepath, "rb" ) )

            t, pca, pca_var  = results.get('t'), results.get('pca'), results.get('pca_var')

            exec("ax_pca = ax" + str(5+pos))

            #ax.text(-(t_analysis_delay+0.07), 0, str(round(pca_var[i],3)*100)+'%', ha="center", va="center", size=params['text.fontsize'])#, bbox=bbox_props)

            import matplotlib.patches as patches
            rect = patches.Rectangle((0,-4), 0.1, 10000, color="#C0C0C0")
            #rect.set_clip_on(False)
            ax_pca.add_patch(rect)

            linewidth0 = 1
            ax_pca.plot(t,pca[:,4]/max(abs(pca[:,4])),p1, linewidth=linewidth0, label=str(round(pca_var[4],3)*100)+'%')
            ax_pca.plot(t,pca[:,3]/max(abs(pca[:,3])),b1, linewidth=linewidth0, label=str(round(pca_var[3],3)*100)+'%')
            ax_pca.plot(t,pca[:,2]/max(abs(pca[:,2])),g1, linewidth=linewidth0, label=str(round(pca_var[2],3)*100)+'%')
            ax_pca.plot(t,pca[:,1]/max(abs(pca[:,1])),o1, linewidth=linewidth0, label=str(round(pca_var[1],3)*100)+'%')
            ax_pca.plot(t,pca[:,0]/max(abs(pca[:,0])),r1, linewidth=linewidth0, label=str(round(pca_var[0],3)*100)+'%')

            #ax.set_ylabel(str(round(pca_var[i],4)))
            ax_pca.axis(xmin=-0.5, xmax=1)
            ax_pca.axis(ymin=-1.05, ymax=1.05)



            lg = ax_pca.legend(labelspacing=0.05, loc=1, bbox_to_anchor=(1.05, 1.065), handlelength=0, handletextpad=0.1, numpoints=1) #
            #lg.draw_frame(False)
            fr = lg.get_frame()
            fr.set_lw(0.2)

            color_v2 = [p1,b1,g1,o1,r1] # [r1,o1,g1,b1,p1]
            txt = lg.get_texts()
            for i, t in enumerate(txt):
                t.set_color(color_v2[i])

            if "_first_" in do:
                adjust_spines(ax_pca, ['left','bottom'], d_out = d_out, d_down = d_down)
                ax_pca.set_ylabel("a.u.", labelpad=1)
                ax_pca.yaxis.set_ticks(array([-1,0,1]))
                ax_pca.set_yticklabels(('-1', '0', '1'))

            else:
                adjust_spines(ax_pca, ['bottom'], d_out = d_out, d_down = d_down)

            ax_pca.xaxis.set_ticks(array([-0.5,0,0.5,1]))
            ax_pca.set_xticklabels(('-0.5','0','0.5','1'))
            ax_pca.set_xlabel("s", labelpad=0)

            if "_noinv_" in do:
                ax_pca.set_title("PCA \n (w="+ str(conntype[0]['w']) +", no push-pull)")
            else:
                ax_pca.set_title("PCA \n (w="+ str(conntype[0]['w']) +")")

            filename="./figs/Pub/" + opt + "_" + str(pickle_prefix)
            if use_pc is False: plt.savefig(filename + ".pdf", dpi = 300) # save it
            plt.savefig(filename + imgf, dpi = 300) # save it

    if "_updown_" in do:
        pca_var, pca_start, pca_max, mean_spike_freq, basis_error, basis_vaf, signals, times2, _ = randomnet(params)

    if "_sigplot_" in do:

        params['seed0'] = 3
        simprop = pickle_prefix
        filename = slugify(simprop)
        filepath = data_dir + str(filename) + "_basis"

        if use_h5:
            filepath = filepath + '.hdf5'
        else:
            filepath = filepath + '.p'

        print filepath
        if do_run_constr or (os.path.isfile(filepath) is False):

            params['plot_all'] = 0
            params['dumpsave']  = 0

            if use_c:
                if myid == 0:
                    pca_var, pca_start, pca_max, mean_spike_freq, basis_error, basis_vaf, signals, times2, _ = randomnet(params)
            else:
                pca_var, pca_start, pca_max, mean_spike_freq, basis_error, basis_vaf, signals, times2, _ = randomnet(params)


        if myid == 0:

            if use_h5:
                results = rw_hdf5(filepath, export = export_m)
            else:
                results = pickle.load( gzip.GzipFile( filepath, "rb" ) )

            t_basis, basis_in  = results.get('t_basis'), results.get('basis_in')
            #print np.shape(basis_in)
            if "fig1" in opt:
                exec("ax_sig = ax" + str(5+pos))
            else:
                exec("ax_sig = ax" + str(pos))

            #ax.text(-(t_analysis_delay+0.07), 0, str(round(pca_var[i],3)*100)+'%', ha="center", va="center", size=params['text.fontsize'])#, bbox=bbox_props)

            import matplotlib.patches as patches
            import matplotlib.colors as mcolors

            c = mcolors.ColorConverter().to_rgb
            rvb = make_colormap([c(k1), c(ye1), 0.1, c(ye1), c(o1), 0.3,  c(o1), c(r1), 0.5, c(r1), c(g1), 0.7, c(g1), c(b1), 0.8, c(b1), c(p1)])
            
            linewidth0 = 1
            i = 0
            ii = 0

            if ("fig1" in opt) or ("fig3b" in opt):

                l = 20
                if "_specy1_" in do: l=100

                if "fig1test" in opt:
                    basis_in = mean(basis_in,1)
                    ax_sig.plot(t_basis-21,basis_in, linewidth=linewidth0)

                else:

                    if "fig1fens" in opt:
                        basis_in = basis_in[:,5:100]
                    elif "_specy1_" in do:
                        basis_in = basis_in
                    else:
                        basis_in = basis_in[:,36:100]

                    li = np.shape(basis_in)[1]
                    for i in range(li):
                        if ii < l:
                            mx = max(basis_in[:,i])
                            if mx > 0:
                                if "goc" in do:
                                    ax_sig.plot(t_basis-21,basis_in[:,i]*1, linewidth=linewidth0, color=rvb(1.*ii/l))
                                else:
                                    ax_sig.plot(t_basis-21,basis_in[:,i], linewidth=linewidth0, color=rvb(1.*ii/l))

                                ii += 1

                ax_sig.axis(xmin=-0.1, xmax=0.55)


                if "_test_" in do:
                    adjust_spines(ax_sig, ['left','bottom'], d_out = d_out, d_down = d_down)

                else:

                    if "_specy1_" in do:
                        #ax_sig.axis(ymin=0.4, ymax=1.1)
                        ax_sig.axis(ymin=0.718, ymax=0.746)
                    elif "_goc_" in do:
                        ax_sig.axis(ymin=0.00, ymax=0.015)
                    else:
                        ax_sig.axis(ymin=-0.005, ymax=0.5)

                    if "_first_" in do:
                        if "_specy1_" in do:
                            adjust_spines(ax_sig, ['left','bottom'], d_out = d_out, d_down = d_down)
                            ax_sig.set_ylabel(r"$\mathsf{z_i(t)}$", labelpad=-4)
                            ax_sig.yaxis.set_ticks(array([0.72, 0.73, 0.74]))
                            ax_sig.set_yticklabels(('0.72', '', '0.74'))
                        else:
                            if "_goc_" in do:
                                adjust_spines(ax_sig, ['left','bottom'], d_out = d_out, d_down = d_down)
                                #if "_end_" in do: ax_sig.set_ylabel(r"$\mathsf{z_i(t)}$", labelpad=0)
                                ax_sig.yaxis.set_ticks(array([0, 0.005,0.01,0.015]))
                                ax_sig.set_yticklabels(('0', '0.005', '0.01', '0.015'))
                            else:
                                adjust_spines(ax_sig, ['left','bottom'], d_out = d_out, d_down = d_down)
                                if "_end_" in do: ax_sig.set_ylabel(r"$\mathsf{z_i(t)}$", labelpad=0)
                                ax_sig.yaxis.set_ticks(array([0,0.1,0.2,0.3,0.4,0.5]))
                                ax_sig.set_yticklabels(('0', '0.1', '0.2', '0.3', '0.4', '0.5'))

                    else:
                        adjust_spines(ax_sig, ['bottom'], d_out = d_out, d_down = d_down)

                ax_sig.xaxis.set_ticks(array([0,0.2,0.4]))
                ax_sig.set_xticklabels(('0','0.2','0.4'))
                if ("fig1" in opt):
                    ax_sig.set_xlabel("s", labelpad=2)
                else:
                    ax_sig.set_xlabel("s", labelpad=2)


                if "_grc_" in do:
                    if "_noinv_" in do:
                        ax_sig.set_title("GC rates \n (w="+ str(conntype[0]['w']) +", no push-pull)")
                    else:
                        ax_sig.set_title("GC rates \n (w="+ str(conntype[0]['w']) +")")

                if "_ubc_" in do:
                    if "_noinv_" in do:
                        ax_sig.set_title("UBC rates \n (w="+ str(conntype[0]['w']) +", no push-pull)")
                    else:
                        ax_sig.set_title("UBC rates \n (w="+ str(conntype[0]['w']) +")")


                if ("fig3b" in opt):
                    rect = patches.Rectangle((0,0.48), 0.05, 0.01, color="#000000", zorder=1000)
                elif "_specy1_" in do:
                    rect = patches.Rectangle((0,0.747), 0.05, 0.0005, color="#000000", zorder=1000)
                else:
                    rect = patches.Rectangle((0,0.51), 0.05, 0.01, color="#000000", zorder=1000)
                rect.set_clip_on(False)
                ax_sig.add_patch(rect)

            else:

                l = 20
                basis_in = basis_in[:,50:100]
                li = np.shape(basis_in)[1]
                for i in range(li):
                    if ii < l:
                        mx = max(basis_in[:,i])
                        if mx > 0:
                            ax_sig.plot(t_basis-21,((basis_in[:,i]-basis_in[0,i])), linewidth=linewidth0, color=rvb(1.*ii/l))
                            ii += 1

                ax_sig.axis(xmin=-0.25, xmax=1)

                if "_first_" in do:
                    adjust_spines(ax_sig, ['left','bottom'], d_out = d_out, d_down = d_down)
                    ax_sig.set_ylabel(r"$\mathsf{z_i(t)}$", labelpad=-7)

                else:
                    adjust_spines(ax_sig, ['bottom'], d_out = d_out, d_down = d_down)

                ax_sig.xaxis.set_ticks(array([0,0.5,1]))
                ax_sig.set_xticklabels(('0','','1'))
                ax_sig.set_xlabel("s", labelpad=0)


            filename="./figs/Pub/" + opt + "_" + str(pickle_prefix)
            if use_pc is False: plt.savefig(filename + ".pdf", dpi = 300) # save it
            plt.savefig(filename + imgf, dpi = 300) # save it


    if "_runplot_" in do:

        params['seed0'] = 3
        simprop = pickle_prefix
        filename = slugify(simprop)
        filepath = data_dir + str(filename) + "_basis"

        if use_h5:
            filepath = filepath + '.hdf5'
        else:
            filepath = filepath + '.p'

        #print filepath
        if do_run_constr or (os.path.isfile(filepath) is False):

            params['plot_all'] = 0
            params['dumpsave']  = 0
            pca_var, pca_start, pca_max, mean_spike_freq, basis_error, basis_vaf, signals, times2, _ = randomnet(params)

        if myid == 0:

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
                vaf = 1 - ( var(filtered_test_v[:,i]-utest[:,i]) / var(filtered_test_v[:,i] ) )
                if vaf < 0: vaf = 0

                error = (scipy.stats.pearsonr(utest[:,i], filtered_test_v[:,i])[0])**2

                basis_error.append(error)
                basis_vaf.append(vaf)
                print "Basis error:", basis_error[-1], "vaf:", basis_vaf[-1], "weight range:", max(x[:,i]), min(x[:,i]), "mean:", mean(abs(np.array([a for a in x[:,i] if a != 0]))), "zero weight percentage:", float(len(np.where(x[:,i]==0)[0]))/len(x[:,i])*100
            print "Basis error all:", mean(basis_error), "vaf all:", mean(basis_vaf)


            color_v.append([])

            for i in range(3):

                exec("ax_noise = ax" + str(i+1))
                exec("ax_basis = ax" + str(i+1+3))

                #filtered_est = filtered_est_v[:,i]
                filtered_test = filtered_test_v[:,i]
                filtered_basis = filtered_basis_v[:,i]

                #uest1 = uest[:,i]
                utest1 = utest[:,i]
                ubasis1 = ubasis[:,i]

                #ax_basis.plot(t_in, filtered_est/max(filtered_basis), 'k--')
                #ax_basis.plot(t_in, est, 'k:')
                #ax_basis.plot(t_in, uest1/max(filtered_basis), color = colgr[i])

                zo = 1
                if conntype[0]['w'] == 1.4:
                    zo = 2
                ax_noise.plot(t_test-27.9, utest1/max(filtered_basis), color = colgr[i], linewidth=2, linestyle=linestyle, zorder=zo)
                ax_basis.plot(t_basis-21, ubasis1/max(filtered_basis), color = colgr[i], label="w = "+ str(conntype[0]['w']), linewidth=2, linestyle=linestyle, zorder=zo)

                lg = ax_basis.legend(labelspacing=0.05, bbox_to_anchor=(0.42,0.85), loc=3, handlelength=2, handletextpad=0.1, numpoints=2)
                fr = lg.get_frame()
                fr.set_lw(0.2)

                color_v[-1].append(colgr[i])
                txt = lg.get_texts()
                for j, t in enumerate(txt):
                    t.set_color(color_v[j][i])

                if "_first_" in do:
                    #ax_noise.plot(t_test-27.9, test, color="gray", linewidth=1)
                    #ax_basis.plot(t_basis-21, basis, color="gray", linewidth=1)
                    pass
                if "_end_" in do:
                    ax_noise.plot(t_test-27.9, filtered_test/max(filtered_basis), 'k-', linewidth=1)
                    ax_basis.plot(t_basis-21, filtered_basis/max(filtered_basis), 'k-', linewidth=1)

                if "fig2b" in opt:
                    ax_basis.axis(ymin=-0.2, ymax=1.45)
                else:
                    ax_basis.axis(ymin=-1.0, ymax=1.3)
                ax_noise.axis(ymin=-1.4, ymax=1.6)

                if i == 0:
                    adjust_spines(ax_basis, ['left','bottom'], d_out = d_out, d_down = d_down)

                    #ax_basis.axis(xmin=-0.05, xmax=0.2)
                    #ax_noise.axis(xmin=0, xmax=0.25)
                    #ax_basis.xaxis.set_ticks(array([0,0.1,0.2]))
                    #ax_basis.set_xticklabels(('0', '0.1', '0.2'))

                    ax_basis.axis(xmin=-0.1, xmax=0.6)
                    ax_noise.axis(xmin=1.4, xmax=2.2)
                    ax_basis.xaxis.set_ticks(array([0,0.2,0.4,0.6]))
                    ax_basis.set_xticklabels(('0', '0.2', '0.4', '0.6'))

                    if "fig2b" in opt:
                        ax_basis.yaxis.set_ticks(array([0,1]))
                        ax_basis.set_yticklabels(('0','1'))
                    else:
                        ax_basis.yaxis.set_ticks(array([-1,0,1]))
                        ax_basis.set_yticklabels(('-1','0','1'))

                    if ("fig1" in opt):
                        ax_basis.set_xlabel("s", labelpad=2)
                    else:
                        ax_basis.set_xlabel("s", labelpad=0)

                    ax_basis.set_ylabel("a.u.", labelpad=2)
                    adjust_spines(ax_noise, ['left'], d_out = d_out, d_down = d_down)
                    ax_noise.yaxis.set_ticks(array([-1,0,1]))
                    ax_noise.set_yticklabels(('-1', '0', '1'))
                    ax_noise.set_ylabel("a.u.", labelpad=1)
                    ax_noise.set_title(r"Filter $\mathsf{\tau}$ = 10 ms")

                if i == 1:
                    adjust_spines(ax_basis, ['bottom'], d_out = d_out, d_down = d_down)

                    ax_basis.axis(xmin=-0.1, xmax=0.6)
                    ax_noise.axis(xmin=1.4, xmax=2.2)
                    ax_basis.xaxis.set_ticks(array([0,0.2,0.4,0.6]))
                    ax_basis.set_xticklabels(('0', '0.2', '0.4', '0.6'))

                    if ("fig1" in opt):
                        ax_basis.set_xlabel("s", labelpad=2)
                    else:
                        ax_basis.set_xlabel("s", labelpad=0)

                    adjust_spines(ax_noise, [], d_out = d_out, d_down = d_down)
                    ax_noise.set_title(r"Filter $\mathsf{\tau}$ = 100 ms")

                if i == 2:
                    adjust_spines(ax_basis, ['bottom'], d_out = d_out, d_down = d_down)

                    #ax_basis.axis(xmin=-0.4, xmax=2)
                    #ax_noise.axis(xmin=0, xmax=2.4)
                    #ax_basis.xaxis.set_ticks(array([0,1,2]))
                    #ax_basis.set_xticklabels(('0', '1', '2'))

                    ax_basis.axis(xmin=-0.1, xmax=0.6)
                    ax_noise.axis(xmin=1.4, xmax=2.2)
                    ax_basis.xaxis.set_ticks(array([0,0.2,0.4,0.6]))
                    ax_basis.set_xticklabels(('0', '0.2', '0.4', '0.6'))

                    if ("fig1" in opt):
                        ax_basis.set_xlabel("s", labelpad=2)
                    else:
                        ax_basis.set_xlabel("s", labelpad=0)

                    adjust_spines(ax_noise, [], d_out = d_out, d_down = d_down)
                    ax_noise.set_title(r"Filter $\mathsf{\tau}$ = 500 ms")

            #filename="./figs/Pub/" + str(pickle_prefix)
            filename="./figs/Pub/" + opt + "_" + str(pickle_prefix)
            if use_pc is False: plt.savefig(filename + ".pdf", dpi = 300) # save it
            plt.savefig(filename + imgf, dpi = 300) # save it


    if "_relalpha_" in do:

        prefix_plot = prefix[:]

        vec_string = str(do).split("_vec")[1].split("_")[0]
        iline = vec_string[0]
        stype = vec_string[1:]

        prefix_plot = prefix_plot + "_vec" + iline + stype
        filepath = data_dir + prefix_plot + ".hdf5"

        pl = rw_hdf5(filepath, export = export_m)

        p_v = pl['p_v']

        f_v_m = nanmean(pl['f_v'][0])
        stdf_v_m = nanmean(pl['stdf_v'][0])

        vaf1_v_m = nanmean(pl['vaf1_v'])
        vaf2_v_m = nanmean(pl['vaf2_v'])
        vaf3_v_m = nanmean(pl['vaf3_v'])
        err1_v_m = nanmean(pl['err1_v'])
        err2_v_m = nanmean(pl['err2_v'])
        err3_v_m = nanmean(pl['err3_v'])
        if "_4thb" in do:
            vaf4_v_m = nanmean(pl['vaf4_v'])
            err4_v_m = nanmean(pl['err4_v'])
        ly_v = pl['ly_v']

        mx1_v_m = nanmean(pl['mx1_v'][:,:,0])
        mx2_v_m = nanmean(pl['mx2_v'][:,:,0])
        mx3_v_m = nanmean(pl['mx3_v'][:,:,0])

        stdx1_v_m = nanmean(pl['mx1_v'][:,:,1])
        stdx2_v_m = nanmean(pl['mx2_v'][:,:,1])
        stdx3_v_m = nanmean(pl['mx3_v'][:,:,1])

        nullp1_v_m = nanmean(pl['nullp1_v'])
        nullp2_v_m = nanmean(pl['nullp2_v'])
        nullp3_v_m = nanmean(pl['nullp3_v'])

        exec("ax_a = ax" + str(pos))

        m = np.array([err1_v_m, err2_v_m, err3_v_m])
        m = mean(m,0)
        m_r = max(m)
        print ridge_alpha, m_r
        ax_a.semilogx(ridge_alpha, m_r, '*')

        filename="./figs/Pub/" + opt + "_" + str(pickle_prefix)
        if use_pc is False: plt.savefig(filename + ".pdf", dpi = 300) # save it
        plt.savefig(filename + imgf, dpi = 300) # save it


    if "_lyap_" in do:
        pickle_prefix_ly = prefix_ly + "_" + iline + "w" + "v" + str(val)
        params['pickle_prefix'] = pickle_prefix_ly
        params['pickle_prefix_ly'] = pickle_prefix_ly
        ly = lyap(params, runs = 1)

    else:

        params['plot_all'] = 1
        params['dumpsave']  = 0

        if "_vec" in do:

            if myid == 0: print "- Vector run: " + pickle_prefix

            params['plot_all'] = 2
            params['dumpsave']  = 1

            vec_string = str(do).split("_vec")[1].split("_")[0]
            iline = vec_string[0]
            stype = vec_string[1:]

            if "_2vec" in do:
                vec_string2 = str(do).split("_2vec")[1].split("_")[0]
                iline2 = vec_string2[0]
                stype2 = vec_string2[1:]

            exec("vec = vec"+iline+stype)

            p_v = nans(len(vec))
            ly_v = nans(len(vec))

            f_v = nans((len(N), runs, len(vec)))
            stdf_v = nans((len(N), runs, len(vec)))

            err1_v = nans((runs, len(vec)))
            vaf1_v = nans((runs, len(vec)))
            mx1_v = nans((runs, len(vec), 2))
            minmaxx1_v = nans((runs, len(vec), 2))
            nullp1_v = nans((runs, len(vec)))

            err2_v = nans((runs, len(vec)))
            vaf2_v = nans((runs, len(vec)))
            mx2_v = nans((runs, len(vec), 2))
            minmaxx2_v = nans((runs, len(vec), 2))
            nullp2_v = nans((runs, len(vec)))

            err3_v = nans((runs, len(vec)))
            vaf3_v = nans((runs, len(vec)))
            mx3_v = nans((runs, len(vec), 2))
            minmaxx3_v = nans((runs, len(vec), 2))
            nullp3_v = nans((runs, len(vec)))

            if "_4thb" in do:
                err4_v = nans((runs, len(vec)))
                vaf4_v = nans((runs, len(vec)))
                mx4_v = nans((runs, len(vec), 2))
                minmaxx4_v = nans((runs, len(vec), 2))
                nullp4_v = nans((runs, len(vec)))


            prefix_plot = prefix[:]
            if "_specx" in do:
                spec = str(do).split("_specx")[1].split("_")[0]
                prefix_plot = prefix_plot + "_specx" + str(int(spec))

            prefix_plot = prefix_plot + "_vec" + iline + stype

            filepath = data_dir + prefix_plot + ".hdf5"

            if (os.path.isfile(filepath) is False) or (fit_restart_mean is True):

                for r in range(runs):

                    if "_relyap_" in do:
                        if r > 0:
                            restart_lyap = False
                        else:
                            restart_lyap = True

                    params['seed0'] = r+1

                    if use_c == False:

                        for i, val in enumerate(vec):

                            p = val

                            pickle_prefix = prefix + "_" + iline + stype + "v" + str(val)
                            filepath = data_dir + pickle_prefix + "_fit_results"
                            pickle_prefix_ly = prefix_ly + "_" + iline + stype + "v" + str(val)

                            if r > 0: filepath = filepath + "_run" + str(r)

                            if use_h5:
                                filepath = filepath + '.hdf5'
                            else:
                                filepath = filepath + '.p'

                            filepath2 = data_dir + pickle_prefix + "_results_pop_randomnet.hdf5"
                            if ((os.path.isfile(filepath) is True) or (os.path.isfile(filepath2) is True)) and (fit_restart is False):

                            #if (os.path.isfile(filepath) is True) and (fit_restart is False):

                               if myid == 0: print "- Already computed constr. for", p

                            else:

                                params['conntype'][int(iline)][stype] = val

                                if "_2vec" in do:
                                    params['conntype'][int(iline2)][stype2] = val

                                if myid == 0: print "- This run: " + pickle_prefix

                                if "_norunly" not in do:
                                    filepath = data_dir + pickle_prefix_ly + "_lyap.hdf5"
                                    if (os.path.isfile(filepath) is False) or restart_lyap:
                                        params['plot_all'] = 0
                                        params['dumpsave']  = 0
                                        params['pickle_prefix'] = pickle_prefix_ly
                                        params['pickle_prefix_ly'] = pickle_prefix_ly
                                        ly = lyap(params, runs = 10)
                                        params['plot_all'] = 2
                                        params['dumpsave']  = 1

                                params['pickle_prefix'] = pickle_prefix
                                params['pickle_prefix_ly'] = pickle_prefix_ly

                                params['do_run'] = 1
                                params['do_run_constr'] = 2 # do not try to run construction
                                #params['seed0'] = r+1
                                pca_var, pca_start, pca_max, mean_spike_freq, err, vaf, signals, times2, x = randomnet(params)

                        barrier(use_mpi, use_pc)

                        for i in range(int(myid), len(vec), int(psize)):

                            val = vec[i]

                            pickle_prefix = prefix + "_" + iline + stype + "v" + str(val)
                            filepath = data_dir + pickle_prefix + "_fit_results"
                            pickle_prefix_ly = prefix_ly + "_" + iline + stype + "v" + str(val)
                            if r > 0: filepath = filepath + "_run" + str(r)

                            p = val

                            if use_h5:
                                filepath = filepath + '.hdf5'
                            else:
                                filepath = filepath + '.p'

                            if (os.path.isfile(filepath) is True) and (fit_restart is False):

                                #p, err, vaf, ly, mstdx = save_results(pickle_prefix, use_pc = use_pc, use_h5 = use_h5)
                                print "- Already computed for", p, "\n" #, ", err:", err, ", vaf:", vaf, ", ly:", ly, ", run:", r ,"\n"

                            else:

                                params['conntype'][int(iline)][stype] = val
                                params['pickle_prefix'] = pickle_prefix

                                print "- id:", myid, ", this constr. run: " + pickle_prefix, ", run:", r ,"\n"

                                params['plot_all'] = 1
                                params['do_run'] = 0
                                params['do_run_constr'] = 3 # run construction on one node
                                pca_var, pca_start, pca_max, mean_spike_freq, err, vaf, signals, times2, x = randomnet(params)

                                mstdx = []
                                for j, t2 in enumerate(tau2_basis):
                                    xabs = abs(x[:,j])
                                    xabs = xabs[np.where(xabs>0)] # remove zero entries
                                    mstdx.append([ mean(xabs), std(xabs), max(x[:,j]), min(x[:,j]), float(len(np.where(x[:,j]==0)[0]))/len(x[:,j])*100 ])

                                fstd = nans((len(N),2))
                                if mean_spike_freq is not None:

                                    for n in range(len(N)):
                                        fstd[n,0] = mean(mean_spike_freq[n])
                                        fstd[n,1] = std(mean_spike_freq[n])

                                print p, fstd


                                filepath = data_dir + pickle_prefix_ly + "_lyap.hdf5"
                                ly = rw_hdf5(filepath, export = export)['ly']

                                save_results(pickle_prefix, p, err = err, vaf = vaf, ly = ly, mstdx = mstdx, fstd = fstd, params = params, use_h5 = use_h5, run = r, data_dir=data_dir)

                                filepath = data_dir + pickle_prefix + "_results_pop_randomnet.hdf5"
                                os.remove(filepath)

                        barrier(use_mpi = True, use_pc = use_pc)

                    else:

                        params['plot_all'] = 2
                        params['dumpsave']  = 0

                        for j in range(int(myid), len(vec), int(psize)):

                            val = vec[j]

                            pickle_prefix = prefix + "_" + iline + stype + "v" + str(val)
                            filepath = data_dir + pickle_prefix + "_fit_results"
                            pickle_prefix_ly = prefix_ly + "_" + iline + stype + "v" + str(val)
                            if r > 0: filepath = filepath + "_run" + str(r)

                            p = val

                            if use_h5:
                                filepath = filepath + '.hdf5'
                            else:
                                filepath = filepath + '.p'

                            params['conntype'][int(iline)][stype] = val

                            if (os.path.isfile(filepath) is True) and (fit_restart is False):

                                print "- Already computed for", p, "\n"

                                if restart_lyap:
                                    print "- Redoing Lyap\n"
                                    params['pickle_prefix'] = pickle_prefix_ly
                                    params['pickle_prefix_ly'] = pickle_prefix_ly
                                    ly = lyap(params, runs = 10)

                                    params['pickle_prefix'] = pickle_prefix
                                    params['pickle_prefix_ly'] = pickle_prefix_ly
                                    p, err, vaf, _, mstdx, fstd = save_results(pickle_prefix, use_pc = use_pc, use_h5 = use_h5, run = r, export = export, data_dir=data_dir)
                                    save_results(pickle_prefix, p, err = err, vaf = vaf, ly = ly, mstdx = mstdx, fstd = fstd, params = params, use_h5 = use_h5, run = r, data_dir=data_dir)

                                if readd_lyap:
                                    filepath_ly = data_dir + pickle_prefix_ly + "_lyap.hdf5"
                                    ly = rw_hdf5(filepath_ly, export = export)['ly']
                                    p, err, vaf, _, mstdx, fstd = save_results(pickle_prefix, use_pc = use_pc, use_h5 = use_h5, run = r, export = export, data_dir=data_dir)
                                    save_results(pickle_prefix, p, err = err, vaf = vaf, ly = ly, mstdx = mstdx, fstd = fstd, params = params, use_h5 = use_h5, run = r, data_dir=data_dir)


                            else:

                                print "- This run (mpi used for loop): " + pickle_prefix

                                if "_norunly" not in do:
                                    filepath = data_dir + pickle_prefix_ly + "_lyap.hdf5"
                                    if (os.path.isfile(filepath) is False) or restart_lyap:
                                        params['pickle_prefix'] = pickle_prefix_ly
                                        params['pickle_prefix_ly'] = pickle_prefix_ly
                                        ly = lyap(params, runs = 10)
                                    else:
                                        ly = rw_hdf5(filepath, export = export)['ly']
                                else:
                                    ly = -1

                                params['pickle_prefix'] = pickle_prefix
                                params['pickle_prefix_ly'] = pickle_prefix_ly

                                pca_var, pca_start, pca_max, mean_spike_freq, err, vaf, signals, times2, x = randomnet(params)

                                mstdx = []
                                for j, t2 in enumerate(tau2_basis):
                                    xabs = abs(x[:,j])
                                    xabs = xabs[np.where(xabs>0)] # remove zero entries
                                    mstdx.append([ mean(xabs), std(xabs), max(x[:,j]), min(x[:,j]), float(len(np.where(x[:,j]==0)[0]))/len(x[:,j])*100 ])

                                fstd = nans((len(N),2))
                                if mean_spike_freq is not None:

                                    for n in range(len(N)):
                                        fstd[n,0] = mean(mean_spike_freq[n])
                                        fstd[n,1] = std(mean_spike_freq[n])

                                print p, fstd

                                save_results(pickle_prefix, p, err = err, vaf = vaf, ly = ly, mstdx = mstdx, fstd = fstd, params = params, use_h5 = use_h5, run = r, data_dir=data_dir)

                        barrier(use_mpi = True, use_pc = use_pc)

                    if myid == 0:

                        for i, val in enumerate(vec):

                            pickle_prefix = prefix + "_" + iline + stype + "v" + str(val)
                            pickle_prefix_ly = prefix_ly + "_" + iline + stype + "v" + str(val)
                            p = val

                            p, err, vaf, ly, mstdx, fstd = save_results(pickle_prefix, use_pc = use_pc, use_h5 = use_h5, run = r, export = export, data_dir=data_dir)

                            filepath = data_dir + pickle_prefix_ly + "_lyap.hdf5"
                            if (os.path.isfile(filepath) is True):
                                ly = rw_hdf5(filepath, export = export)['ly']

                            p_v[i] = p
                            ly_v[i] = ly

                            if fstd is not None:
                                for n in range(len(N)):
                                    f_v[n,r,i] = fstd[n,0]
                                    stdf_v[n,r,i] = fstd[n,1]

                            err1_v[r,i] = err[0]
                            vaf1_v[r,i] = vaf[0]
                            mx1_v[r,i,0] = mstdx[0][0]
                            mx1_v[r,i,1] = mstdx[0][1]
                            minmaxx1_v[r,i,0] = mstdx[0][2]
                            minmaxx1_v[r,i,1] = mstdx[0][3]
                            nullp1_v[r,i] = mstdx[0][4]

                            err2_v[r,i] = err[1]
                            vaf2_v[r,i] = vaf[1]
                            mx2_v[r,i,0] = mstdx[1][0]
                            mx2_v[r,i,1] = mstdx[1][1]
                            minmaxx2_v[r,i,0] = mstdx[1][2]
                            minmaxx2_v[r,i,1] = mstdx[1][3]
                            nullp2_v[r,i] = mstdx[1][4]

                            err3_v[r,i] = err[2]
                            vaf3_v[r,i] = vaf[2]
                            mx3_v[r,i,0] = mstdx[2][0]
                            mx3_v[r,i,1] = mstdx[2][1]
                            minmaxx3_v[r,i,0] = mstdx[2][2]
                            minmaxx3_v[r,i,1] = mstdx[2][3]
                            nullp3_v[r,i] = mstdx[2][4]

                            if "_4thb" in do:
                                err4_v[r,i] = err[3]
                                vaf4_v[r,i] = vaf[3]
                                mx4_v[r,i,0] = mstdx[3][0]
                                mx4_v[r,i,1] = mstdx[3][1]
                                minmaxx4_v[r,i,0] = mstdx[3][2]
                                minmaxx4_v[r,i,1] = mstdx[3][3]
                                nullp4_v[r,i] = mstdx[3][4]


                        vaf1_v_m = nanmean(vaf1_v)
                        vaf2_v_m = nanmean(vaf2_v)
                        vaf3_v_m = nanmean(vaf3_v)

                        err1_v_m = nanmean(err1_v)
                        err2_v_m = nanmean(err2_v)
                        err3_v_m = nanmean(err3_v)

                        if "_4thb" in do:
                            vaf4_v_m = nanmean(vaf4_v)
                            err4_v_m = nanmean(err4_v)

                        f_v_m = nanmean(f_v[0])
                        stdf_v_m = nanmean(stdf_v[0])

                        #print err1_v[:,10]

                        if (r == runs-1):

                            pl = {}
                            pl['p_v'] = p_v

                            pl['f_v'] = f_v
                            pl['stdf_v'] = stdf_v

                            pl['vaf1_v'] = vaf1_v
                            pl['vaf2_v'] = vaf2_v
                            pl['vaf3_v'] = vaf3_v

                            pl['err1_v'] = err1_v
                            pl['err2_v'] = err2_v
                            pl['err3_v'] = err3_v

                            pl['mx1_v'] = mx1_v
                            pl['mx2_v'] = mx2_v
                            pl['mx3_v'] = mx3_v

                            pl['minmaxx1_v'] = minmaxx1_v
                            pl['minmaxx2_v'] = minmaxx2_v
                            pl['minmaxx3_v'] = minmaxx3_v

                            pl['nullp1_v'] = nullp1_v
                            pl['nullp2_v'] = nullp2_v
                            pl['nullp3_v'] = nullp3_v

                            if "_4thb" in do:
                                pl['vaf4_v'] = vaf4_v
                                pl['err4_v'] = err4_v
                                pl['mx4_v'] = mx4_v
                                pl['minmaxx4_v'] = minmaxx4_v
                                pl['nullp4_v'] = nullp4_v

                            pl['ly_v'] = ly_v
                            filepath = data_dir + prefix_plot + ".hdf5"
                            rw_hdf5(filepath, pl)


                        fig2 = plt.figure('vectemp')
                        plt.clf()

                        gsX = matplotlib.gridspec.GridSpec(5, 1,
                           width_ratios=[1],
                           height_ratios=[1,1,1,1,1]
                           )

                        ax01b = plt.subplot(gsX[0,0])
                        ax02b = plt.subplot(gsX[1,0])
                        ax03b = plt.subplot(gsX[2,0])
                        ax04b = plt.subplot(gsX[3,0])
                        ax05b = plt.subplot(gsX[4,0])

                        ax01b.plot(p_v, err1_v[r,:], color=colgr[0], linestyle = linestyle, linewidth=linewidth, clip_on = False)
                        if "_4thb" in do: ax01b.plot(p_v, err4_v[r,:], color=colgr[3], linestyle = linestyle, linewidth=linewidth, clip_on = False)
                        ax01b.plot(p_v, err2_v[r,:], color=colgr[1], linestyle = linestyle, linewidth=linewidth, clip_on = False)
                        ax01b.plot(p_v, err3_v[r,:], color=colgr[2], linestyle = linestyle, linewidth=linewidth, clip_on = False)

                        ax02b.plot(p_v, ly_v, color=color_ly2, linestyle = linestyle, linewidth=linewidth)
                        ax02b.axhline(y=0, color="k", linestyle = "dashed", linewidth=0.5)
                        ax02b.axis(xmin=xmin, xmax=xmax)

                        ax03b.plot(p_v, mx1_v[r,:,0], color=colgr[0], linestyle = linestyle, linewidth=linewidth, clip_on = False)
                        if "_4thb" in do: ax03b.plot(p_v, mx4_v[r,:], color=colgr[3], linestyle = linestyle, linewidth=linewidth, clip_on = False)
                        ax03b.plot(p_v, mx2_v[r,:,0], color=colgr[1], linestyle = linestyle, linewidth=linewidth, clip_on = False)
                        ax03b.plot(p_v, mx3_v[r,:,0], color=colgr[2], linestyle = linestyle, linewidth=linewidth, clip_on = False)

                        ax04b.plot(p_v, nullp1_v[r,:], color=colgr[0], linestyle = linestyle, linewidth=linewidth, clip_on = False)
                        if "_4thb" in do: ax04b.plot(p_v, nullp4_v[r,:], color=colgr[3], linestyle = linestyle, linewidth=linewidth, clip_on = False)
                        ax04b.plot(p_v, nullp2_v[r,:], color=colgr[1], linestyle = linestyle, linewidth=linewidth, clip_on = False)
                        ax04b.plot(p_v, nullp3_v[r,:], color=colgr[2], linestyle = linestyle, linewidth=linewidth, clip_on = False)

                        ax05b.errorbar(p_v, f_v[0,r,:], yerr = stdf_v[0,r,:], color=color_ly2, linestyle = linestyle, linewidth=linewidth, clip_on = False)

                        filename="./figs/Pub/" + str(prefix_plot) + "_run" + str(r)
                        if use_pc is False: plt.savefig(filename  + ".pdf", dpi = 300) # save it
                        plt.savefig(filename + imgf, dpi = 300) # save it

                    barrier(use_mpi = True, use_pc = use_pc)

            else:

                if myid == 0:

                    import csv

                    pl = rw_hdf5(filepath, export = export_m)

                    p_v = pl['p_v']

                    f_v_m = nanmean(pl['f_v'][0])
                    stdf_v_m = nanmean(pl['stdf_v'][0])

                    vaf1_v_m = nanmean(pl['vaf1_v'])
                    vaf2_v_m = nanmean(pl['vaf2_v'])
                    vaf3_v_m = nanmean(pl['vaf3_v'])
                    err1_v_m = nanmean(pl['err1_v'])
                    err2_v_m = nanmean(pl['err2_v'])
                    err3_v_m = nanmean(pl['err3_v'])
                    if "_4thb" in do:
                        vaf4_v_m = nanmean(pl['vaf4_v'])
                        err4_v_m = nanmean(pl['err4_v'])
                    ly_v = pl['ly_v']

                    mx1_v_m = nanmean(pl['mx1_v'][:,:,0])
                    mx2_v_m = nanmean(pl['mx2_v'][:,:,0])
                    mx3_v_m = nanmean(pl['mx3_v'][:,:,0])

                    stdx1_v_m = nanmean(pl['mx1_v'][:,:,1])
                    stdx2_v_m = nanmean(pl['mx2_v'][:,:,1])
                    stdx3_v_m = nanmean(pl['mx3_v'][:,:,1])

                    nullp1_v_m = nanmean(pl['nullp1_v'])
                    nullp2_v_m = nanmean(pl['nullp2_v'])
                    nullp3_v_m = nanmean(pl['nullp3_v'])

                    #if export_m:
                    #    filepathtxt = export_m + "/" +prefix_plot + ".txt"
                    #    filepathtxt2 = export_m + "/" +prefix_plot + "2.txt"
                    #else:
                    #    filepathtxt = data_dir + prefix_plot + ".txt"
                    #    filepathtxt2 = data_dir + prefix_plot + "2.txt"

                    filepathtxt = "./figs/txt/" + prefix_plot + ".txt"
                    filepathtxt2 = "./figs/txt/" + prefix_plot + "2.txt"


                    fd = open(filepathtxt2, 'wb')
                    writer = csv.writer(fd, delimiter='\t')
                    writer.writerow( ('f', 'stdf') )
                    for ir in range(10):
                        for i, w in enumerate(p_v):
                            writer.writerow( (pl['f_v'][0][ir][i], pl['stdf_v'][0][ir][i]) )
                    fd.close()

                    fd = open(filepathtxt, 'wb')
                    writer = csv.writer(fd, delimiter='\t')
                    writer.writerow( ('weight','lyapunov','err1','err2','err3','m1','m2','m3','std1','std2','std3','null1','null2','null3', 'f', 'stdf') )
                    for i, w in enumerate(p_v):
                        writer.writerow( (w, ly_v[i], err1_v_m[i], err2_v_m[i], err3_v_m[i], mx1_v_m[i], mx2_v_m[i], mx3_v_m[i], stdx1_v_m[i], stdx2_v_m[i], stdx3_v_m[i], nullp1_v_m[i], nullp2_v_m[i], nullp3_v_m[i], f_v_m[i], stdf_v_m[i]) )
                    fd.close()


            if myid == 0:

                fig1 = plt.figure('vec')

                ax01b = ax01
                if "_noly" not in do: ax02b = ax02
                if "_fstd" in do: ax02b = ax02

                if "_lyline_" in do:
                    if np.nanmax(ly_v) > 0:
                        ly_cross = len(ly_v)-np.where(ly_v[::-1]<=0)[0][0]
                        ly_border = (p_v[ly_cross-1]+p_v[ly_cross])/2
                        ax01.axvline(x=ly_border*normp,color=color_ly, linestyle = linestyle)
                        if "_noly" not in do: ax02.axvline(x=ly_border*normp,color=color_ly , linestyle = linestyle)


                if (("fig1" in opt) and ("_ax3_" in do) and ("_noinv_" not in do)) or (("fig3" in opt) and ("_ax1_" in do) and ("_ifun2_" not in do)):
                    if "_onlyslow_" not in do: ax01b.plot(p_v, err1_v_m, color=colgr[0], linestyle = linestyle, linewidth=linewidth, clip_on = False)
                    if "_4thb" in do: ax01b.plot(p_v, err4_v_m, color=colgr[3], linestyle = linestyle, linewidth=linewidth, clip_on = False)
                    if "_onlyslow_" not in do: ax01b.plot(p_v, err2_v_m, color=colgr[1], linestyle = linestyle, linewidth=linewidth, clip_on = False)
                    ax01b.plot(p_v, err3_v_m, color=colgr[2], linestyle = linestyle, linewidth=linewidth, clip_on = False)

                else:
                    if "_onlyslow_" not in do: ax01b.plot(p_v*normp, err1_v_m, color=colgr[0], linestyle = linestyle, label=r"$\tau$ = 10 ms", linewidth=linewidth, clip_on = False)
                    if "_4thb" in do: ax01b.plot(p_v*normp, err4_v_m, color=colgr[3], linestyle = linestyle, label=r"$\tau$ = 50 ms", linewidth=linewidth, clip_on = False)
                    if "_onlyslow_" not in do: ax01b.plot(p_v*normp, err2_v_m, color=colgr[1], linestyle = linestyle, label=r"$\tau$ = 100 ms", linewidth=linewidth, clip_on = False)
                    ax01b.plot(p_v*normp, err3_v_m, color=colgr[2], linestyle = linestyle, label=r"$\tau$ = 500 ms", linewidth=linewidth, clip_on = False)


                ax01b.axis(xmin=xmin, xmax=xmax)


                if "_noly" not in do:

                    if "_dots" in do:
                        ax02b.plot(p_v*normp, ly_v, 'o',color=color_ly2, markersize=3, markeredgecolor = 'none')
                    else:
                        ax02b.plot(p_v*normp, ly_v, color=color_ly2, linestyle = linestyle, linewidth=linewidth)
                    ax02b.axhline(y=0, color="k", linestyle = "dashed", linewidth=0.5)
                    ax02b.axis(xmin=xmin, xmax=xmax)

                    if lymax:
                        if lymin:
                            ax02b.axis(ymin=lymin, ymax=lymax)
                        else:
                            ax02b.axis(ymin=-1, ymax=lymax)

                elif "_fstd" in do:
                    ax02b.semilogy(p_v, f_v_m, color=color_ly2, linestyle = linestyle, linewidth=linewidth)
                    ax02b.semilogy(p_v, stdf_v_m, color=p1, linestyle = linestyle, linewidth=linewidth)
                    ax02b.axis(xmin=xmin, xmax=xmax)
                    ax02b.axis(ymin=0.01, ymax=2000)

                if "_first_" in do:

                    if "_noly" not in do:

                        if ("fig2" in opt):
                            adjust_spines(ax02b, ['left'], d_out = d_out, d_down = d_down)
                        else:
                            adjust_spines(ax02b, ['left','bottom'], d_out = d_out, d_down = d_down)

                        if ("figRe6" in opt):
                            if ("_ax2" in do):
                                ax01b.set_title(r"No push-pull input")
                            else:
                                ax02b.set_ylabel("Lyapunov exp.", labelpad=6)

                        elif "fig5" in opt:
                            ax02b.set_ylabel("Lyapunov exp.", labelpad=6)
                        else:
                            ax02b.set_ylabel("Lyapunov exponent", labelpad=2)

                        if lymax == 17:
                            ax02b.yaxis.set_ticks(array([0,5,10,15]))
                            ax02b.set_yticklabels(('0','5','10','15'))
                        elif lymax == 4.2:
                            ax02b.yaxis.set_ticks(array([0,2,4]))
                            ax02b.set_yticklabels(('0','2','4'))
                        elif lymax == 7:
                            ax02b.yaxis.set_ticks(array([0,2,4,6]))
                            ax02b.set_yticklabels(('0','2','4','6'))
                        elif lymin == -3:
                            ax02b.yaxis.set_ticks(array([-2, -1, 0]))
                            ax02b.set_yticklabels(('-2','-1','0'))
                        elif lymax == 0.5:
                            ax02b.yaxis.set_ticks(array([-0.5,0,0.5]))
                            ax02b.set_yticklabels(('-0.5','0','0.5'))
                        elif lymax == 2:
                            ax02b.yaxis.set_ticks(array([0,1,2]))
                            ax02b.set_yticklabels(('0','1','2'))
                        elif lymax == 3:
                            ax02b.yaxis.set_ticks(array([0,1,2,3]))
                            ax02b.set_yticklabels(('0','1','2','3'))


                        if ("_horly" in do) or ("fig2" in opt):
                            adjust_spines(ax01b, ['left','bottom'], d_out = d_out, d_down = d_down)
                        else:
                            adjust_spines(ax01b, ['left'], d_out = d_out, d_down = d_down)

                    elif "_fstd" in do:
                        adjust_spines(ax01b, ['left'], d_out = d_out, d_down = d_down)
                        adjust_spines(ax02b, ['left','bottom'], d_out = d_out, d_down = d_down)

                        ax02b.set_ylabel(r"mean($\mathsf{z_i}$), std($\mathsf{z_i}$)", labelpad=0)

                        ax02b.yaxis.set_ticks(array([0.01,0.1,1,10,100,1000]))
                        ax02b.set_yticklabels(('0.01','0.1','1','10','100','1000'))

                    else:

                        if ("fig3b" in opt):
                            adjust_spines(ax01b, ['left'], d_out = d_out, d_down = d_down)
                        else:
                            adjust_spines(ax01b, ['left','bottom'], d_out = d_out, d_down = d_down)

                    if ("figRe6" in opt) and ("_ax2" in do):
                        pass
                    else:
                        ax01b.set_ylabel(r"goodness of fit $\mathsf{(R^{2})}$", labelpad=2)

                    ax01b.yaxis.set_ticks(array([0,0.2,0.4,0.6,0.8,1]))
                    ax01b.set_yticklabels(('0','0.2','0.4','0.6','0.8','1'))

                else:

                    if ("fig2" in opt):
                        adjust_spines(ax02b, [], d_out = d_out, d_down = d_down)
                        adjust_spines(ax01b, ['bottom'], d_out = d_out, d_down = d_down)

                    elif ("_noly" not in do) or ("_fstd" in do):
                        adjust_spines(ax01b, [], d_out = d_out, d_down = d_down)
                        adjust_spines(ax02b, ['bottom'], d_out = d_out, d_down = d_down)

                    else:
                        adjust_spines(ax01b, ['bottom'], d_out = d_out, d_down = d_down)


                if ("_noly" not in do) or ("_fstd" in do):

                    if ("fig2" in opt):
                        ax_ticks = ax01b
                    else:
                        ax_ticks = ax02b

                    if "_vec3w_" in do:
                        ax_ticks.set_xlabel("b", labelpad=2)
                    else:
                        if "_normp_" in do:
                            ax_ticks.set_xlabel("w*a", labelpad=0)
                        else:
                            ax_ticks.set_xlabel("w", labelpad=0)

                    if "_normp_" in do:
                        ax_ticks.xaxis.set_ticks(array([0,0.2,0.4,0.6,0.8,1]))
                        ax_ticks.set_xticklabels(('0','0.2','0.4','0.6','0.8','1'))
                    elif xmax==4:
                        ax_ticks.xaxis.set_ticks(array([0,1,2,3,4]))
                        ax_ticks.set_xticklabels(('0','1','2','3','4'))
                    elif xmax==3:
                        ax_ticks.xaxis.set_ticks(array([0,1,2]))
                        ax_ticks.set_xticklabels(('0','1','2'))
                    elif xmax==40:
                        ax_ticks.xaxis.set_ticks(array([0,10,20,30,40]))
                        ax_ticks.set_xticklabels(('0','10','20','30','40'))
                    elif xmax==0.02:
                        ax_ticks.xaxis.set_ticks(array([0,0.005,0.01,0.015,0.02]))
                        ax_ticks.set_xticklabels(('0','','0.01','','0.02'))
                    elif xmax==0.01:
                        ax_ticks.xaxis.set_ticks(array([0,0.005,0.01]))
                        ax_ticks.set_xticklabels(('0','0.005','0.01'))
                    elif xmax==0.03:
                        ax_ticks.xaxis.set_ticks(array([0,0.01,0.02,0.03]))
                        ax_ticks.set_xticklabels(('0','0.01','0.02','0.03'))

                elif "fig3b" not in opt:

                    if ("fig2" in opt):
                        ax_ticks = ax01b
                    else:
                        ax_ticks = ax01b

                    ax_ticks.set_xlabel("w", labelpad=0)

                    if "_normp_" in do:
                        pass
                    elif xmax==4:
                        ax_ticks.xaxis.set_ticks(array([0,1,2,3,4]))
                        ax_ticks.set_xticklabels(('0','1','2','3','4'))
                    elif xmax==3:
                        ax_ticks.xaxis.set_ticks(array([0,1,2]))
                        ax_ticks.set_xticklabels(('0','1','2'))
                    elif xmax==2:
                        ax_ticks.xaxis.set_ticks(array([0,1,2]))
                        ax_ticks.set_xticklabels(('0','1','2'))
                    elif xmax==40:
                        ax_ticks.xaxis.set_ticks(array([0,10,20,30,40]))
                        ax_ticks.set_xticklabels(('0','10','20','30','40'))
                    elif xmax==14:
                        ax_ticks.xaxis.set_ticks(array([0,2,4,6,8,10,12,14]))
                        ax_ticks.set_xticklabels(('0','2','4','6','8','10','12','14'))
                    elif xmax==80:
                        ax_ticks.xaxis.set_ticks(array([0,20,40,60,80]))
                        ax_ticks.set_xticklabels(('0','20','40','60','80'))
                    elif xmax==0.05:
                        ax_ticks.xaxis.set_ticks(array([0,0.01,0.02,0.03,0.04,0.05]))
                        ax_ticks.set_xticklabels(('0','','0.02','','0.04',''))
                    elif xmax==0.5:
                        ax_ticks.xaxis.set_ticks(array([0,0.1,0.2,0.3,0.4,0.5]))
                        ax_ticks.set_xticklabels(('0','','0.2','','0.4',''))
                    elif xmax==0.005:
                        ax_ticks.xaxis.set_ticks(array([0,0.001,0.002,0.003,0.004,0.005]))
                        ax_ticks.set_xticklabels(('0','','0.002','','0.004',''))
                    elif xmax==0.02:
                        ax_ticks.xaxis.set_ticks(array([0,0.005,0.01,0.015,0.02]))
                        ax_ticks.set_xticklabels(('0','','0.01','','0.02'))
                    elif xmax==0.03:
                        ax_ticks.xaxis.set_ticks(array([0,0.01,0.02,0.03]))
                        ax_ticks.set_xticklabels(('0','0.01','0.02','0.03'))

                ax01b.axis(ymin=-0.02, ymax=1.02)

                if "fig2" in opt:

                    ax02b.set_title(r"$\mathsf{\tau_{w}}$ = " + str(int(conntype[0]['tau1']*1e3)) + " ms")

                    if "_ax3_" in do:

                        idx = np.where(p_v==0.5)[0][0]
                        c1 = ax01b.plot((0.5),(err1_v_m[idx]*100),'o',color=b1, markersize=5, clip_on = False)
                        c2 = ax01b.plot((0.5),(err2_v_m[idx]*100),'o',color=g1, markersize=5, clip_on = False)
                        c3 = ax01b.plot((0.5),(err3_v_m[idx]*100),'o',color=r1, markersize=5, clip_on = False)

                        idx = np.where(p_v==2)[0][0]
                        c4 = ax01b.plot((2),(err1_v_m[idx]*100),'o',color=b2, markersize=5, clip_on = False)
                        c5 = ax01b.plot((2),(err2_v_m[idx]*100),'o',color=g2, markersize=5, clip_on = False)
                        c6 = ax01b.plot((2),(err3_v_m[idx]*100),'o',color=r2, markersize=5, clip_on = False)


                if "fig3" in opt:
                    #ax01b.set_title(r"$\mathsf{\tau_{w}}$ = " + str(int(conntype[0]['tau1']*1e3)) + " ms")

                    #if ("_ax3_" in do) and ("_noinv_" in do):
                    #
                    #    color_v.append(colgr)
                    #
                    #    lg = ax01b.legend(labelspacing=0.05, bbox_to_anchor=(0.45,0.5), loc=3, handlelength=0, handletextpad=0.1, numpoints=1)
                    #    #lg.draw_frame(False)
                    #    fr = lg.get_frame()
                    #    fr.set_lw(0.2)
                    #
                    #    txt = lg.get_texts()
                    #    color_v0 = [val for subl in color_v for val in subl]
                    #    for k, t in enumerate(txt):
                    #        t.set_color(color_v0[k])

                    if ("_colgrlight_" in do):

                        if "fig3b" in opt:
                            if ("_ax2_" in do): ax01b.set_title(r"No push-pull input")
                            if ("_ax3_" in do): ax01b.set_title("No push-pull input, positive lasso")
                        else:
                            #if ("_ax1_" in do): ax01b.set_title(r"No push-pull input")
                            if ("_ax1_" in do): ax01b.set_title(r"Additive white noise (n=0.01)")
                            if ("_ax2_" in do): ax01b.set_title(r"Increased variability")
                            if ("_ax3_" in do): ax01b.set_title(r"Increased sparseness (a=0.04)")
                            #if ("_ax2_" in do): ax01b.text(0.4, 1.3, 'Effect of modulation amplitude (a= 0.1 to 0.01) and push-pull input', transform=ax01b.transAxes, fontsize = 10, va='center', ha='center')
                            #if ("_ax5_" in do): ax01b.text(0.5, 1.3, 'Effect of increased input variability (v= 0.1 to 0.5)', transform=ax01b.transAxes, fontsize = 10, va='center', ha='center')
                            #if ("_ax8_" in do): ax01b.text(0.5, 1.3, 'Effect of sparseness (decreased connectivity p= 0.4 to 0.04)', transform=ax01b.transAxes, fontsize = 10, va='center', ha='center')


                if "fig4" in opt:
                    if ("_title1_" in do): ax01b.set_title(r"$\mathsf{\tau_{w}}$=" + str(int(conntype[0]['tau1']*1e3)) + "ms\n" + r"$\mathsf{\tau_{u}}$=" + str(int(conntype[1]['tau1']*1e3)) + "ms")
                    if ("_title2_" in do): ax01b.set_title(r"Increased sparseness and size")
                    if ("_title3_" in do): ax01b.set_title(r"Increased weight variability")

                if "fig5" in opt:
                    #if ("_title4_" in do): ax01b.set_title(r"u=0.1")
                    if ("_title5_" in do): ax01b.set_title(r"Increased excitation u=50")
                    if ("_title6_" in do): ax01b.set_title(r"No push-pull input")

                    if ("_ax1_" in do) and ("_ifun2_" in do):

                        pass
                        #color_v.append(colgr)

                        #lg = ax01b.legend(labelspacing=0.05, bbox_to_anchor=(0.5,0.7), loc=3, handlelength=0, handletextpad=0.1, numpoints=1)
                        ##lg.draw_frame(False)
                        #fr = lg.get_frame()
                        #fr.set_lw(0.2)

                        #txt = lg.get_texts()
                        #color_v0 = [val for subl in color_v for val in subl]
                        #for k, t in enumerate(txt):
                        #    t.set_color(color_v0[k])


                filename="./figs/Pub/" + opt + "_" + str(prefix_plot)
                if use_pc is False: plt.savefig(filename  + ".pdf", dpi = 300) # save it
                plt.savefig(filename + imgf, dpi = 300) # save it

            if do_run == 1:
                barrier(use_mpi = use_mpi, use_pc = use_pc)


        if "_fmin_" in do:

            p = fmin(residuals, p0, args=(params))
            print p
            err = 0
            save_results(pickle_prefix,p,err,params, use_h5 = use_h5, data_dir=data_dir)

        if "_brute_" in do:

            p = brute(residuals, ranges, args=(params), Ns=10)
            print p
            err = 0

        if "_mybrute_" in do:

            if fit_restart is False:
                if myid == 0:
                    params = pickle.load( gzip.GzipFile( filepath, "rb" ) )

            params = broadcast(params)

            for i in myranges[0]:
                for j in myranges[1]:
                    if [i,j] in params['fit_para']:
                        err = params['fit_err'][params['fit_para'].index([i,j])]
                        if myid == 0: print "Already computed for", [i,j], ", err:", err
                    else:
                        p0 = [i,j]
                        err = residuals(p0, params)
                        if myid == 0: print "Newly computed for", [i,j], ", err:", err



        if "_anneal_" in do:

            p = anneal(residuals, p0, args=(params), upper=upper_bounds, lower=lower_bounds)
            print p
            err = 0
            save_results(pickle_prefix,p,err, use_h5 = use_h5, data_dir=data_dir)


        if "_basinhopping_" in do:

            p = basinhopping(residuals, p0, minimizer_kwargs={"args":params})
            print p
            err = 0
            save_results(pickle_prefix,p,err, use_h5 = use_h5, data_dir=data_dir)


if ((opt == "fig1") or ("fig4" in opt) or ("fig5" in opt) or ("figRe6" in opt)):

    from pyPdf import PdfFileReader, PdfFileWriter

    output = PdfFileWriter()
    pdfOne = PdfFileReader(file(filename + ".pdf", "rb"))
    input1 = pdfOne.getPage(0)

    if "fig1" in opt:
        pdfTwo = PdfFileReader(file("./figs/Inserts/cl01.pdf", "rb"))
        input1.mergeTransformedPage(pdfTwo.getPage(0),[1, 0, 0, 1, 0, 357])

    elif "fig51" in opt:
        pdfTwo = PdfFileReader(file("./figs/Inserts/cl03.pdf", "rb"))
        input1.mergeTransformedPage(pdfTwo.getPage(0),[1, 0, 0, 1, 0, 185])

    elif "fig52" in opt:
        pdfTwo = PdfFileReader(file("./figs/Inserts/cl04.pdf", "rb"))
        input1.mergeTransformedPage(pdfTwo.getPage(0),[1, 0, 0, 1, 0, 182])

    elif "fig5" in opt:
        pdfTwo = PdfFileReader(file("./figs/Inserts/cl03.pdf", "rb"))
        input1.mergeTransformedPage(pdfTwo.getPage(0),[1, 0, 0, 1, 0, 455])
        pdfTwo = PdfFileReader(file("./figs/Inserts/cl04.pdf", "rb"))
        input1.mergeTransformedPage(pdfTwo.getPage(0),[1, 0, 0, 1, 0, 180])

    elif "fig4" in opt:
        pdfTwo = PdfFileReader(file("./figs/Inserts/cl02.pdf", "rb"))
        input1.mergeTransformedPage(pdfTwo.getPage(0),[1, 0, 0, 1, 0, 348])

    elif "figRe6" in opt:
        pdfTwo = PdfFileReader(file("./figs/Inserts/cl05.pdf", "rb"))
        input1.mergeTransformedPage(pdfTwo.getPage(0),[1, 0, 0, 1, 0, 179])



    output.addPage(input1)
    outputStream = file(filename + "_m.pdf", "wb")
    output.write(outputStream)
    outputStream.close()


plt.show()
