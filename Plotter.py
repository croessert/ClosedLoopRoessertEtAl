# -*- coding: utf-8 -*-
"""
@author: chris
"""

from __future__ import division

from pylab import *

import os

from Stimhelp import *

def adjust_spines(ax, spines, color = 'k', d_out = 10, d_down = []):
    
    if d_down == []:
        d_down = d_out
        
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    
    for loc, spine in ax.spines.iteritems():
        if loc in spines:
            if loc == 'bottom': 
                spine.set_position(('outward',d_down)) # outward by 10 points
            else:
                spine.set_position(('outward',d_out)) # outward by 10 points
        else:
            spine.set_visible(False) # set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
        
        if color is not 'k':
            
            ax.spines['left'].set_color(color)
            ax.yaxis.label.set_color(color)
            ax.tick_params(axis='y', colors=color)
        
        
    elif 'right' not in spines:
        # no yaxis ticks
        ax.yaxis.set_ticks([])
        
    if 'right' in spines:
        ax.yaxis.set_ticks_position('right')
        
        if color is not 'k':
            
            ax.spines['right'].set_color(color)
            ax.yaxis.label.set_color(color)
            ax.tick_params(axis='y', colors=color)
               
    if 'bottom' in spines:
        pass
        ax.xaxis.set_ticks_position('bottom')
        
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])
        ax.axes.get_xaxis().set_visible(False)

