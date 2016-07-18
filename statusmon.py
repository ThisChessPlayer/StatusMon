'''*-----------------------------------------------------------------------*---
                                                         Author: Jason Ma
                                                         Date  : Jul 17 2016
   File Name  : statusmon.py
   Description: 
---*-----------------------------------------------------------------------*'''

import matplotlib.pyplot as plt
#TODO import matplotlib.animation as animation
#import matplotlib.gridspec as gridspec
import numpy as np
#import sys
#import time
#import re
LIGHT_GREEN = '#33ff3c'
DARK_GREEN = '#007505'

'''initPlot--------------------------------------------------------------------
Generates subplots and starting image of figure to display

[return] - figure with initialized subplots and colors
----------------------------------------------------------------------------'''
def initPlot():
  plt.style.use('dark_background')
  plt.rc('grid', linewidth = 1, linestyle = '-')
  
  #create figure with 16:8 (width:height) ratio
  fig = plt.figure(figsize = (16, 8), dpi = 100)

  #create subplots on a 2 row 4 column grid
  ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan = 2, colspan = 2, polar = True)
  ax2 = plt.subplot2grid((2, 4), (0, 2), rowspan = 1, colspan = 1)
  ax3 = plt.subplot2grid((2, 4), (0, 3), rowspan = 1, colspan = 1)
  ax4 = plt.subplot2grid((2, 4), (1, 2), rowspan = 1, colspan = 1)
  ax5 = plt.subplot2grid((2, 4), (1, 3), rowspan = 1, colspan = 1)
  plt.tight_layout()

  '''[Polar Targets]--------------------------------------------------------'''
  #set labels appropriately
  ax1.set_theta_zero_location("N")
  ax1.set_theta_direction(-1)
  ax1XLabels = np.pi/180. * np.linspace(-180, 180, 36, endpoint = False)
  ax1.set_xticks(ax1XLabels)

  ax1.set_yticklabels([])
  
  #set color of border
  ax1.spines['polar'].set_color(LIGHT_GREEN)

  #set color of ticks
  ax1.tick_params(axis = 'x', direction = 'out', colors = LIGHT_GREEN)

  #set gridline colors to green/darkgreen
  for line in ax1.get_xgridlines():
    line.set_color(DARK_GREEN)
    line.set_linestyle('dotted')

  for line in ax1.get_ygridlines():
    line.set_color(LIGHT_GREEN)
    line.set_linestyle('dotted')

  '''[Position]-------------------------------------------------------------'''
  ax2.spines['bottom'].set_color(LIGHT_GREEN)
  ax2.spines['top'].set_color(LIGHT_GREEN)
  ax2.spines['right'].set_color(LIGHT_GREEN)
  ax2.spines['left'].set_color(LIGHT_GREEN)
  
  ax2.tick_params(axis='x', colors = LIGHT_GREEN)
  ax2.tick_params(axis='y', colors = LIGHT_GREEN)

  ax2.grid(True)
  for line in ax2.get_xgridlines():
    line.set_color(DARK_GREEN)
    line.set_linestyle('dashed')

  for line in ax2.get_ygridlines():
    line.set_color(DARK_GREEN)
    line.set_linestyle('dashed')

  '''[Velocity]-------------------------------------------------------------'''
  ax3.spines['bottom'].set_color(LIGHT_GREEN)
  ax3.spines['top'].set_color(LIGHT_GREEN)
  ax3.spines['right'].set_color(LIGHT_GREEN)
  ax3.spines['left'].set_color(LIGHT_GREEN)

  ax3.tick_params(axis='x', colors = LIGHT_GREEN)
  ax3.tick_params(axis='y', colors = LIGHT_GREEN)
  
  ax3.grid(True)
  for line in ax3.get_xgridlines():
    line.set_color(DARK_GREEN)
    line.set_linestyle('dashed')

  for line in ax3.get_ygridlines():
    line.set_color(DARK_GREEN)
    line.set_linestyle('dashed')

  '''[Thruster Status]------------------------------------------------------'''
  ax4.spines['bottom'].set_color(LIGHT_GREEN)
  ax4.spines['top'].set_color(LIGHT_GREEN)
  ax4.spines['right'].set_color(LIGHT_GREEN)
  ax4.spines['left'].set_color(LIGHT_GREEN)

  ax4.tick_params(axis='x', colors = LIGHT_GREEN)
  ax4.tick_params(axis='y', colors = LIGHT_GREEN)

  ax4.grid(True)
  for line in ax4.get_xgridlines():
    line.set_color(DARK_GREEN)
    line.set_linestyle('dashed')

  for line in ax4.get_ygridlines():
    line.set_color(DARK_GREEN)
    line.set_linestyle('dashed')

  '''[Robot Model Rotation]-------------------------------------------------'''
  ax5.spines['bottom'].set_color(LIGHT_GREEN)
  ax5.spines['top'].set_color(LIGHT_GREEN)
  ax5.spines['right'].set_color(LIGHT_GREEN)
  ax5.spines['left'].set_color(LIGHT_GREEN)

  ax5.tick_params(axis='x', colors = LIGHT_GREEN)
  ax5.tick_params(axis='y', colors = LIGHT_GREEN)

  ax5.grid(True)
  for line in ax5.get_xgridlines():
    line.set_color(DARK_GREEN)
    line.set_linestyle('dashed')

  for line in ax5.get_ygridlines():
    line.set_color(DARK_GREEN)
    line.set_linestyle('dashed')

  return fig

fig = initPlot()
plt.show()
