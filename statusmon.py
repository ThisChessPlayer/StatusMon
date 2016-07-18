'''*-----------------------------------------------------------------------*---
                                                         Author: Jason Ma
                                                         Date  : Jul 17 2016
   File Name  : statusmon.py
   Description: 
---*-----------------------------------------------------------------------*'''
import matplotlib
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
  
  #create figure with 16:8 (width:height) ratio
  fig = plt.figure(figsize = (16, 8), dpi = 100)
  fig.canvas.set_window_title("Cubeception 3 Status Monitor")
  fig.suptitle('Cubeception 3 Status Monitor')
  
  #create subplots on a 2 row 4 column grid
  ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan = 2, colspan = 2, polar = True)
  ax2 = plt.subplot2grid((2, 4), (0, 2), rowspan = 1, colspan = 1)
  ax3 = plt.subplot2grid((2, 4), (0, 3), rowspan = 1, colspan = 1)
  ax4 = plt.subplot2grid((2, 4), (1, 2), rowspan = 1, colspan = 1)
  ax5 = plt.subplot2grid((2, 4), (1, 3), rowspan = 1, colspan = 1)
  plt.tight_layout(pad = 2)

  '''[Polar Targets]--------------------------------------------------------'''
  #set label locations appropriately
  ax1.set_theta_zero_location("N")
  ax1.set_theta_direction(-1)
  
  #set color of border
  ax1.spines['polar'].set_color(LIGHT_GREEN)

  #format and set color of ticks and labels
  ax1.set_thetagrids(np.linspace(-180, 180, 36, endpoint = False), 
                     color = LIGHT_GREEN, fontsize = 8, frac = 1.05)
  
  #TODO in animation, change this dynamically based on targets
  ax1.set_yticklabels([10, 20, 30, 40], color = LIGHT_GREEN, fontsize = 8)
  ax1.set_rlabel_position(90)

  #set gridline colors to green/darkgreen
  for line in ax1.get_xgridlines():
    line.set_color(DARK_GREEN)
    line.set_linestyle('dotted')

  for line in ax1.get_ygridlines():
    line.set_color(LIGHT_GREEN)
    line.set_linestyle('dotted')

  #TODO figure out why this code wipes half of the x axis lines
  '''
  li = np.zeros((5,2))
  li[0] = [.2, 1]
  li[1] = [0.4, 1]
  [[0.2, 1], [0.4, 1], [0.6, 1], [0.8, 1], [-2, 1]]

  for x,y in li:
    #TODO put in animate section
    #TODO modify size by confidence
    #TODO change color scheme a bit
    point = ax1.plot(x, y, marker='o', color='r', markersize=10)
    ax1.text(x, y, 'CV Target', bbox = dict(facecolor=DARK_GREEN, alpha=0.5), fontsize = 8)
    #annotation = ax1.annotate("Computer Vision Target",
    #    xy=(x,y), xycoords='data',
    #    xytext=(x,y), textcoords='data', color='#ff0000', backgroundcolor='#ffaaaa'
  #)
  '''
  
  '''[Position]-------------------------------------------------------------'''
  #set color of spines
  ax2.spines['bottom'].set_color(LIGHT_GREEN)
  ax2.spines['top'].set_color(LIGHT_GREEN)
  ax2.spines['right'].set_color(LIGHT_GREEN)
  ax2.spines['left'].set_color(LIGHT_GREEN)

  #set color of ticks
  ax2.tick_params(axis='x', colors = LIGHT_GREEN)
  ax2.tick_params(axis='y', colors = LIGHT_GREEN)

  #enable and set color of grid
  ax2.grid(True)
  for line in ax2.get_xgridlines():
    line.set_color(DARK_GREEN)
    line.set_linestyle('dashed')

  for line in ax2.get_ygridlines():
    line.set_color(DARK_GREEN)
    line.set_linestyle('dashed')

  #set color of tick labels
  for label in ax2.get_xticklabels():
    label.set_fontsize(8)

  for label in ax2.get_yticklabels():
    label.set_fontsize(8)

  '''[Velocity]-------------------------------------------------------------'''
  #set color of spines
  ax3.spines['bottom'].set_color(LIGHT_GREEN)
  ax3.spines['top'].set_color(LIGHT_GREEN)
  ax3.spines['right'].set_color(LIGHT_GREEN)
  ax3.spines['left'].set_color(LIGHT_GREEN)

  #set color of ticks
  ax3.tick_params(axis='x', colors = LIGHT_GREEN)
  ax3.tick_params(axis='y', colors = LIGHT_GREEN)

  #enable and set color of grid
  ax3.grid(True)
  for line in ax3.get_xgridlines():
    line.set_color(DARK_GREEN)
    line.set_linestyle('dashed')

  for line in ax3.get_ygridlines():
    line.set_color(DARK_GREEN)
    line.set_linestyle('dashed')

  #set color of tick labels
  for label in ax3.get_xticklabels():
    label.set_fontsize(8)

  for label in ax3.get_yticklabels():
    label.set_fontsize(8)

  '''[Thruster Status]------------------------------------------------------'''
  #set color of spines
  ax4.spines['bottom'].set_color(LIGHT_GREEN)
  ax4.spines['top'].set_color(LIGHT_GREEN)
  ax4.spines['right'].set_color(LIGHT_GREEN)
  ax4.spines['left'].set_color(LIGHT_GREEN)

  #set color of ticks
  ax4.tick_params(axis='x', colors = LIGHT_GREEN)
  ax4.tick_params(axis='y', colors = LIGHT_GREEN)

  #enable and set color of grid
  ax4.grid(True)
  for line in ax4.get_xgridlines():
    line.set_color(DARK_GREEN)
    line.set_linestyle('dashed')

  for line in ax4.get_ygridlines():
    line.set_color(DARK_GREEN)
    line.set_linestyle('dashed')

  #set color of tick labels
  for label in ax4.get_xticklabels():
    label.set_fontsize(8)

  for label in ax4.get_yticklabels():
    label.set_fontsize(8)

  '''[Robot Model Rotation]-------------------------------------------------'''
  #set color of spines
  ax5.spines['bottom'].set_color(LIGHT_GREEN)
  ax5.spines['top'].set_color(LIGHT_GREEN)
  ax5.spines['right'].set_color(LIGHT_GREEN)
  ax5.spines['left'].set_color(LIGHT_GREEN)

  #set color of ticks
  ax5.tick_params(axis='x', colors = LIGHT_GREEN)
  ax5.tick_params(axis='y', colors = LIGHT_GREEN)

  #enable and set color of grid
  ax5.grid(True)
  for line in ax5.get_xgridlines():
    line.set_color(DARK_GREEN)
    line.set_linestyle('dashed')

  for line in ax5.get_ygridlines():
    line.set_color(DARK_GREEN)
    line.set_linestyle('dashed')

  #set color of tick labels
  for label in ax5.get_xticklabels():
    label.set_fontsize(8)

  for label in ax5.get_yticklabels():
    label.set_fontsize(8)

  return fig

fig = initPlot()
plt.show()
