'''*-----------------------------------------------------------------------*---
                                                         Author: Jason Ma
                                                         Date  : Jul 17 2016
   File Name  : statusmon.py
   Description: 
---*-----------------------------------------------------------------------*'''
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import matplotlib.gridspec as gridspec
import numpy as np
#import sys
#import time
#import re

NUM_TARGETS = 2

FIG_WIDTH   = 16
FIG_HEIGHT  = 8
FIG_NAME    = 'Cubeception 3 Status Monitor'
PLOT_STYLE  = 'dark_background'
LIGHT_GREEN = '#33ff3c'
DARK_GREEN  = '#007505'

LIGHT_RED   = '#ffaaaa'
DARK_RED    = '#ff0000'

'''initGlobals-----------------------------------------------------------------
Generates figure and subplots, sets basic layout.
----------------------------------------------------------------------------'''
plt.style.use(PLOT_STYLE)

#create figure with 16:8 (width:height) ratio
fig = plt.figure(figsize = (FIG_WIDTH, FIG_HEIGHT), dpi = 100)
fig.canvas.set_window_title(FIG_NAME)
fig.suptitle(FIG_NAME)
  
#create subplots on a 2 row 4 column grid
ax1 = plt.subplot2grid((2, 4), (0, 0), rowspan = 2, colspan = 2, polar = True)
ax2 = plt.subplot2grid((2, 4), (0, 2), rowspan = 1, colspan = 1)
ax3 = plt.subplot2grid((2, 4), (0, 3), rowspan = 1, colspan = 1)
ax4 = plt.subplot2grid((2, 4), (1, 2), rowspan = 1, colspan = 1)
ax5 = plt.subplot2grid((2, 4), (1, 3), rowspan = 1, colspan = 1)
plt.tight_layout(pad = 2)

'''initPlot--------------------------------------------------------------------
Sets up subplots and starting image of figure to display

plotStyle - color scheme for background
----------------------------------------------------------------------------'''
def initFigure():

  '''[Polar Targets]--------------------------------------------------------'''
  #set label locations appropriately
  ax1.set_theta_zero_location("N")
  ax1.set_theta_direction(-1)
  
  #set color of border
  ax1.spines['polar'].set_color(LIGHT_GREEN)

  #format and set color of ticks and labels
  ax1.set_thetagrids(np.linspace(0, 360, 36, endpoint = False), 
                     color = LIGHT_GREEN, fontsize = 8, frac = 1.05)
  
  #TODO in animation, change this dynamically based on targets
  ax1.set_yticklabels([2.5, 5, 7.5, 10], color = LIGHT_GREEN, fontsize = 8)
  ax1.set_rlabel_position(90)

  #set gridline colors to green/darkgreen
  for line in ax1.get_xgridlines():
    line.set_color(DARK_GREEN)
    line.set_linestyle('dotted')

  for line in ax1.get_ygridlines():
    line.set_color(LIGHT_GREEN)
    line.set_linestyle('dotted')
 
  '''[Position/Velocity/Acceleration]---------------------------------------'''
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

  #set size of tick labels
  for label in ax2.get_xticklabels():
    label.set_fontsize(8)

  for label in ax2.get_yticklabels():
    label.set_fontsize(8)

  '''[Thruster Heatmap]-----------------------------------------------------'''
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

  #set size of tick labels
  for label in ax3.get_xticklabels():
    label.set_fontsize(8)

  for label in ax3.get_yticklabels():
    label.set_fontsize(8)

  '''[Robot Rotations]------------------------------------------------------'''
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

  #set size of tick labels
  for label in ax4.get_xticklabels():
    label.set_fontsize(8)

  for label in ax4.get_yticklabels():
    label.set_fontsize(8)

  '''[Status]---------------------------------------------------------------'''
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

  #set size of tick labels
  for label in ax5.get_xticklabels():
    label.set_fontsize(8)

  for label in ax5.get_yticklabels():
    label.set_fontsize(8)

'''
def animate(i):
  #TODO get all the data:
  computer vision forward target
  sonar target
  
  control goal
  x,y,z position
  x,y,z velocity
  x,y,z acceleration

  motor info
  
  accelerometer data
  gyro data
  magnetometer data
  pressure sensor data

  statuses of everything
  

  #TODO make rotating line on polar plot
  #fig.ax1.

  li = np.zeros((2,3))
  li[0][0] = 3
  li[0][1] = 5
  li[0][2] = -4

  li[1][0] = 0
  li[1][1] = 0
  li[1][2] = -6

  #TODO modify size by confidence
  #TODO change color scheme based on depth
  #point = ax1.plot(x, y, marker='o', color='r', markersize=10)
  for j in range(0, NUM_TARGETS):
    fig.ax1.plot(li[j][0], li[j][1], marker = 'o', c = DARK_RED, markersize = 10)
  
  #CV forward text
  fig.ax1.text(li[0][0], li[0][1], 
           'CV Forward x:{0:5.3f} y:{1:5.3f} z:{2:5.3f}'.format(li[0][0],
                                                                li[0][1],
                                                                li[0][2]), 
           bbox = dict(facecolor = DARK_GREEN, alpha = 0.3), fontsize = 8)
  
  #CV down text
  fig.ax1.text(li[0][0], li[0][1], 
           'CV Down x:{0:5.3f} y:{1:5.3f} z:{2:5.3f}'.format(li[1][0],
                                                             li[1][1],
                                                             li[1][2]), 
           bbox = dict(facecolor = DARK_GREEN, alpha = 0.3), fontsize = 8)


  #Sonar target text
  #ax1.text(li[0][0], li[0][1], 
  #         'CV Target x:{0:5.3f} y:{1:5.3f} z:{2:5.3f}'.format(li[0][0]
  #                                                             li[0][1]
  #                                                             li[0][2]), 
  #         bbox = dict(facecolor = DARK_GREEN, alpha = 0.5), fontsize = 8)

'''
  

initFigure()

#ani = animation.FuncAnimation(fig, animate, interval = 5)
plt.show()
