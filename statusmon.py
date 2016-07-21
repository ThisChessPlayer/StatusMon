'''*-----------------------------------------------------------------------*---
                                                         Author: Jason Ma
                                                         Date  : Jul 17 2016
   File Name  : statusmon.py
   Description: 
---*-----------------------------------------------------------------------*'''
import matplotlib as mpl
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

mpl.rc(('text', 'xtick', 'ytick'), color = LIGHT_GREEN)
mpl.rc(('lines', 'grid'), color = DARK_GREEN)
mpl.rc('axes', edgecolor = LIGHT_GREEN, titlesize = 10)
mpl.rc('font', size = 8)
mpl.rc('grid', linestyle = ':')

#create figure with 16:8 (width:height) ratio
fig = plt.figure(figsize = (FIG_WIDTH, FIG_HEIGHT), dpi = 100)
fig.canvas.set_window_title(FIG_NAME)
#fig.suptitle(FIG_NAME)
  
#create subplots on a 4 row 8 column grid
ax1 = plt.subplot2grid((4, 8), (0, 0), rowspan = 4, colspan = 4, polar = True)
ax2 = plt.subplot2grid((4, 8), (0, 4), rowspan = 2, colspan = 2)
ax3 = plt.subplot2grid((4, 8), (0, 6), rowspan = 2, colspan = 2)
ax4 = plt.subplot2grid((4, 8), (2, 4), rowspan = 1, colspan = 2)
ax5 = plt.subplot2grid((4, 8), (3, 4), rowspan = 1, colspan = 2)
ax6 = plt.subplot2grid((4, 8), (2, 6), rowspan = 2, colspan = 2)
plt.tight_layout(pad = 2)

#most recent data
data = np.zeros((2,4))
data[0][0] = 3
data[0][1] = 5
data[0][2] = -4
data[0][3] = 128

data[1][0] = 0
data[1][1] = 0
data[1][2] = -6
data[1][3] = 0

#3 for pos, 4 for vel, 4 for accel, log 50 values
dataHist = np.zeros((11, 50))

'''initPlot--------------------------------------------------------------------
Sets up subplots and starting image of figure to display
----------------------------------------------------------------------------'''
def initFigure():

  '''[Polar Targets]--------------------------------------------------------'''
  #set title
  ax1.set_title('Targets')
  
  #set label locations appropriately
  ax1.set_theta_zero_location("N")
  ax1.set_theta_direction(-1)
  
  #format ticks and labels
  ax1.set_thetagrids(np.linspace(0, 360, 36, endpoint = False), frac = 1.05)
  ax1.set_rlabel_position(90)

  #make ygridlines more visible (circular lines)
  for line in ax1.get_ygridlines():
    line.set_color(LIGHT_GREEN)
 
  '''[Position/Velocity/Acceleration]---------------------------------------'''
  #set title
  ax2.set_title('Movement')

  #enable grid
  ax2.grid(True)

  '''[Thruster Heatmap]-----------------------------------------------------'''
  #set title
  ax3.set_title('Thruster Heatmap')
 
  #enable grid
  ax3.grid(True)

  '''[XY Orientation]-------------------------------------------------------'''
  #set title
  ax4.set_title('XY Orientation')

  '''[YZ Orientation]-------------------------------------------------------'''
  #set title
  ax5.set_title('YZ Orientation')

  '''[Status]---------------------------------------------------------------'''
  #set title
  ax6.set_title('Status')

  for ax in ax4, ax5, ax6:
    ax.tick_params(axis = 'both', which = 'both', bottom = 'off', top = 'off',
           left = 'off', right = 'off', labelbottom = 'off', labelleft = 'off')


  #return ax1, ax2, ax3, ax4, ax5, ax6
'''animate---------------------------------------------------------------------
Updates subplots of figure
----------------------------------------------------------------------------'''
def animate(i):
  
  ax1.clear()
  #ax2.clear()
  #ax3.clear()
  #ax4.clear()
  #ax5.clear()
  initFigure()
  
  #TODO get all the data:
  #computer vision forward target
  #sonar target
  
  #control goal
  #x,y,z position
  #x,y,z velocity
  #x,y,z acceleration

  #motor info
  
  #accelerometer data
  #gyro data
  #magnetometer data
  #pressure sensor data

  #statuses of everything
  
  data[0][0] += 0.05
  data[0][1] = data[0][1] + 0.1 % 5

  data[1][0] += 0.05
  data[1][1] = data[1][1] + 0.3 % 5

  #ax1.hold(False)

  #TODO modify size by confidence
  #TODO change color scheme based on depth
  #point = ax1.plot(x, y, marker='o', color='r', markersize=10)
  for j in range(0, NUM_TARGETS):
    ax1.plot(data[j][0], data[j][1], marker = 'o', c = DARK_RED, markersize = 10)
  
  #CV forward text
  ax1.text(data[0][0], data[0][1], 
           'CVForw\nx:{0:5.3f}\ny:{1:5.3f}\nz:{2:5.3f}\nc:{3}'.format(
                                                               data[0][0],
                                                               data[0][1],
                                                               data[0][2],
                                                               data[0][3]), 
           bbox = dict(facecolor = DARK_GREEN, alpha = 0.3), color = 'w')
  
  #CV down text
  ax1.text(data[1][0], data[1][1], 
           'CVDown\nx:{0:5.3f}\ny:{1:5.3f}\nz:{2:5.3f}\nc:{3}'.format(
                                                               data[1][0],
                                                               data[1][1],
                                                               data[1][2],
                                                               data[1][3]), 
           bbox = dict(facecolor = DARK_GREEN, alpha = 0.3), color = 'w')


  #Sonar target text
  #ax1.text(data[2][0], data[2][1], 
  #         'CV Target\nx:{0:5.3f}\ny:{1:5.3f}\nz:{2:5.3f}'.format(data[2][0]
  #                                                             data[2][1]
  #                                                             data[2][2]), 
  #         bbox = dict(facecolor = DARK_GREEN, alpha = 0.5), fontsize = 8)
  

  #TODO implement this
  #for j in range(0, 3):
    #ax2.plot(x, pos[j])
 
  dataHist[0][49] = 1
  dataHist[0][48] = 2
  dataHist[0][47] = 4
  dataHist[0][46] = 10
  dataHist[0][45] = 5

  ax2.plot(dataHist[0], '-', color = DARK_RED)
  ax2.legend(['pos x: {}'.format(dataHist[0][49])], loc = 'upper left', numpoints = 1)
  
  #ax4.


  return ax1, ax2

  

#initFigure()

ani = animation.FuncAnimation(fig, animate, init_func = initFigure, interval = 1)
plt.show()
