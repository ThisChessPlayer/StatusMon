'''*-----------------------------------------------------------------------*---
                                                         Author: Jason Ma
                                                         Date  : Jul 17 2016
   File Name  : statusmon.py
   Description: 
---*-----------------------------------------------------------------------*'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#import sys
#import time
#import re

NUM_TARGETS = 2

FIG_WIDTH   = 16
FIG_HEIGHT  = 8
FIG_NAME    = 'Cubeception 3 Status Monitor'
PLOT_STYLE  = 'dark_background'
LIGHT_GREEN = '#00ff00'
DARK_GREEN  = '#008000'

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
ax1 = plt.subplot2grid((6, 12), (0, 0), rowspan = 6, colspan = 6, polar = True)
ax2 = plt.subplot2grid((6, 12), (0, 6), rowspan = 3, colspan = 3)
ax3 = plt.subplot2grid((6, 12), (0, 9), rowspan = 2, colspan = 3)
ax4 = plt.subplot2grid((6, 12), (3, 6), rowspan = 3, colspan = 3, projection = '3d')
ax5 = plt.subplot2grid((6, 12), (2, 9), rowspan = 4, colspan = 3)
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

cvfMark, = ax1.plot(data[0][0], data[0][1], marker = 'o', c = DARK_RED,
markersize = 10)

cvdMark, = ax1.plot(data[0][0], data[0][1], marker = 'o', c = DARK_RED,
markersize = 10)

#CV forward text
cvfText = ax1.text(data[0][0], data[0][1], '',
            bbox = dict(facecolor = DARK_GREEN, alpha = 0.3), color = 'w')
  
#CV down text
cvdText = ax1.text(data[1][0], data[1][1], '', 
            bbox = dict(facecolor = DARK_GREEN, alpha = 0.3), color = 'w')

#heatmap
heatmap = ax3.imshow(np.random.uniform(size = (3, 4)), cmap = 'RdBu', 
            interpolation = 'nearest')

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
 
  ax3.set_xticks([0, 1, 2, 3])
  ax3.set_yticks([0, 1, 2])

  ax3.set_xticklabels(['1', '2', '3', '4'])
  ax3.set_yticklabels(['X', 'Y', 'Z'])
  
  '''[Orientation]----------------------------------------------------------'''
  #set title
  ax4.set_title('Orientation')

  #enable grid
  ax4.grid(b = True)

  #set color of grid lines
  ax4.w_xaxis._axinfo.update({'grid' : {'color': (0, 0.25, 0, 1)}})
  ax4.w_yaxis._axinfo.update({'grid' : {'color': (0, 0.25, 0, 1)}})
  ax4.w_zaxis._axinfo.update({'grid' : {'color': (0, 0.25, 0, 1)}})
  
  #set color of backgrounds
  ax4.w_xaxis.set_pane_color((0, 0.075, 0, 1))
  ax4.w_yaxis.set_pane_color((0, 0.075, 0, 1))
  ax4.w_zaxis.set_pane_color((0, 0.125, 0, 1))

  #set color of axis lines
  ax4.w_xaxis.line.set_color((0, 1, 0, 1))
  ax4.w_yaxis.line.set_color((0, 1, 0, 1))
  ax4.w_zaxis.line.set_color((0, 1, 0, 1))

  ax4.set_xticks([0, 0.25, 0.5, 0.75, 1])
  ax4.set_yticks([0, 0.25, 0.5, 0.75, 1])
  ax4.set_zticks([0, 0.25, 0.5, 0.75, 1])

  #set green axis labels
  ax4.set_xlabel('X axis', color = LIGHT_GREEN)
  ax4.set_ylabel('Y axis', color = LIGHT_GREEN)
  ax4.set_zlabel('Z axis', color = LIGHT_GREEN)
  
  '''
  for line in ax4.get_xticklines():
    line.set_color('#000000')
  for line in ax4.get_yticklines():
    line.set_color('#000000')
  for line in ax4.get_zticklines():
    line.set_color('#000000')

  ax4.set_frame_on(True)
  '''
  #sets background color of subplot
  #ax4.patch.set_facecolor('#000000')

  #ax4.plot_wireframe(x, y, z, *args, **kwargs)
  #ax4.unit_cube(vals = None)

  '''[Status]---------------------------------------------------------------'''
  #set title
  ax5.set_title('Status')

  '''[Multiple Axes]--------------------------------------------------------'''
  for ax in ax3, ax4, ax5:
    ax.tick_params(axis = 'both', which = 'both', bottom = 'off', top = 'off',
           left = 'off', right = 'off')
           
  for ax in ax4, ax5:
    ax.tick_params(labelbottom = 'off', labelleft = 'off')

  return ax1, ax2, ax3, ax4, ax5

'''animate---------------------------------------------------------------------
Updates subplots of figure
----------------------------------------------------------------------------'''
def animate(i):
  
  global ax1, ax2, ax3, ax4, ax5, data, dataHist
  
  data[0][0] += 0.05
  data[0][1] = data[0][1] + 0.1 % 5

  data[1][0] += 0.05
  data[1][1] = data[1][1] + 0.3 % 5

  #determine max for scale adjustments
  if data[0][1] > data[1][1]:
    max = data[0][1]
  else:
    max = data[1][1]

  #adjust scale of ax1 to fit data nicely
  ax1.set_yticks(np.linspace(0, max * 6 / 5, 7))

  #TODO update CV forward data
  cvfMark.set_data(data[0][0], data[0][1])
  #cvfMark.set_color(LIGHT_GREEN) #TODO modify size by confidence
  #cvfMark.set_markersize(20)     #TODO modify color based on depth

  #TODO update CV down data
  cvdMark.set_data(data[1][0], data[1][1])
  #cvdMark.set_color(LIGHT_GREEN)
  #cvdMark.set_markersize(20)

  #CV forward text
  cvfText.set_position((data[0][0], data[0][1]))  
  cvfText.set_text('CVForw\nx:{0:5.3f}\ny:{1:5.3f}\nz:{2:5.3f}\nc:{3}'.format(
                             data[0][0], data[0][1], data[0][2], data[0][3]))
                                                                
  #CV down text
  cvdText.set_position((data[1][0], data[1][1]))  
  cvdText.set_text('CVDown\nx:{0:5.3f}\ny:{1:5.3f}\nz:{2:5.3f}\nc:{3}'.format(
                             data[1][0], data[1][1], data[1][2], data[1][3]))

  '''
  dataHist[0][49] = 1
  dataHist[1][44] = 2
  dataHist[2][39] = 4
  dataHist[3][34] = 6
  dataHist[4][29] = 8
  dataHist[5][24] = 10
  dataHist[6][19] = 12
  dataHist[7][14] = 14
  dataHist[8][9] = 16
  dataHist[9][4] = 18
  dataHist[10][1] = 20

  ax2.plot(dataHist[0], '-', color = '#ff0000')
  ax2.plot(dataHist[1], '-', color = '#cf0000')
  ax2.plot(dataHist[2], '-', color = '#8f0000')
  
  ax2.plot(dataHist[3], '-', color = '#00ff00')
  ax2.plot(dataHist[4], '-', color = '#00cf00')
  ax2.plot(dataHist[5], '-', color = '#008f00')
  ax2.plot(dataHist[6], '-', color = '#004f00')

  ax2.plot(dataHist[7], '-', color = '#0000ff')
  ax2.plot(dataHist[8], '-', color = '#0000cf')
  ax2.plot(dataHist[9], '-', color = '#00008f')
  ax2.plot(dataHist[10], '-', color = '#00004f')


  ax2.legend(['pos x: {}'.format(dataHist[0][49]),
              'pos y: {}'.format(dataHist[1][49]),
              'pos y: {}'.format(dataHist[2][49]),
              'vel x: {}'.format(dataHist[3][49]),
              'vel y: {}'.format(dataHist[4][49]),
              'vel z: {}'.format(dataHist[5][49]),
              'vel t: {}'.format(dataHist[6][49]),
              'acc x: {}'.format(dataHist[7][49]),
              'acc y: {}'.format(dataHist[8][49]),
              'acc z: {}'.format(dataHist[9][49]),
              'acc t: {}'.format(dataHist[10][49])], 
              loc = 'upper left', numpoints = 1)
  '''

  
  heatmap.set_array(np.random.uniform(size = (3, 4)))

  return cvfMark, cvdMark, cvfText, cvdText, ax1, heatmap

ani = animation.FuncAnimation(fig, animate, init_func = initFigure, interval =
50, blit = True)
plt.show()
