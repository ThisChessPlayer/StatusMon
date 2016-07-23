'''*-----------------------------------------------------------------------*---
                                                         Author: Jason Ma
                                                         Date  : Jul 17 2016
   File Name  : statusmon.py
   Description: Displays random data similar to Cubeception 3.0 data
---*-----------------------------------------------------------------------*'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from itertools import product, combinations
import numpy as np
from numpy import sin, cos

NUM_TARGETS  = 2
FIG_WIDTH    = 16
FIG_HEIGHT   = 8
FIG_NAME     = 'Cubeception 3 Status Monitor'
PLOT_STYLE   = 'dark_background'
LIGHT_GREEN  = (0, 1, 0, 1)
DARK_GREEN   = (0, 0.5, 0, 1)
LIGHT_RED    = (1, 0.75, 0.75, 1)
DARK_RED     = (1, 0, 0, 1)
LIGHT_YELLOW = (1, 1, 0, 1)

'''initGlobals-----------------------------------------------------------------
Generates figure and subplots, sets basic layout.
----------------------------------------------------------------------------'''
plt.style.use(PLOT_STYLE)

mpl.rc(('text', 'xtick', 'ytick'), color = LIGHT_GREEN)
mpl.rc(('lines', 'grid'), color = DARK_GREEN)
mpl.rc('axes', edgecolor = LIGHT_GREEN, titlesize = 10)
mpl.rc('font', size = 8)
mpl.rc('grid', linestyle = ':')

#most recent data
data = np.zeros((2,4))
data[0][0] = 3
data[0][1] = 5
data[0][2] = -4
data[0][3] = 0

data[1][0] = 0
data[1][1] = 0
data[1][2] = -6
data[1][3] = 256

#past data
dataHist = np.zeros((11, 50))
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

#colors for ax2 plots
colors = ['#ff0000', '#cf0000', '#8f0000', '#00ff00', '#00cf00', '#008f00',
          '#004f00', '#0000ff', '#0000cf', '#00008f', '#00004f']

#create figure with 16:8 (width:height) ratio
fig = plt.figure(figsize = (FIG_WIDTH, FIG_HEIGHT), dpi = 100)
fig.canvas.set_window_title(FIG_NAME)
fig.suptitle("DEMO VERSION")
  
#create subplots on a 4 row 8 column grid
ax1 = plt.subplot2grid((6, 12), (0, 0), rowspan = 6, colspan = 6, polar = True)
ax2 = plt.subplot2grid((6, 12), (0, 6), rowspan = 3, colspan = 3, projection = '3d')
ax3 = plt.subplot2grid((6, 12), (0, 9), rowspan = 2, colspan = 3)
ax4 = plt.subplot2grid((6, 12), (3, 6), rowspan = 3, colspan = 3)
ax5 = plt.subplot2grid((6, 12), (2, 9), rowspan = 4, colspan = 3)
plt.tight_layout(pad = 2)

#polar target marks and text
cvfMark, = ax1.plot(0, 0, marker = 'o', c = DARK_RED, markersize = 10)
cvdMark, = ax1.plot(0, 0, marker = 'o', c = DARK_RED, markersize = 10)
cvfText = ax1.text(0, 0, '', bbox = dict(facecolor = DARK_GREEN, alpha = 0.3), color = 'w')
cvdText = ax1.text(0, 0, '', bbox = dict(facecolor = DARK_GREEN, alpha = 0.3), color = 'w')

#cube for orientation viewer
cube = np.zeros((3, 16))
cube[0] = [-1, -1, -1, 1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1,  1]
cube[1] = [-1, -1,  1, 1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1]
cube[2] = [-1,  1,  1, 1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1]
cubeLines = ax2.plot_wireframe(cube[0], cube[1], cube[2], colors = LIGHT_GREEN)

#arrow for locating front face of cube
ca = np.zeros((3, 8))
ca[0] = [0, 2, 1.75,  1.75, 2, 1.75,  1.75, 2]
ca[1] = [0, 0, 0.25, -0.25, 0,    0,     0, 0]
ca[2] = [0, 0,    0,     0, 0, 0.25, -0.25, 0]
cubeArrow = ax2.plot_wireframe(ca[0], ca[1], ca[2], colors = LIGHT_YELLOW)

#heatmap
heatmap = ax3.imshow(np.random.uniform(size = (3, 4)), cmap = 'RdBu', interpolation = 'nearest')

#position graph plots
mLines = [ax4.plot([], '-', color = colors[j])[0] for j in range(11)]

status = ax5.text(0, 0, '')

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
  
  '''[Orientation]----------------------------------------------------------'''
  #set title
  ax2.set_title('Orientation')

  #enable grid
  ax2.grid(b = False)

  #set color of grid lines (only for reference)
  #ax2.w_xaxis._axinfo.update({'grid' : {'color': (0, 0.25, 0, 1)}})
  #ax2.w_yaxis._axinfo.update({'grid' : {'color': (0, 0.25, 0, 1)}})
  #ax2.w_zaxis._axinfo.update({'grid' : {'color': (0, 0.25, 0, 1)}})
  
  #set color of backgrounds
  #ax2.w_xaxis.set_pane_color((0, 0, 0, 1))
  #ax2.w_yaxis.set_pane_color((0, 0, 0, 1))
  #ax2.w_zaxis.set_pane_color((0, 0, 0, 1))
  ax2.w_xaxis.set_pane_color((0, 0.075, 0, 1))
  ax2.w_yaxis.set_pane_color((0, 0.075, 0, 1))
  ax2.w_zaxis.set_pane_color((0, 0.125, 0, 1))

  #set color of axis lines
  ax2.w_xaxis.line.set_color(LIGHT_GREEN)
  ax2.w_yaxis.line.set_color(LIGHT_GREEN)
  ax2.w_zaxis.line.set_color(LIGHT_GREEN)

  #set tick lines
  ax2.set_xticks([])
  ax2.set_yticks([])
  ax2.set_zticks([])

  #set green axis labels
  ax2.set_xlabel('X axis', color = LIGHT_GREEN)
  ax2.set_ylabel('Y axis', color = LIGHT_GREEN)
  ax2.set_zlabel('Z axis', color = LIGHT_GREEN)

  '''[Thruster Heatmap]-----------------------------------------------------'''
  #set title
  ax3.set_title('Thruster Heatmap')

  #set ticks to properly extract parts of data
  ax3.set_xticks([0, 1, 2, 3])
  ax3.set_yticks([0, 1, 2])

  #label ticks so they correspond to motors
  ax3.set_xticklabels(['1', '2', '3', '4'])
  ax3.set_yticklabels(['X', 'Y', 'Z'])
  
  '''[Position/Velocity/Acceleration]---------------------------------------'''
  #set title
  ax4.set_title('Movement')
  
  #set x scale
  ax4.set_xticks(np.linspace(0, 50, 11))
  
  #enable grid
  ax4.grid(True)

  '''[Status]---------------------------------------------------------------'''
  #set title
  ax5.set_title('Status')

  '''[Multiple Axes]--------------------------------------------------------'''
  for ax in ax2, ax3, ax5:
    ax.tick_params(axis = 'both', which = 'both', bottom = 'off', top = 'off',
           left = 'off', right = 'off')
           
  for ax in ax2, ax5:
    ax.tick_params(labelbottom = 'off', labelleft = 'off')

'''quaternionFuncs-------------------------------------------------------------
Functions to create and use quaternions for robot orientation viewer
----------------------------------------------------------------------------'''
def normalize(v, tolerance = 0.00001):
  magnSqr = sum(n * n for n in v)
  if abs(magnSqr - 1.0) > tolerance:
    magn = pow(magnSqr, 1/2)
    v = tuple(n / magn for n in v)
  return v

def q_mult(q1, q2):
  w1, x1, y1, z1 = q1
  w2, x2, y2, z2 = q2
  w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
  x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
  y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
  z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
  return w, x, y, z

def q_conjugate(q):
  w, x, y, z = q
  return (w, -x, -y, -z)

def qv_mult(q1, v1):
  q2 = (0.0,) + v1
  return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

def axisangle_to_q(v, theta):
  v = normalize(v)
  x, y, z = v
  theta /= 2
  w = cos(theta)
  x = x * sin(theta)
  y = y * sin(theta)
  z = z * sin(theta)
  return w, x, y, z

def q_to_axisangle(q):
  w, v = q[0], q[1:]
  theta = acos(w) * 2.0
  return normalize(v), theta

'''animate---------------------------------------------------------------------
Updates subplots of figure
----------------------------------------------------------------------------'''
def animate(i):
  
  global ax1, ax2, ax3, ax4, ax5, data, dataHist, cubeLines, cubeArrow
  
  #store data updates
  data[0][0] += 0.05
  data[0][1] = data[0][1] + 0.1 % 5
  data[0][2] -= 1
  data[0][3] += 16
  data[1][0] += 0.05
  data[1][1] = data[1][1] + 0.3 % 5
  data[1][2] += 1
  data[1][3] -= 16

  '''[Polar Targets]--------------------------------------------------------'''  
  #determine max for scale adjustments
  if data[0][1] > data[1][1]:
    max = data[0][1]
  else:
    max = data[1][1]

  #adjust scale of ax1 to fit data nicely
  ax1.set_yticks(np.linspace(0, max * 6 / 5, 7))

  #ensure statusmon doesn't crash if CV returns crazy values
  if data[0][2] > 0:
    data[0][2] = 0
  elif data[0][2] < -10:
    data[0][2] = -10

  if data[1][2] > 10:
    data[1][2] = 10
  elif data[1][2] < -10:
    data[1][2] = -10

  if data[0][3] < 0:
    data[0][3] = 0
  elif data[0][3] > 255:
    data[0][3] = 255

  if data[1][3] < 0:
    data[1][3] = 0
  elif data[1][3] > 255:
    data[1][3] = 255

  #update CV forward data
  cvfMark.set_data(data[0][0], data[0][1])
  cvfMark.set_color((1, data[0][2] / -10, 0, 1))
  cvfMark.set_markersize(20 - data[0][3] * 5 / 128) 

  #update CV down data
  cvdMark.set_data(data[1][0], data[1][1])
  cvdMark.set_color((1, data[1][2] / -20 + 0.5, 0, 1))
  cvdMark.set_markersize(20 - data[1][3] * 5 / 128)

  #CV forward text
  cvfText.set_position((data[0][0], data[0][1]))  
  cvfText.set_text('CVForw\nx:{0:5.3f}\ny:{1:5.3f}\nz:{2:5.3f}\nc:{3}'.format(
                             data[0][0], data[0][1], data[0][2], data[0][3]))
                                                                
  #CV down text
  cvdText.set_position((data[1][0], data[1][1]))  
  cvdText.set_text('CVDown\nx:{0:5.3f}\ny:{1:5.3f}\nz:{2:5.3f}\nc:{3}'.format(
                             data[1][0], data[1][1], data[1][2], data[1][3]))

  '''[Orientation]----------------------------------------------------------'''
  #create 3 random quaternions
  q1 = axisangle_to_q((1, 0, 0), np.pi / 8)
  q2 = axisangle_to_q((0, 1, 0), np.pi / 8)
  q3 = axisangle_to_q((0, 1, 1), np.pi / 8)
  
  #multiply all 3 quaternions into one for a single rotation transformation
  quat = q_mult(q1, q2)
  quat = q_mult(quat, q3)

  #apply transformation to all points of cube
  for j in range(16):
    v = qv_mult(quat, (cube[0][j], cube[1][j], cube[2][j]))
    cube[0][j] = v[0]
    cube[1][j] = v[1]
    cube[2][j] = v[2]
  
  #apply transformation to all points of front facing arrow
  for j in range(8):
    v = qv_mult(quat, (ca[0][j], ca[1][j], ca[2][j]))
    ca[0][j] = v[0]
    ca[1][j] = v[1]
    ca[2][j] = v[2]
  
  #remove old wireframes and plot new ones
  cubeLines.remove()
  cubeLines = ax2.plot_wireframe(cube[0], cube[1], cube[2], colors = LIGHT_GREEN)
  cubeArrow.remove()
  cubeArrow = ax2.plot_wireframe(ca[0], ca[1], ca[2], colors = LIGHT_YELLOW)

  '''[Thruster Heatmap]-----------------------------------------------------'''
  #update motor heatmap
  heatmap.set_array(np.random.uniform(size = (3, 4)))

  '''[Position/Velocity/Acceleration]---------------------------------------'''
  #update data for ax4 plots
  x = np.linspace(0, 49, 50)

  #update data for each plot
  for j in range(11):
    mLines[j].set_data(x, dataHist[j])
  
  #determine highest value to scale y axis properly
  ymax = dataHist[0][0]
  ymin = dataHist[0][0]
  for j in range(11):
    for k in range(50):
      if dataHist[j][k] > ymax:
        ymax = dataHist[j][k]
      elif dataHist[j][k] < ymin:
        ymin = dataHist[j][k]
      dataHist[j][k] += 1

  #scale ax4 plot
  ax4.set_ylim(ymin, ymax + (ymax - ymin) / 5)
  ax4.set_yticks(np.linspace(ymin, ymax + (ymax - ymin) / 5, 7))

  #update legend with latest data values
  ax4.legend(['px: {}'.format(dataHist[0][49]),
              'py: {}'.format(dataHist[1][49]),
              'py: {}'.format(dataHist[2][49]),
              'vx: {}'.format(dataHist[3][49]),
              'vy: {}'.format(dataHist[4][49]),
              'vz: {}'.format(dataHist[5][49]),
              'vt: {}'.format(dataHist[6][49]),
              'ax: {}'.format(dataHist[7][49]),
              'ay: {}'.format(dataHist[8][49]),
              'az: {}'.format(dataHist[9][49]),
              'at: {}'.format(dataHist[10][49])], 
              loc = 'upper left', numpoints = 1)

  '''[Multiple Axes]--------------------------------------------------------'''
  #TODO
  #status.set_text()

#set up animation
ani = animation.FuncAnimation(fig, animate, init_func = initFigure, 
                              interval = 1000)

#show the figure
plt.show()

