#!/usr/bin/python3
'''*-----------------------------------------------------------------------*---
                                                         Author: Jason Ma
                                                         Date  : Jul 17 2016
   File Name  : statusmon.py
   Description: Displays data from buffers that Cubeception 3 writes to.
                The monitor includes a polar graph for targets, orientation 
                viewer, thruster heatmap, location/velocity/acceleration plots,
                and buffer status messages. 
---*-----------------------------------------------------------------------*'''
import sys, getopt
sys.path.insert(0, '../DistributedSharedMemory/build')
sys.path.insert(0, '../PythonSharedBuffers/src')
import pydsm
from ctypes import *
from Sensor import *
from Master import *
from Navigation import *
from Vision import *
from Serialization import *
from Constants import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations
import numpy as np
from numpy import sin, cos

'''[RUN VARS]---------------------------------------------------------------'''
#DSM Constants
CLIENT_SERV  = SONAR_SERVER_ID #Server id to connect to
CLIENT_ID    = 60              #Client id to register to server
NUM_BUFFERS  = 12              #Number of buffers to read from
NUM_DEBUG    = 1               #Number of buffers to read debug from
#DSM Buffer Values
bufNames = [MOTOR_KILL,         MOTOR_HEALTH,       MOTOR_OUTPUTS,
            SENSORS_LINEAR,     SENSORS_ANGULAR,    SENSORS_DATA, 
            MASTER_CONTROL,     MASTER_GOALS,       MASTER_SENSOR_RESET,
            TARGET_LOCATION,    TARGET_LOCATION,    TARGET_LOCATION]
bufIps = [MOTOR_SERVER_IP,          MOTOR_SERVER_IP,           MOTOR_SERVER_IP,
          SENSOR_SERVER_IP,         SENSOR_SERVER_IP,          SENSOR_SERVER_IP, 
          '10.0.1.8',         MASTER_SERVER_IP,          MASTER_SERVER_IP, 
          FORWARD_VISION_SERVER_IP, DOWNWARD_VISION_SERVER_IP, SONAR_SERVER_IP]
bufIds = [MOTOR_SERVER_ID,          MOTOR_SERVER_ID,           MOTOR_SERVER_ID,
          SENSOR_SERVER_ID,         SENSOR_SERVER_ID,          SENSOR_SERVER_ID, 
          MASTER_SERVER_ID,         MASTER_SERVER_ID,          MASTER_SERVER_ID, 
          FORWARD_VISION_SERVER_ID, DOWNWARD_VISION_SERVER_ID, SONAR_SERVER_ID]

#Arg parse constants
MODE_LIVE  = 0
MODE_DEBUG = 1
MODE_DEMO  = 2

INIT_ZERO  = 0
INIT_RAND  = 1

#Animation Constants
NUM_PL_LINES = 36   #Number of polar theta lines to plot
NUM_TARGETS  = 2    #Number of targets to plot on polar targets viewer
CUBE_POINTS  = 16   #Number of points in cube orientation plot
ARROW_POINTS = 8    #Number of points in cube arrow plot
NUM_MV_LINES = 11   #Number o movement lines to plot
HIST_LENGTH  = 50   #Number of past data points to store for movement viewer
DELAY        = 1000 #Millisecond delay between drawings

#Display Constants
FIG_WIDTH    = 16                             #Aspect width
FIG_HEIGHT   = 8                              #Aspect height
FIG_NAME     = 'Cubeception 3 Status Monitor' #Name displayed in window
PLOT_STYLE   = 'dark_background'              #Background style
LIGHT_GREEN  = (0, 1, 0, 1)                   #RGBA color
DARK_GREEN   = (0, 0.5, 0, 1)                 #RGBA color
LIGHT_RED    = (1, 0.75, 0.75, 1)             #RGBA color
DARK_RED     = (1, 0, 0, 1)                   #RGBA color
LIGHT_YELLOW = (1, 1, 0, 1)                   #RGBA color
DPI_DISPLAY  = 100                            #Dots per inch of display
FONT_SIZE    = 8                              #Default text font size
TITLE_SIZE   = 10                             #Default title font size

'''Parse Args------------------------------------------------------------------
Parse command line args
----------------------------------------------------------------------------'''
mode = MODE_LIVE     #Default mode is ReadBufferMode
randInit = INIT_ZERO  #Default init is 0 init

modeStr = ['Live ', 'Debug', 'Demo ']
initStr = ['Zero', 'Rand']

if len(sys.argv) > 1:
  try:
    opts, args = getopt.getopt(sys.argv[1:], 'hm:r')
  except getopt.GetoptError as err:
    #Print error message, then print short usage, then exit
    print(err)
    print('statusmon.py [-m] (Set mode) [-r] (Rand init) | [-h] (Show Help)')
    sys.exit(2)

  for opt, arg in opts:
    if opt == '-h':    #Print help and exit
      print('statusmon.py [-m] | [-h]\n'\
            '-m   Set Mode (debug, demo)\n'\
            '-r   Random Data Init\n'\
            '-h   Show help')
      sys.exit()
    elif opt == '-m':  #Set mode to demo
      if arg == 'debug':
        mode = MODE_DEBUG
      elif arg == 'demo':
        mode = MODE_DEMO
    elif opt == '-r':  #Set init mode to random
      randInit = INIT_RAND

print('[Mode: {}  ]'.format(modeStr[mode]))
print('[Init: {}   ]'.format(initStr[randInit]))


'''Init------------------------------------------------------------------------
Generates figure and subplots, sets base layout and initializes data
----------------------------------------------------------------------------'''

'''[Connect to DSM Server]--------------------------------------------------'''
#Initialize client
client = pydsm.Client(CLIENT_SERV, CLIENT_ID, True)

#Initialize remote buffers
for i in range(NUM_BUFFERS):
  client.registerRemoteBuffer(bufNames[i], bufIps[i], int(bufIds[i]))

'''[Initialize Figure/Subplots]---------------------------------------------'''
#Background style of figure
plt.style.use(PLOT_STYLE)

#Set default matplotlib artist values
mpl.rc(('text', 'xtick', 'ytick'), color = LIGHT_GREEN)
mpl.rc(('lines', 'grid'), color = DARK_GREEN)
mpl.rc('axes', edgecolor = LIGHT_GREEN, titlesize = TITLE_SIZE)
mpl.rc('font', size = FONT_SIZE)
mpl.rc('grid', linestyle = ':')

#Create figure with 16:8 (width:height) ratio
fig = plt.figure(figsize = (FIG_WIDTH, FIG_HEIGHT), dpi = DPI_DISPLAY)
fig.canvas.set_window_title(FIG_NAME)

#Set title of figure
fig.suptitle('{} Mode'.format(modeStr[mode]))

#Create subplots on a 4 row 8 column grid
ax1 = plt.subplot2grid((6, 12), (0, 0), rowspan = 6, colspan = 6, polar = True)
ax2 = plt.subplot2grid((6, 12), (0, 6), rowspan = 3, colspan = 3, 
                                                             projection = '3d')
ax3 = plt.subplot2grid((6, 12), (0, 9), rowspan = 2, colspan = 3)
ax4 = plt.subplot2grid((6, 12), (3, 6), rowspan = 3, colspan = 3)
ax5 = plt.subplot2grid((6, 12), (2, 9), rowspan = 4, colspan = 3)
plt.tight_layout(pad = 2)

'''[Initialize Data]--------------------------------------------------------'''
#Holds all displayed data from buffers
cvforwardData    = np.zeros((3, 5))
cvdownData       = np.zeros((5))
orientationData  = np.zeros((4))
thrusterData     = np.zeros((2, 4))
movementData     = np.zeros((3, 4))
statusData       = np.zeros((3))

#Debug data
masterControlData = np.zeros((3, 3, 3))

'''[Init Polar Targets]-----------------------------------------------------'''

if randInit == INIT_RAND:
  for i in range(3):
    cvforwardData[i][0] = np.random.randint(0, 4)
    cvforwardData[i][1] = np.random.randint(0, 5)
    cvforwardData[i][2] = np.random.randint(-10, 10)
    cvforwardData[i][3] = np.random.randint(0, 255)
    cvforwardData[i][4] = np.random.randint(0, 5)

  cvdownData[0] = np.random.randint(0, 4)
  cvdownData[1] = np.random.randint(0, 5)
  cvdownData[2] = np.random.randint(-10, 10)
  cvdownData[3] = np.random.randint(0, 255)
  cvdownData[4] = np.random.randint(0, 5)

cvfMark = np.empty(3, dtype = object)
cvfText = np.empty(3, dtype = object)

#Polar target marks and text
for i in range(3):
  cvfMark[i], = ax1.plot(0, 0, marker = 'o', c = DARK_RED, markersize = 10)
  cvfText[i] = ax1.text(0, 0, '', 
                bbox = dict(facecolor = DARK_GREEN, alpha = 0.3), color = 'w')

cvdMark, = ax1.plot(0, 0, marker = 'o', c = DARK_RED, markersize = 10)
cvdText = ax1.text(0, 0, '', 
                bbox = dict(facecolor = DARK_GREEN, alpha = 0.3), color = 'w')

'''[Init Orientation]-------------------------------------------------------'''
#Cube for orientation viewer
cube = np.zeros((3, CUBE_POINTS))
cube[0] = [-1, -1, -1, 1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1,  1]
cube[1] = [-1, -1,  1, 1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1]
cube[2] = [-1,  1,  1, 1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1]
cubeLines = ax2.plot_wireframe(cube[0], cube[1], cube[2], colors = LIGHT_GREEN)

#Arrow for locating front face of cube
ca = np.zeros((3, ARROW_POINTS))
ca[0] = [0, 2, 1.75,  1.75, 2, 1.75,  1.75, 2]
ca[1] = [0, 0, 0.25, -0.25, 0,    0,     0, 0]
ca[2] = [0, 0,    0,     0, 0, 0.25, -0.25, 0]
cubeArrow = ax2.plot_wireframe(ca[0], ca[1], ca[2], colors = LIGHT_YELLOW)

'''[Init Heatmap]-----------------------------------------------------------'''
heatmap = ax3.imshow(np.random.uniform(size = (3, 4)), 
                     cmap = 'RdBu', interpolation = 'nearest')

if randInit == INIT_ZERO:
  heatmap.set_array(np.zeros((3, 4)))

'''[Init Movement]----------------------------------------------------------'''
#Past ax4 data to plot
dataHist = np.zeros((NUM_MV_LINES, HIST_LENGTH))
if randInit == INIT_ZERO:
  ax4.set_xticks(np.linspace(0, HIST_LENGTH - 1, HIST_LENGTH))
  ax4.set_yticks(np.linspace(-1, 1, 5))
  ax4.set_ylim(-1, 1)  
if randInit == INIT_RAND:
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

#Colors for ax4 plots
colors = ['#ff0000', '#cf0000', '#8f0000', '#00ff00', '#00cf00', '#008f00',
          '#004f00', '#0000ff', '#0000cf', '#00008f', '#00004f']

#Initialize position graph plots
mLines = [ax4.plot([], '-', color = colors[j])[0] for j in range(NUM_MV_LINES)]


'''[Init Status]------------------------------------------------------------'''
statusStrings = np.empty(NUM_BUFFERS, dtype = 'object')
status = ax5.text(0.05, .55, 'Loading')
debugStatus = ax5.text(0.05, .3, '')
status.set_family('monospace')
debugStatus.set_family('monospace')
'''initPlot--------------------------------------------------------------------
Sets up subplots and starting image of figure to display
----------------------------------------------------------------------------'''
def initFigure():

  '''[Polar Targets]--------------------------------------------------------'''
  #Set subplot title
  ax1.set_title('Targets')
  
  #Set label locations appropriately
  ax1.set_theta_zero_location("N")
  ax1.set_theta_direction(-1)
  
  #Format ticks and labels
  ax1.set_thetagrids(np.linspace(0, 360, NUM_PL_LINES, endpoint = False), 
                     frac = 1.05)
  ax1.set_rlabel_position(90)

  #Make ygridlines more visible (circular lines)
  for line in ax1.get_ygridlines():
    line.set_color(LIGHT_GREEN)
  
  '''[Orientation]----------------------------------------------------------'''
  #Set subplot title
  ax2.set_title('Orientation')

  #Enable grid
  ax2.grid(b = False)
  
  #Set color of backgrounds
  ax2.w_xaxis.set_pane_color((0, 0.075, 0, 1))
  ax2.w_yaxis.set_pane_color((0, 0.075, 0, 1))
  ax2.w_zaxis.set_pane_color((0, 0.125, 0, 1))

  #Set color of axis lines
  ax2.w_xaxis.line.set_color(LIGHT_GREEN)
  ax2.w_yaxis.line.set_color(LIGHT_GREEN)
  ax2.w_zaxis.line.set_color(LIGHT_GREEN)

  #Set tick lines
  ax2.set_xticks([])
  ax2.set_yticks([])
  ax2.set_zticks([])

  #Set green axis labels
  ax2.set_xlabel('X axis', color = LIGHT_GREEN)
  ax2.set_ylabel('Y axis', color = LIGHT_GREEN)
  ax2.set_zlabel('Z axis', color = LIGHT_GREEN)

  '''[Thruster Heatmap]-----------------------------------------------------'''
  #Set subplot title
  ax3.set_title('Thruster Heatmap')

  #Set ticks to properly extract parts of data
  ax3.set_xticks([0, 1, 2, 3])
  ax3.set_yticks([0, 1, 2])

  #Label ticks so they correspond to motors
  ax3.set_xticklabels(['1', '2', '3', '4'])
  ax3.set_yticklabels(['X', 'Y', 'Z'])
  
  '''[Position/Velocity/Acceleration]---------------------------------------'''
  #Set subplot title
  ax4.set_title('Movement')
  
  #Set x scale
  ax4.set_xticks(np.linspace(0, HIST_LENGTH, NUM_MV_LINES))
  
  #Enable grid
  ax4.grid(True)

  '''[Status]---------------------------------------------------------------'''
  #Set subplot title
  ax5.set_title('Status')

  '''[Multiple Axes]--------------------------------------------------------'''
  for ax in ax2, ax3, ax5:
    ax.tick_params(axis = 'both', which = 'both', bottom = 'off', top = 'off',
           left = 'off', right = 'off')
           
  for ax in ax2, ax5:
    ax.tick_params(labelbottom = 'off', labelleft = 'off')

  print('[Init Complete]')

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

'''genData---------------------------------------------------------------------
Generates fake data to display
----------------------------------------------------------------------------'''
def genData():
  #Set all buffer strings to active
  for i in range(NUM_BUFFERS):
    statusStrings[i] = 'Up  '

  #Generate forward and downward computer vision data
  for i in range(3):
    cvforwardData[i][0] = np.random.randint(0, 3)
    cvforwardData[i][1] = np.random.randint(0, 5)
    cvforwardData[i][2] = np.random.randint(-10, 10)
    cvforwardData[i][3] = np.random.randint(0, 255)
    cvforwardData[i][4] = np.random.randint(0, 5)

  cvdownData[0] = np.random.randint(0, 3)
  cvdownData[1] = np.random.randint(0, 5)
  cvdownData[2] = np.random.randint(-10, 10)
  cvdownData[3] = np.random.randint(0, 255)
  cvdownData[4] = np.random.randint(0, 5)

  #Generate 3 quaternions representing 3 rotations
  q1 = axisangle_to_q((1, 0, 0), np.random.randint(0, 3) / 8)
  q2 = axisangle_to_q((0, 1, 0), np.random.randint(0, 3) / 8)
  q3 = axisangle_to_q((0, 1, 1), np.random.randint(0, 3) / 8)

  #Multiply all 3 quaternions into one for a single rotation transformation
  quat = q_mult(q_mult(q1, q2), q3)
  
  for i in range(4):
    orientationData[i] = quat[i]
  
  #Generate thruster output data
  for i in range(2):
    for j in range(4):
      thrusterData[i][j] = np.random.randint(0, 20) / 20
  
  #Generate movement data
  for i in range(3):
    for j in range(3):
      movementData[i][j] += np.random.randint(-2, 2)

'''getBufferData---------------------------------------------------------------
Obtains most recent buffer data
----------------------------------------------------------------------------'''
def getBufferData(debug):
  
  #Check each buffer's status and update data array if active
  for i in range(NUM_BUFFERS):
    temp, active = client.getRemoteBufferContents(bufNames[i], bufIps[i], 
                                                  bufIds[i])
    if active:
      if i == 0:                          #Motor Kill
        temp = Unpack(Kill, temp)
        statusData[0] = temp.isKilled
      elif i == 1:                        #Motor Health
        temp = Unpack(Health, temp)
        statusData[1] = temp.saturated
        statusData[2] = temp.direction
      elif i == 2:                        #Motor Outputs
        temp = Unpack(Outputs, temp)
        for j in range(4):
          thrusterData[0][j] = temp.motors[j]
        for j in range(4):
          thrusterData[1][j] = temp.motors[j + 4]
      elif i == 3:                        #Sensors Linear
        temp = Unpack(Linear, temp)
        for j in range(3):
          movementData[0][j] = temp.pos[j]
          movementData[1][j] = temp.vel[j]
          movementData[2][j] = temp.acc[j]
      elif i == 4:                        #Sensors Angular
        temp = Unpack(Angular, temp)
        for j in range(4):
          orientationData[j] = temp.pos[j]
      #elif i == 5:                       #Sensors Data
      elif i == 6:                       #Master Control
        if debug:
          temp = Unpack(ControlInput, temp)
          masterControlData[2][0][0] = int(temp.mode)
          
          #Unpack angular data
          for j in range(3):
            if ((1 << j) & int(temp.mode)):
              masterControlData[0][j][0] = 0
              for k in range(2):
                masterControlData[0][j][k + 1] = temp.angular[j].pos[k]
            else:
              masterControlData[0][j][0] = temp.angular[j].vel
              for k in range(2):
                masterControlData[0][j][k + 1] = 0

          #Unpack linear data
          for j in range(3):
            if ((1 << (j + 3)) & int(temp.mode)):
              masterControlData[1][j][0] = 0
              for k in range(2):            
                masterControlData[1][j][k + 1] = temp.linear[j].pos[k]
            else:
              masterControlData[1][j][0] = temp.linear[j].vel
              for k in range(2):            
                masterControlData[1][j][k + 1] = 0

      #elif i == 7:                       #Master Goals
      #elif i == 8:                       #Master Sensor Reset
      elif i == 9:                        #CV Forw Target Location
        temp = Unpack(LocationArray, temp)
        for j in range(3):
          cvforwardData[j][0] = temp.locations[j].x
          cvforwardData[j][1] = temp.locations[j].y
          cvforwardData[j][2] = temp.locations[j].z
          cvforwardData[j][3] = temp.locations[j].confidence
          cvforwardData[j][4] = temp.locations[j].loctype
      elif i == 10:                       #CV Down Target Location
        temp = Unpack(Location, temp)
        cvdownData[0][0] = temp.x
        cvdownData[0][1] = temp.y
        cvdownData[0][2] = temp.z
        cvdownData[0][3] = temp.confidence
        cvdownData[0][4] = temp.loctype
      #elif i == 11:                      #Sonar Target Location

      #Set status string to indicate whether buffer is up or down
      statusStrings[i] = 'Up  '
    else:
      statusStrings[i] = 'Down'

'''animate---------------------------------------------------------------------
Updates subplots of figure
----------------------------------------------------------------------------'''
def animate(i):
  
  global mode, ax1, ax2, ax3, ax4, ax5, data, dataHist, cubeLines, cubeArrow

  #Grab latest data to plot as well as info on whether buffers are online
  if mode == MODE_DEMO:
    genData()
  else:
    getBufferData(mode)
  
  '''[Polar Targets]--------------------------------------------------------'''  

  #Ensure statusmon doesn't crash if CV returns crazy values
  for j in range(3):
    if cvforwardData[j][2] > 0:
      cvforwardData[j][2] = 0
    elif cvforwardData[j][2] < -10:
      cvforwardData[j][2] = -10

    if cvforwardData[j][3] < 0:
      cvforwardData[j][3] = 0
    elif cvforwardData[j][3] > 255:
      cvforwardData[j][3] = 255

  if cvdownData[2] > 10:
    cvdownData[2] = 10
  elif cvdownData[2] < -10:
    cvdownData[2] = -10

  if cvdownData[3] < 0:
    cvdownData[3] = 0
  elif cvdownData[3] > 255:
    cvdownData[3] = 255

  maxR = 0
  for j in range(3):
    polarR = pow(pow(cvforwardData[j][0], 2) + 
                 pow(cvforwardData[j][1], 2), 1/2)

    if polarR > maxR:
      maxR = polarR

    if cvforwardData[j][0] != 0:
      polarT = np.arctan(cvforwardData[j][1] / cvforwardData[j][0])
    else:
      polarT = np.pi / 2

    #Update CV forward data
    cvfMark[j].set_data(polarT, polarR)
    cvfMark[j].set_color((1, cvforwardData[j][2] / -10, 0, 1))
    cvfMark[j].set_markersize(20 - cvforwardData[j][3] * 5 / 128)

    #Update CV forward text
    cvfText[j].set_position((polarT, polarR))
    cvfText[j].set_text('CVForw\nx:{0:5.3f}\n'\
                        'y:{1:5.3f}\nz:{2:5.3f}\nc:{3}'.format(
                             cvforwardData[j][0], cvforwardData[j][1], 
                             cvforwardData[j][2], cvforwardData[j][3]))

  polarR = pow(pow(cvdownData[0], 2) + 
                 pow(cvdownData[1], 2), 1/2)

  if polarR > maxR:
    maxR = polarR

  if cvdownData[0] != 0:
    polarT = np.arctan(cvdownData[1] / cvdownData[0])
  else:
    polarT = np.pi / 2

  #Update CV down data
  cvdMark.set_data(polarT, polarR)
  cvdMark.set_color((1, cvdownData[2] / -20 + 0.5, 0, 1))
  cvdMark.set_markersize(20 - cvdownData[3] * 5 / 128)

  #Update CV down text
  cvdText.set_position((polarT, polarR))  
  cvdText.set_text('CVDown\nx:{0:5.3f}\n'\
                   'y:{1:5.3f}\nz:{2:5.3f}\nc:{3}'.format(
                             cvdownData[0], cvdownData[1], 
                             cvdownData[2], cvdownData[3]))
  
  #Adjust scale of ax1 to fit data nicely
  if maxR != 0:
    ax1.set_yticks(np.linspace(0, maxR * 6 / 5, 7))
    ax1.set_ylim(0, maxR * 6 / 5)

  '''[Orientation]----------------------------------------------------------'''
  #Only rotate model if stream is online
  if statusStrings[4] == 'Up  ':
    quat = (orientationData[0], orientationData[1], 
            orientationData[2], orientationData[3])
  else:
    #Default quaternion results in no rotation
    quat = (1, 0, 0, 0)
   
  #Reset orientation of cube and arrow
  cube[0] = [-1, -1, -1, 1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1,  1]
  cube[1] = [-1, -1,  1, 1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1]
  cube[2] = [-1,  1,  1, 1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1]

  ca[0] = [0, 2, 1.75,  1.75, 2, 1.75,  1.75, 2]
  ca[1] = [0, 0, 0.25, -0.25, 0,    0,     0, 0]
  ca[2] = [0, 0,    0,     0, 0, 0.25, -0.25, 0]
  
  #Apply transformation to all points of cube
  for j in range(16):
    v = qv_mult(quat, (cube[0][j], cube[1][j], cube[2][j]))
    cube[0][j] = v[0]
    cube[1][j] = v[1]
    cube[2][j] = v[2]
  
  #Apply transformation to all points of front facing arrow
  for j in range(8):
    v = qv_mult(quat, (ca[0][j], ca[1][j], ca[2][j]))
    ca[0][j] = v[0]
    ca[1][j] = v[1]
    ca[2][j] = v[2]
  
  #Remove old wireframes and plot new ones
  cubeLines.remove()
  cubeArrow.remove()  
  cubeLines = ax2.plot_wireframe(cube[0], cube[1], cube[2], 
                                 colors = LIGHT_GREEN)
  cubeArrow = ax2.plot_wireframe(ca[0], ca[1], ca[2], 
                                 colors = LIGHT_YELLOW)
  
  '''[Thruster Heatmap]-----------------------------------------------------'''
  #Map data to heatmap
  heatArray = [[thrusterData[1][0], thrusterData[1][1], 0, 0], 
               [thrusterData[1][2], thrusterData[1][3], 0, 0], 
               [thrusterData[0][0], thrusterData[0][1], 
                thrusterData[0][2], thrusterData[0][3]]]
  
  #Update motor heatmap
  heatmap.set_array(heatArray)

  '''[Movement]-------------------------------------------------------------'''
  #Update data for ax4 plots
  moveX = np.linspace(0, HIST_LENGTH - 1, HIST_LENGTH)

  #Transfer data into data history
  for j in range(NUM_MV_LINES):
    for k in range(HIST_LENGTH - 1):
      dataHist[j][k] = dataHist[j][k + 1]
  for j in range(3):
    dataHist[j][HIST_LENGTH - 1] = movementData[0][j]
  for j in range(3):
    dataHist[j + 3][HIST_LENGTH - 1] = movementData[1][j]
  for j in range(3):
    dataHist[j + 7][HIST_LENGTH - 1] = movementData[2][j]

  #Calculate total velocity/acceleration
  dataHist[6][HIST_LENGTH - 1]  = pow(pow(movementData[1][0], 2) + 
                                      pow(movementData[1][1], 2) + 
                                      pow(movementData[1][2], 2), 1/2)
  dataHist[10][HIST_LENGTH - 1] = pow(pow(movementData[2][0], 2) + 
                                      pow(movementData[2][1], 2) + 
                                      pow(movementData[2][2], 2), 1/2)

  #Update data for each plot
  for j in range(NUM_MV_LINES):
    mLines[j].set_data(moveX, dataHist[j])
  
  #Determine highest value to scale y axis properly
  ymax = dataHist[0][0]
  ymin = dataHist[0][0]
  for j in range(NUM_MV_LINES):
    for k in range(HIST_LENGTH):
      if dataHist[j][k] > ymax:
        ymax = dataHist[j][k]
      elif dataHist[j][k] < ymin:
        ymin = dataHist[j][k]

  if ymin != ymax:
    ax4.set_ylim(ymin, ymax + (ymax - ymin) / 5)
    movementTicks = np.linspace(ymin, ymax + (ymax - ymin) / 5, 7)    
    ax4.set_yticks(movementTicks)

  #Update legend with latest data values
  ax4.legend(['px:{}'.format(round(dataHist[0][HIST_LENGTH - 1], 3)),
              'py:{}'.format(round(dataHist[1][HIST_LENGTH - 1], 3)),
              'py:{}'.format(round(dataHist[2][HIST_LENGTH - 1], 3)),
              'vx:{}'.format(round(dataHist[3][HIST_LENGTH - 1], 3)),
              'vy:{}'.format(round(dataHist[4][HIST_LENGTH - 1], 3)),
              'vz:{}'.format(round(dataHist[5][HIST_LENGTH - 1], 3)),
              'vt:{}'.format(round(dataHist[6][HIST_LENGTH - 1], 3)),
              'ax:{}'.format(round(dataHist[7][HIST_LENGTH - 1], 3)),
              'ay:{}'.format(round(dataHist[8][HIST_LENGTH - 1], 3)),
              'az:{}'.format(round(dataHist[9][HIST_LENGTH - 1], 3)),
              'at:{}'.format(round(dataHist[10][HIST_LENGTH - 1], 3))],
              loc = 'upper left', numpoints = 1)

  '''[Multiple Axes]--------------------------------------------------------'''
  #Update status text
  status.set_text('BUFFER STATUS---------------------------\n' \
         'Motor  Kill   : {}\nMotor  Health : {}\nMotor  Outputs: {}\n' \
         'Sensor Lin    : {}\nSensor Ang    : {}\nSensor Data   : {}\n' \
         'Master Control: {}\nMaster Goals  : {}\nMaster SensRes: {}\n' \
         'CVDown Target : {}\nCVForw Target : {}\nSonar  Target : {}\n\n' \
         'Kill Switch   : {}'.format(
            statusStrings[0], statusStrings[1], statusStrings[2], 
            statusStrings[3], statusStrings[4], statusStrings[5],
            statusStrings[6], statusStrings[7], statusStrings[8], 
            statusStrings[9], statusStrings[10], statusStrings[11],
            
            statusData[0]))
  if mode == MODE_DEBUG:
    debugStatus.set_text('BUFFER DEBUG----------------------------\n' \
         '[Master Control]\n' \
         'Ang X: vel: {} pos1: {} pos2: {}\n' \
         'Ang Y: vel: {} pos1: {} pos2: {}\n' \
         'Ang Z: vel: {} pos1: {} pos2: {}\n' \
         'Lin X : vel: {} pos1: {} pos2: {}\n' \
         'Lin Y : vel: {} pos1: {} pos2: {}\n' \
         'Lin Z : vel: {} pos1: {} pos2: {}\n' \
         'Mode  : {}'.format(
         round(masterControlData[0][0][0], 3),
         round(masterControlData[0][0][1], 3),
         round(masterControlData[0][0][2], 3),
         round(masterControlData[0][1][0], 3),
         round(masterControlData[0][1][1], 3),
         round(masterControlData[0][1][2], 3),
         round(masterControlData[0][2][0], 3),
         round(masterControlData[0][2][1], 3),
         round(masterControlData[0][2][2], 3),
         round(masterControlData[1][0][0], 3),
         round(masterControlData[1][0][1], 3),
         round(masterControlData[1][0][2], 3),
         round(masterControlData[1][1][0], 3),
         round(masterControlData[1][1][1], 3),
         round(masterControlData[1][1][2], 3),
         round(masterControlData[1][2][0], 3),
         round(masterControlData[1][2][1], 3),
         round(masterControlData[1][2][2], 3),
         round(masterControlData[2][0][0], 3),))

#Set up animation
ani = animation.FuncAnimation(fig, animate, init_func = initFigure, 
                              interval = DELAY)

#Show the figure
plt.show()

