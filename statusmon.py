#import matplotlib
import matplotlib.pyplot as plt
#TODO import matplotlib.animation as animation
#import matplotlib.gridspec as gridspec
import numpy as np
#import sys
#import time
#import re

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

'''[Polar Targets]----------------------------------------------------------'''
#set labels appropriately
ax1.set_theta_zero_location("N")
ax1.set_xticks(np.pi/180. * np.linspace(-180, 180, 36, endpoint=False))
ax1.tick_params(axis = 'x', direction = 'inout', colors = '#33ff3c')

ax1.set_yticklabels([])

#set gridline colors to green/darkgreen
for line in ax1.get_xgridlines():
  line.set_color('#007505')

for line in ax1.get_ygridlines():
  line.set_color('#33ff3c')

'''[Position]---------------------------------------------------------------'''
#TODO
'''[Velocity]---------------------------------------------------------------'''
#TODO
'''[Thruster Status]--------------------------------------------------------'''
#TODO
'''[Robot Model Rotation]---------------------------------------------------'''
#TODO
plt.show()
