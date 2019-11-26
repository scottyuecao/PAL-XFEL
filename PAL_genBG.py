#==============================================================================
# Data processing for tr experiments at PAL-XFEL
# 2019-11-23
# Sample: LBCO and TaS2
# 
# Ver. 4
# Yue Cao (ycao@colorado.edu)
#
# Usage: in cmd line, go to the scripts folder
#
# python -i PAL_genBG.py $1
# $1: bgID
#
#==============================================================================


#==============================================================================
# Importing all plotting packages
#==============================================================================

# import all plotting packages
import matplotlib.pyplot as plt
from matplotlib import colors, cm

# importing system packages
import os
import sys
import h5py
import time
import itertools
import glob

# importing the workhorse
import numpy as np
import pandas as pd

# tiff packages
from PIL import Image
import tifffile

# for the fitting procedures
from lmfit import *

#==============================================================================
# Global variables
#==============================================================================

# cmd line inputs
# Remember sys.argv[0] is the name of the python script
bgID = str(sys.argv[1])

folder = '/xfel/ffs/dat/ue_191123_FXS/reduced'

#==============================================================================
# Main program
#==============================================================================

# Summing all tiffs
tiffList = glob.glob1(folder+'/r'+str(bgID).zfill(3), '*.tif')

for j, nm in enumerate(tiffList):
    fpath = os.path.join(folder, 'r'+str(bgID).zfill(3), nm)
    img = tifffile.imread(fpath)
    if j:
        total += img
    else:
        total = img

# Summing all shots
csvNm = 'run'+str(bgID).zfill(3)+'_on_s'+str(1).zfill(3)+'.csv'
fpath = os.path.join(folder, 'r'+str(bgID).zfill(3), csvNm)
dfon = pd.read_csv(fpath, index_col=0)
onshots = np.nansum(dfon['shots'])

csvNm = 'run'+str(bgID).zfill(3)+'_off_s'+str(1).zfill(3)+'.csv'
fpath = os.path.join(folder, 'r'+str(bgID).zfill(3), csvNm)
dfoff = pd.read_csv(fpath, index_col=0)
offshots = np.nansum(dfoff['shots'])

bgImg = total/(onshots+offshots)
fpath = os.path.join(folder, 'BG_r'+str(bgID).zfill(3)+'.tif')
tifffile.imsave(fpath, bgImg)

print('****************************************')
print('* Total shots: {}'.format(onshots+offshots))
print('* BG saved as: BG_r'+str(bgID).zfill(3)+'.tif')
print('****************************************')
