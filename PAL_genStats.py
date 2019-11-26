#==============================================================================
# Data processing for tr experiments at PAL-XFEL
# 2019-11-22
# Sample: LBCO and TaS2
# 
# Ver. 5
# Yue Cao (ycao@colorado.edu)
#
# Usage: load in jupyter notebook
#
#==============================================================================


#==============================================================================
# Importing all plotting packages
#==============================================================================
# import all plotting packages
import matplotlib.pyplot as plt
from matplotlib import colors, cm
plt.style.use('ggplot')

# importing system packages
import os
import sys
import glob
import h5py
import time
import itertools
import tables as tb

# importing the workhorse
import numpy as np
import pandas as pd
import seaborn as sns

# tiff packages
from PIL import Image
import tifffile

# for the fitting procedures
from lmfit import *

#==============================================================================
# Defining user inputs
#==============================================================================

# Input file path
slrFolder = '/xfel/ffs/dat/ue_191123_FXS/raw_data/h5/type=measurement'
imgFolder = '/xfel/ffs/dat/ue_191123_FXS/raw_data/h5/type=raw'


#==============================================================================
# Defining Global variables
#==============================================================================

# Motor nickname dict
motorKey = {
'delay': 'delay_input',
'th': 'th_input', 
'laserStatus': 'event_info.FIFTEEN_HERTZ',
'laser_h': 'laser_h_input',
'laser_v': 'laser_v_input'}

motorRBV = {
'th_rbv': 'gonio:eh1:FourC:th',
'tth_rbv': 'gonio:eh1:FourC:tth',
'chi_rbv': 'gonio:eh1:FourC:chi', 
'phi_rbv': 'gonio:eh1:FourC:phi'
}

# All the scalars needed

# qbpm and pd readings
qbpm = {
'pink-ch1': 'qbpm:oh:qbpm1:ch1',
'pink-ch2': 'qbpm:oh:qbpm1:ch2',
'pink-ch3': 'qbpm:oh:qbpm1:ch3',
'pink-ch4': 'qbpm:oh:qbpm1:ch4',
'pink-sum': 'qbpm:oh:qbpm1:sum',
'mono-up-ch1': 'qbpm:oh:qbpm2:ch1',
'mono-up-ch2': 'qbpm:oh:qbpm2:ch2',
'mono-up-ch3': 'qbpm:oh:qbpm2:ch3',
'mono-up-ch4': 'qbpm:oh:qbpm2:ch4',
'mono-up-sum': 'qbpm:oh:qbpm2:sum',
'sample-ch1': 'qbpm:eh1:qbpm1:ch1',
'sample-ch2': 'qbpm:eh1:qbpm1:ch2',
'sample-ch3': 'qbpm:eh1:qbpm1:ch3',
'sample-ch4': 'qbpm:eh1:qbpm1:ch4',
'sample-sum': 'qbpm:eh1:qbpm1:sum',
#'pd1-ch1': 'pd:es:pd1:ch1',
'pd1-ch2': 'pd:es:pd1:ch2'#,
#'pd1-ch3': 'pd:es:pd1:ch3',
#'pd1-ch4': 'pd:es:pd1:ch4'
}

# beam positions
qbpm_pos = {
'pink-posx': 'qbpm:oh:qbpm1:pos_X',
'pink-posy': 'qbpm:oh:qbpm1:pos_Y',
'mono-up-posx': 'qbpm:oh:qbpm2:pos_X',
'mono-up-posy': 'qbpm:oh:qbpm2:pos_Y',
'sample-posx': 'qbpm:eh1:qbpm1:pos_X',
'sample-posy': 'qbpm:eh1:qbpm1:pos_Y'
}

# MPCCD readings
mpccd = {
'full-max': 'detector:eh1:mpccd1:frame.max', 
'full-min': 'detector:eh1:mpccd1:frame.min',
'full-mean': 'detector:eh1:mpccd1:frame.mean',
'full-std': 'detector:eh1:mpccd1:frame.std',
'roi1-max': 'detector:eh1:mpccd1:ROI1_stat.max',
'roi1-min': 'detector:eh1:mpccd1:ROI1_stat.min',
'roi1-mean': 'detector:eh1:mpccd1:ROI1_stat.mean',
'roi1-std': 'detector:eh1:mpccd1:ROI1_stat.std',
'roi1-sum': 'detector:eh1:mpccd1:ROI1_stat.sum'
}

# Peak positions on the MPCCD - not critical unless drifts
mpccd_pos = {
'roi1-comy': 'detector:eh1:mpccd1:ROI1_stat.center_of_mass.y',
'roi1-comx': 'detector:eh1:mpccd1:ROI1_stat.center_of_mass.x'
}

#==============================================================================
# Main program: Getting stats - note each h5 contains one motor position
#
# Reading an h5 could only be done as below due to missing h5py plugins
# at PAL-XFEL. This problem can be OS-dependent. Be aware!
#
# import tables as tb
#
# imgpath = os.path.join(imgFolder, imgNm)
# h5 = tb.open_file(imgpath)
# df_rawimage = pd.DataFrame(
#    {
#        'timestamp': h5.root.detector.eh1.mpccd1.image.axis1[:], 
#        'image':h5.root.detector.eh1.mpccd1.image.block0_values.tolist()
#    }
# ).set_index('timestamp')
#
# If the following code returns True, then no image is missing...
#
# list(df.index) == h5.root.detector.eh1.mpccd1.image.axis1[:].tolist()
#==============================================================================  

def genStats(runID, scanID=1, sliceNum=-1, ROI=[]):
    '''
    Generating stats for a given run, scan and motor position (one h5 file)
    scanID:         Default is 1
    sliceNum:       Default is -1, which is the center slice
    ROI:            Default is the entire screen
    '''
    # Getting the total number of motor positions
    motorNum = len(glob.glob1(os.path.abspath(slrFolder+'/run='+str(runID).zfill(3)+'/scan='+str(scanID).zfill(3)), '*.h5'))
    if sliceNum<0:
        sliceNum = int(motorNum/2)

    try:
        # Getting the h5 files
        fnm = 'p'+str(sliceNum).zfill(4)+'.h5'
        slrpath = os.path.join(slrFolder, 'run='+str(runID).zfill(3), 'scan='+str(scanID).zfill(3), fnm)
        df = pd.read_hdf(slrpath)

        imgpath = os.path.join(imgFolder, 'run='+str(runID).zfill(3), 'scan='+str(scanID).zfill(3), fnm)
        h5 = tb.open_file(imgpath)

        # Getting all tags from the img raw data
        imgTags = h5.root.detector.eh1.mpccd1.image.axis1[:].tolist()
        # Keeping only the shots with imgs recorded
        subdf = df.loc[imgTags]

        # Separate laser on from laser off
        ondf = subdf[subdf[motorKey['laserStatus']]==True]
        offdf = subdf[subdf[motorKey['laserStatus']]==False]

        imgStk = h5.root.detector.eh1.mpccd1.image.block0_values
        if ROI==[]:
            roiSum = np.nansum(np.nansum(imgStk,axis=2),axis=1)
        else:
            roiSum = np.nansum(np.nansum(imgStk[:, ROI[2]:ROI[3], ROI[0]:ROI[1]],axis=2),axis=1)

        statsdf = subdf[['qbpm:oh:qbpm1:sum','qbpm:oh:qbpm2:sum','qbpm:eh1:qbpm1:sum','pd:es:pd1:ch2']]
        statsdf = statsdf.rename(columns={'qbpm:oh:qbpm1:sum': 'pink-sum', 'qbpm:oh:qbpm2:sum': 'mono-up-sum',
                    'qbpm:eh1:qbpm1:sum': 'sample-sum', 'pd:es:pd1:ch2': 'pd'})

        statsdf['roi-sum'] = roiSum

        pd.plotting.scatter_matrix(statsdf,  c=subdf['event_info.FIFTEEN_HERTZ'], diagonal="kde", figsize=[8,6])
        plt.suptitle('Run {}, Scan {}: '.format(runID, scanID)+'p'+str(sliceNum).zfill(4))
        plt.tight_layout()

    except ValueError:
        print('* Empty scan or corrupted file.')

    return



