#==============================================================================
# Data processing for tr experiments at PAL-XFEL
# 2019-11-22
# Sample: LBCO and TaS2
# 
# Ver. 5
# Yue Cao (ycao@colorado.edu)
#
# Usage: in cmd line, go to the scripts folder
#
# python -i PAL_reduction.py $1 $2 ($3 $4 $5)
# $1: name of the scanning motor
# $2: runID (only the nonzero part of the runID needed)
# $3: optional. max scan number. Default is 1
# $4: optional. roiKey, default is 'roi1-sum'
# $5: optional. i0Key, default is 'sample-sum'
# 
# Outputs:
# 1) Two csv files containing all the scalars
# 2) Interactive figures for beam stats
# 3) Two stacks of tiffs with one tiff per motor position for laser on/off
#
# As MPCCD is used, no ADU needs to be subtracted per image.
#
# Updated 2019-11-21
# 1) To limit file size, output two single image tiffs per h5 per scan
# 2) Adding capability for multiple scans (repeats)
#==============================================================================


#==============================================================================
# Importing all plotting packages
#==============================================================================
# import all plotting packages
import matplotlib.pyplot as plt
from matplotlib import colors, cm
plt.style.use('ggplot')
plt.ion()

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

# cmd line inputs
# Remember sys.argv[0] is the name of the python script
motorNm = str(sys.argv[1])
runID = sys.argv[2]

# max scans we want to process
if len(sys.argv)>3:
    maxScanNum = int(sys.argv[3])
else:
    maxScanNum = 1

if len(sys.argv)>4:
    roiKey = str(sys.argv[4])
else:
    roiKey = 'roi1-sum'

if len(sys.argv)>5:
    i0Key = str(sys.argv[5])
else:
    i0Key = 'sample-sum'


# Input file path
slrFolder = '/xfel/ffs/dat/ue_191123_FXS/raw_data/h5/type=measurement/'
slrFolder += 'run='+str(runID).zfill(3)

imgFolder = '/xfel/ffs/dat/ue_191123_FXS/raw_data/h5/type=raw/'
imgFolder += 'run='+str(runID).zfill(3)


# Output file name
outfolder = '/xfel/ffs/dat/ue_191123_FXS/reduced'

# All the csv and tiff are written in a subfolder and all the plots go to the png subfolder
if not os.path.isdir(os.path.join(outfolder, 'r'+str(runID).zfill(3))):
    os.mkdir(os.path.join(outfolder, 'r'+str(runID).zfill(3)))

# output nm
onNmOut = 'run'+str(runID).zfill(3)+'_on_'
offNmOut = 'run'+str(runID).zfill(3)+'_off_'

# Thresholding bad shots
# Photodiode as i0
i0thres = -1 # i0 threshold
i0amp = 1.e-10  # i0 scaling factor

# No background needed

# ADU to single photon conversion
# ADU thresholding
# These numbers were extracted from experiments
# ADU_thre = 150  # At 8.8 keV
# ADU_2_ph = 180  # At 8.8 keV

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

tstart = time.time()

print('****************************************')

for scanID in range(maxScanNum):
    print('* Start processing Run '+str(runID)+', Scan '+str(scanID+1))

    # Getting the total number of motor positions
    motorNum = len(glob.glob1(os.path.abspath(slrFolder+'/scan='+str(scanID+1).zfill(3)), '*.h5'))
    cenSlice = int(motorNum/2)
    print ('* Total steps: '+str(motorNum))

    # Creating the keys to be extracted
    outcolumns = list(motorRBV.keys())
    outcolumns.extend(list(qbpm.keys()))
    outcolumns.extend(list(mpccd.keys()))
    outcolumns.extend(list(qbpm_pos.keys()))
    outcolumns.extend(list(mpccd_pos.keys()))

    # Creating the output
    motors = []
    csvon = {}
    csvoff = {}
    aduon = []
    aduoff = []
    # phon = []
    # phoff = []

    for col in outcolumns:
        csvon[col] = []
        csvoff[col] = []

    csvon['shots'] = []
    csvoff['shots'] = []


    # Compiling the data
    for i in range(motorNum):
        try:
            # Getting the h5 files
            fnm = 'p'+str(i+1).zfill(4)+'.h5'
            slrpath = os.path.join(slrFolder, 'scan='+str(scanID+1).zfill(3), fnm)
            df = pd.read_hdf(slrpath)

            imgpath = os.path.join(imgFolder, 'scan='+str(scanID+1).zfill(3), fnm)
            h5 = tb.open_file(imgpath)

            # Getting the motor position
            motors.append(np.array(df[motorKey[motorNm]])[0])

            # Getting all tags from the img raw data
            imgTags = h5.root.detector.eh1.mpccd1.image.axis1[:].tolist()
            # Keeping only the shots with imgs recorded
            subdf = df.loc[imgTags]
            # print(len(subdf.index))   # Getting the number of recorded imgs

            # Filter by good shots - not working now
            #gooddf = subdf[subdf[qbpm[i0Key]]>i0thres]
            #subpd = subpd[subpd[qbpm[i0Key]]>i0thres]

            # Separate laser on from laser off
            ondf = subdf[subdf[motorKey['laserStatus']]==True]
            offdf = subdf[subdf[motorKey['laserStatus']]==False]
            csvon['shots'].append(ondf.shape[0])
            csvoff['shots'].append(offdf.shape[0])

            
            # Fig. 99: linearity check for the center motor position
            if i==cenSlice:
                plt.figure()
                plt.plot(ondf[qbpm[i0Key]], ondf[mpccd[roiKey]], 'ro', label='On')
                plt.plot(offdf[qbpm[i0Key]], offdf[mpccd[roiKey]], 'bo', label='Off')
                plt.xlabel(qbpm[i0Key])
                plt.ylabel(roiKey)
                plt.legend()
                plt.title('Run {}, Scan {}: {} vs. {} at {}={:.2f}'.format(runID, scanID+1, roiKey, i0Key, motorNm, df[motorKey[motorNm]].iloc[0]))
                pngPath = os.path.join(outfolder, 'png', 'run'+str(runID).zfill(3)+'_s'+str(scanID+1).zfill(3)+'_fig99.png')
                plt.savefig(pngPath)


                # statsdf = df[['qbpm:oh:qbpm1:sum','qbpm:oh:qbpm2:sum','qbpm:eh1:qbpm1:sum','pd:es:pd1:ch2','detector:eh1:mpccd1:ROI1_stat.sum']]
                # statsdf = statsdf.rename(columns={'qbpm:oh:qbpm1:sum': 'pink-sum', 'qbpm:oh:qbpm2:sum': 'mono-up-sum',
                #           'qbpm:eh1:qbpm1:sum': 'sample-sum', 'pd:es:pd1:ch2': 'pd',
                #           'detector:eh1:mpccd1:ROI1_stat.sum': 'roi1-sum'})
                statsdf = subdf[['qbpm:oh:qbpm2:sum','qbpm:eh1:qbpm1:sum','detector:eh1:mpccd1:ROI1_stat.sum']]
                statsdf = statsdf.rename(columns={'qbpm:oh:qbpm2:sum': 'mono-up-sum',
                            'qbpm:eh1:qbpm1:sum': 'sample-sum', 'detector:eh1:mpccd1:ROI1_stat.sum': 'roi1-sum'})

                
                #pd.plotting.scatter_matrix(ondf[[qbpm['pink-sum'], qbpm['mono-up-sum'], qbpm['sample-sum'], qbpm['pd1-ch2'], mpccd[roiKey]]], diagonal="kde")
                pd.plotting.scatter_matrix(statsdf,  c=subdf['event_info.FIFTEEN_HERTZ'], diagonal="kde", figsize=[8,6])
                plt.suptitle('Run {}, Scan {}: {}={:.2f}'.format(runID, scanID+1, motorNm, df[motorKey[motorNm]].iloc[0]))
                plt.tight_layout()
                pngPath = os.path.join(outfolder, 'png', 'run'+str(runID).zfill(3)+'_s'+str(scanID+1).zfill(3)+'_fig100.png')
                plt.savefig(pngPath)


            # Reducing the scalars.
            # Beam currents and det counts should be summed.
            # Beam positions and peak positions should be averaged.

            for key in motorRBV.keys():
                try:
                    csvon[key].append(ondf[motorRBV[key]].iloc[0])
                    csvoff[key].append(offdf[motorRBV[key]].iloc[0])
                except:
                    csvon[key].append(np.nan)
                    csvoff[key].append(np.nan)

            for key in qbpm.keys():
                csvon[key].append(np.nansum(ondf[qbpm[key]]))
                csvoff[key].append(np.nansum(offdf[qbpm[key]]))

            for key in mpccd.keys():
                csvon[key].append(np.nansum(ondf[mpccd[key]]))
                csvoff[key].append(np.nansum(offdf[mpccd[key]]))

            for key in qbpm_pos.keys():
                csvon[key].append(np.nanmean(ondf[qbpm_pos[key]]))
                csvoff[key].append(np.nanmean(offdf[qbpm_pos[key]]))

            for key in mpccd_pos.keys():
                csvon[key].append(np.nanmean(ondf[mpccd_pos[key]]))
                csvoff[key].append(np.nanmean(offdf[mpccd_pos[key]]))

            # Reducing the tiffs
            # The tiffs will be summed.
            # Creating empty images
            imgshape = h5.root.detector.eh1.mpccd1.image.block0_values.shape
            tempaduon = np.zeros((imgshape[1], imgshape[2])).astype(np.int32)
            tempaduoff = np.zeros((imgshape[1], imgshape[2])).astype(np.int32)

            for j, imgTag in enumerate(imgTags):
                tempadu = h5.root.detector.eh1.mpccd1.image.block0_values[j]

                # tempph = np.where(tempadu>ADU_thre, tempadu, 0)
                # tempph = np.round(tempph/ADU_2_ph).astype(np.int32)

                if subdf.loc[imgTag][motorKey['laserStatus']]:
                    tempaduon += tempadu
                    # tempphon += tempph
                else:
                    tempaduoff += tempadu
                    # tempphoff += tempph

            # aduon.append(tempaduon)
            # aduoff.append(tempaduoff)
            # phon.append(tempphon)
            # phoff.append(tempphoff)

            # New change in 2019-11-21
            # Smaller files are written
            tempaduon = np.array(tempaduon).astype(np.int32)
            tempNm = onNmOut+'p'+str(i+1).zfill(4)+'_s'+str(scanID+1).zfill(3)+'.tif'
            fpath = os.path.join(outfolder, 'r'+str(runID).zfill(3), tempNm)
            tifffile.imsave(fpath, data=tempaduon)

            tempaduoff = np.array(tempaduoff).astype(np.int32)
            tempNm = offNmOut+'p'+str(i+1).zfill(4)+'_s'+str(scanID+1).zfill(3)+'.tif'
            fpath = os.path.join(outfolder, 'r'+str(runID).zfill(3), tempNm)
            tifffile.imsave(fpath, data=tempaduoff)

            print('* File p'+str(i+1).zfill(4)+' processed.')
        except:
            print('* File p'+str(i+1).zfill(4)+' corrupted.')

            csvon['shots'].append(0.)
            csvoff['shots'].append(0.)

            for col in outcolumns:
                if col in list(qbpm.keys()):
                    csvon[col].append(0.)
                    csvoff[col].append(0.)
                elif col in list(mpccd.keys()):
                    csvon[col].append(0.)
                    csvoff[col].append(0.)
                else:
                    csvon[col].append(np.nan)
                    csvoff[col].append(np.nan)

            tempaduon = np.zeros((imgshape[1], imgshape[2])).astype(np.int32)
            tempNm = onNmOut+'p'+str(i+1).zfill(4)+'_s'+str(scanID+1).zfill(3)+'.tif'
            fpath = os.path.join(outfolder, 'r'+str(runID).zfill(3), tempNm)
            tifffile.imsave(fpath, data=tempaduon)

            tempaduoff = np.zeros((imgshape[1], imgshape[2])).astype(np.int32)
            tempNm = offNmOut+'p'+str(i+1).zfill(4)+'_s'+str(scanID+1).zfill(3)+'.tif'
            fpath = os.path.join(outfolder, 'r'+str(runID).zfill(3), tempNm)
            tifffile.imsave(fpath, data=tempaduoff)

    # output csv
    try:
        csvon = pd.DataFrame(data=csvon, index=motors)
        csvoff = pd.DataFrame(data=csvoff, index=motors)

        outonpath = os.path.join(outfolder, 'r'+str(runID).zfill(3), onNmOut+'s'+str(scanID+1).zfill(3)+'.csv')
        csvon.to_csv(outonpath)
        print('* Scalar file for laser on saved as '+onNmOut+'s'+str(scanID+1).zfill(3)+'.csv')

        outoffpath = os.path.join(outfolder, 'r'+str(runID).zfill(3), offNmOut+'s'+str(scanID+1).zfill(3)+'.csv')
        csvoff.to_csv(outoffpath)
        print('* Scalar file for laser off saved as '+offNmOut+'s'+str(scanID+1).zfill(3)+'.csv')

        # aduon = np.array(aduon)
        # aduoff = np.array(aduoff)

        # fpath = os.path.join(outfolder, aduonNmOut)
        # tifffile.imsave(fpath, data=aduon)
        # print('* Tiff file in ADU for laser on saved as '+aduonNmOut)

        # fpath = os.path.join(outfolder, aduoffNmOut)
        # tifffile.imsave(fpath, data=aduoff)
        # print('* Tiff file in ADU for laser off saved as '+aduoffNmOut)

        # phon = np.array(phon)
        # phoff = np.array(phoff)

        # fpath = os.path.join(outfolder, phonNmout)
        # tifffile.imsave(fpath, data=phon)
        # print('* Tiff file in photon cts for laser on saved as '+phonNmout)

        # fpath = os.path.join(outfolder, phoffNmout)
        # tifffile.imsave(fpath, data=phoff)
        # print('* Tiff file in photon cts for laser off saved as '+phoffNmout)


        # Making plots
        # Fig. 1: qbpm by the motor
        plt.figure(figsize=[10, 8])
        plt.clf()
        ax1 = plt.subplot(211)
        plt.plot(csvon.index, csvon[i0Key], 'ro-', label='On')
        plt.plot(csvoff.index, csvoff[i0Key], 'bo-', label='Off')
        ax1.legend()
        ax1.set_ylabel(qbpm[i0Key])
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        ax1.set_title('Run {}: stats by the motor position'.format(runID))

        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(csvon.index, csvon['shots'], 'ro-', label='On')
        plt.plot(csvoff.index, csvoff['shots'], 'bo-', label='Off')
        ax2.legend()
        ax2.set_ylabel('Saved shots')
        ax2.set_xlabel(motorNm)
        ax2.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))

        pngPath = os.path.join(outfolder, 'png', 'run'+str(runID).zfill(3)+'_s'+str(scanID+1).zfill(3)+'_fig1.png')
        plt.savefig(pngPath)

        # Fig. 2: scanning on vs. scanning off
        plt.figure(figsize=[10, 8])
        plt.clf()
        ax1 = plt.subplot(211)
        plt.plot(csvon.index, csvon[roiKey], 'ro-', label='On')
        plt.plot(csvoff.index, csvoff[roiKey], 'bo-', label='Off')
        ax1.legend()
        ax1.set_ylabel(roiKey)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))
        ax1.set_title('Run {}: {} vs. {}'.format(runID, roiKey, motorNm))

        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(csvon.index, csvon[roiKey]*i0amp/csvon[i0Key], 'ro-', label='On')
        plt.plot(csvoff.index, csvoff[roiKey]*i0amp/csvoff[i0Key], 'bo-', label='Off')
        ax2.legend()
        ax2.set_ylabel('{} / {}'.format(roiKey, i0Key))
        ax2.set_xlabel(motorNm)
        ax2.ticklabel_format(axis='both', style='sci', scilimits=(-2,2))

        pngPath = os.path.join(outfolder, 'png', 'run'+str(runID).zfill(3)+'_s'+str(scanID+1).zfill(3)+'_fig2.png')
        plt.savefig(pngPath)

        # Fig. 3: percentage change of laser on vs. laser off
        diff = csvon[roiKey]*csvoff[i0Key]/csvon[i0Key]/csvoff[roiKey]-1.
        plt.figure(figsize=[10,8])
        plt.plot(csvon.index, diff, 'ko-')
        plt.xlabel(motorNm)
        plt.ylabel('Normalized relative change')
        plt.title('Run {}: on - off'.format(runID))

        pngPath = os.path.join(outfolder, 'png', 'run'+str(runID).zfill(3)+'_s'+str(scanID+1).zfill(3)+'_fig3.png')
        plt.savefig(pngPath)

    except ValueError:
        print('* Empty scan or corrupted file.')


tend = time.time()

# Cmd prints
print('* Total time: {:10.2f}'.format(tend-tstart)+' seconds.')
print('****************************************')


plt.ioff()

