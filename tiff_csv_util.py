#==============================================================================
# Data processing for tr experiments at PAL-XFEL
# 2019-11-23
# Sample: LBCO and TaS2
# 
# Ver. 4
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
import h5py
import time
import itertools

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

# File path, file name, etc.
folder = '/xfel/ffs/dat/ue_191123_FXS/reduced'

# Thresholding bad shots
# Photodiode as i0
i0amp = 1.e-10  # i0 scaling factor

# Regular BG
bgTiff = 'BG_r015.tif'
bgShots = 1.

if not bgTiff=='':
    bgPath = os.path.join(folder, bgTiff)
    bgImg = tifffile.imread(bgPath)/bgShots


#==============================================================================
# Utils
#==============================================================================


def plot2D(img, vmin=-1, vmax=-1):
    if vmin<0.:
        vmin = np.nanpercentile(img, 2)
    else:
        vmin = vmin
    if vmax<0.:
        vmax = np.nanpercentile(img, 98)
    else:
        vmax = vmax

    fig = plt.figure()
    plt.clf()
    plt.imshow(img, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()

    return


def plot1Ds(df, x_axis, tagList):
    '''
    Plotting multiple 1D curves.
    '''
    
    # The total number of lines plotted.
    numLines = len(tagList)
    
    # Generating a color table
    color=iter(cm.rainbow(np.linspace(0,1,numLines)))
    
    # Generating a marker iterator
    # marker = itertools.cycle(('None','o','s','p','*','D','v','^','<', '>'))
    
    plt.figure()
    plt.clf()
    for i in range(len(tagList)):
        plt.plot(x_axis, df[i],label=str(tagList[i]), color=next(color), 
        marker='o', markersize=5)
        # marker=next(marker)
    #plt.legend()
    plt.show()


def sum2DROI(tiffStk, roi):
    '''
    Input:
    tiffStk:    A stack of tiffs
    roi:        A 4-element list of [xmin, xmax, ymin, ymax]
    '''
    subStk = tiffStk[:, roi[2]:roi[3], roi[0]:roi[1]]
    roisum = np.nansum(np.nansum(subStk, axis=2), axis=1)

    return subStk, roisum

def sum1DROI(tiffStk, roi, direction='x'):
    '''
    Input:
    tiffStk:    A stack of tiffs
    roi:        A 4-element list of [xmin, xmax, ymin, ymax]
    direction:  x is horizontal. Default is x.
    '''
    subStk = tiffStk[:, roi[2]:roi[3], roi[0]:roi[1]]
    if direction=='x':
        roisum = np.nansum(subStk, axis=2)
    else:
        roisum = np.nansum(subStk, axis=1)

    return subStk, roisum


#==============================================================================
# Working horses
#==============================================================================

def load_run(runID, scanID=1, plot=True, frameNum=-1, vmin=-1, vmax=-1):
    '''
    Inputs:
    runID:          Run number
    scanID:         The scanID we will look at
    plot:           Plotting the image of a certain slice
    frameNum:       Default is the center slice

    vmin:           Default is 2 percent. The input is in ADU
    vmax:           Default is 98 percent. The input is in ADU

    Returns 2 files:
    dfon:   pandas.DataFrame from the laser on
    dfoff:  pandas.DataFrame from the laser off

    The main goal of this function is to allow selection of a list of ROIs

    2019-11-21
    We noticed that for about 350 motor positions,
    a single tiff is about 1.5 G in size.
    We decided to write single tiffs in the data reduction
    Also this file only allow selection of ROI for a given image to reduce the 
    memory usage.
    '''

    csvNm = 'run'+str(runID).zfill(3)+'_on_s'+str(scanID).zfill(3)+'.csv'
    fpath = os.path.join(folder, 'r'+str(runID).zfill(3), csvNm)
    dfon = pd.read_csv(fpath, index_col=0)

    csvNm = 'run'+str(runID).zfill(3)+'_off_s'+str(scanID).zfill(3)+'.csv'
    fpath = os.path.join(folder, 'r'+str(runID).zfill(3), csvNm)
    dfoff = pd.read_csv(fpath, index_col=0)
    
    if plot:
        if frameNum<0:
            frameNum = int(dfon.shape[0]/2)
        else:
            frameNum = frameNum

        fNm = 'run'+str(runID).zfill(3)+'_on_p'+str(frameNum).zfill(4)+'_s'+str(scanID).zfill(3)+'.tif'
        tiffPath = os.path.join(folder, 'r'+str(runID).zfill(3), fNm)
        tiffOn = tifffile.imread(tiffPath)
        
        fNm = 'run'+str(runID).zfill(3)+'_off_p'+str(frameNum).zfill(4)+'_s'+str(scanID).zfill(3)+'.tif'
        tiffPath = os.path.join(folder, 'r'+str(runID).zfill(3), fNm)
        tiffOff = tifffile.imread(tiffPath)

        if vmin<0.:
            vmin = np.nanpercentile(tiffOff, 2)
        else:
            vmin = vmin

        if vmax<0.:
            vmax = np.nanpercentile(tiffOn, 98)
        else:
            vmax = vmax

        fig, axes = plt.subplots(nrows=1, ncols=2)
        plt.suptitle('Run {}, scan {}: frame {}'.format(runID, scanID, frameNum))
        im = axes[0].imshow(tiffOn, vmin=vmin, vmax=vmax)
        axes[0].set_title('Laser on')

        im = axes[1].imshow(tiffOff, vmin=vmin, vmax=vmax)
        axes[1].set_title('Laser off')
        
        # The two subplots shared the same colorbar scale
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

    return dfon, dfoff


def plotScan(runID, roiList, plot=True, scanID=1, frameNum=-1, vmin=-1, vmax=-1, i0Key=''):
    '''
    Plot a single scan in a single in terms of user-selected ROI sum
    Good for complete scans

    Inputs:
    roiList:        A list of ROIs. Each ROI is a list of
                    [xmin, xmax, ymin, ymax]. Note x is horizontal and y is vertical on the image.

    scanID:         The single scan we would like to focus on
    '''

    if i0Key=='':
        i0Key = 'sample-sum'

    # Getting all dfs
    dfon, dfoff = load_run(runID, scanID=scanID, plot=False)

    # Getting all motor positions
    # Getting all summed shots and summed I_zeros
    motorPos = np.array(dfon.index)
    onShots = np.array(dfon['shots'])
    offShots = np.array(dfoff['shots'])

    onIzero = np.array(dfon[i0Key])
    offIzero = np.array(dfoff[i0Key])


    imgOn = []
    imgOff = []

    for j, m in enumerate(motorPos):
        onNm = 'run'+str(runID).zfill(3)+'_on_p'+str(j+1).zfill(4)
        onNm += '_s'+str(scanID).zfill(3)+'.tif'
        tiffPath = os.path.join(folder, 'r'+str(runID).zfill(3), onNm)
        tempimg = tifffile.imread(tiffPath)

        if not bgTiff=='':
            tempimg = tempimg-bgImg*onShots[j]
            
        imgOn.append(tempimg)

        offNm = 'run'+str(runID).zfill(3)+'_off_p'+str(j+1).zfill(4)
        offNm += '_s'+str(scanID).zfill(3)+'.tif'
        tiffPath = os.path.join(folder, 'r'+str(runID).zfill(3), offNm)
        tempimg = tifffile.imread(tiffPath)

        if not bgTiff=='':
            tempimg = tempimg-bgImg*offShots[j]
            
        imgOff.append(tempimg)

    imgOn = np.array(imgOn)
    imgOff = np.array(imgOff)

    sumOnList = []
    sumOffList = []

    # Starts plotting
    for i, roi in enumerate(roiList):
        subOn, sumOn = sum2DROI(imgOn, roi)
        subOff, sumOff = sum2DROI(imgOff, roi)
        sumOnList.append(sumOn)
        sumOffList.append(sumOff)

        if plot: # One plot per ROI
            if frameNum<0:
                frameNum = int(imgOn.shape[0]/2)
            else:
                frameNum = frameNum

            plt.figure(figsize=[9, 4])
            plt.suptitle('Run {}, Scan {}: ROI {}'.format(runID, scanID, i+1))

            ax1 = plt.subplot2grid((2,10), (0,0), colspan=4)
            ax1.plot(motorPos, sumOn/onIzero*i0amp, 'ro-', label='On', markersize=3)
            ax1.plot(motorPos, sumOff/offIzero*i0amp, 'bo-', label='Off', markersize=3)
            ax1.legend()
            ax1.set_ylabel('ROI sum / I_0')
            plt.setp(ax1.get_xticklabels(), visible=False)
            ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
            
            
            diff = sumOn*offIzero/onIzero/sumOff-1

            ax2 = plt.subplot2grid((2,10), (1,0), colspan=4, sharex=ax1)
            ax2.plot(motorPos, diff, 'ko-', label='On-Off', markersize=3)
            ax2.legend()
            ax2.set_ylabel('Relative\nnormalized\ndifference')
            ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

            if vmin<0:
                vmin = np.nanpercentile(subOn[frameNum], 2)

            if vmax<0:
                vmax = np.nanpercentile(subOff[frameNum], 100)

            ax3 = plt.subplot2grid((2, 10), (0,4), colspan=2, rowspan=2)
            ax3.imshow(subOn[frameNum], vmin=vmin, vmax=vmax)
            ax3.set_title('Laser on')

            ax4 = plt.subplot2grid((2,10), (0,6), colspan=2, rowspan=2)
            ax4.imshow(subOff[frameNum], vmin=vmin, vmax=vmax)
            ax4.set_title('Laser off')

            diffs = i0amp*(subOn[frameNum]/onIzero[frameNum]
                    -subOff[frameNum]/offIzero[frameNum])
            ax5 = plt.subplot2grid((2,10), (0,8), colspan=2, rowspan=2)
            ax5.imshow(diffs)
            ax5.set_title('Laser on-off')

            plt.tight_layout()

    return motorPos, onIzero, offIzero, sumOnList, sumOffList, imgOn, imgOff


def sumScans(runID, roiList, scanIDList, plot=True, frameNum=-1, vmin=-1, vmax=-1, i0Key=''):
    if i0Key=='':
        i0Key = 'sample-sum'

    for j, scanID in enumerate(scanIDList):
        motorPos, onIzero, offIzero, _, _, imgOn, imgOff = plotScan(runID, roiList, plot=False, scanID=scanID, i0Key=i0Key)
        if j:
            sum_onIzero += onIzero
            sum_offIzero += offIzero
            sum_imgOn += imgOn
            sum_imgOff += imgOff
        else:
            sum_onIzero = onIzero
            sum_offIzero = offIzero
            sum_imgOn = imgOn
            sum_imgOff = imgOff

    sumOnList = []
    sumOffList = []

    # Starts plotting
    for i, roi in enumerate(roiList):
        subOn, sumOn = sum2DROI(sum_imgOn, roi)
        subOff, sumOff = sum2DROI(sum_imgOff, roi)
        sumOnList.append(sumOn)
        sumOffList.append(sumOff)

        if plot: # One plot per ROI
            if frameNum<0:
                frameNum = int(imgOn.shape[0]/2)
            else:
                frameNum = frameNum

            plt.figure(figsize=[9, 4])
            plt.suptitle('Run {}: ROI {}'.format(runID, i+1))

            ax1 = plt.subplot2grid((2,10), (0,0), colspan=4)
            ax1.plot(motorPos, sumOn/sum_onIzero*i0amp, 'ro-', label='On', markersize=3)
            ax1.plot(motorPos, sumOff/sum_offIzero*i0amp, 'bo-', label='Off', markersize=3)
            ax1.legend()
            ax1.set_ylabel('ROI sum / I_0')
            plt.setp(ax1.get_xticklabels(), visible=False)
            ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
            
            
            diff = sumOn*sum_offIzero/sum_onIzero/sumOff-1

            ax2 = plt.subplot2grid((2,10), (1,0), colspan=4, sharex=ax1)
            ax2.plot(motorPos, diff, 'ko-', label='On-Off', markersize=3)
            ax2.legend()
            ax2.set_ylabel('Relative\nnormalized\ndifference')
            ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

            if vmin<0:
                vmin = np.nanpercentile(subOn[frameNum], 2)

            if vmax<0:
                vmax = np.nanpercentile(subOff[frameNum], 100)

            ax3 = plt.subplot2grid((2, 10), (0,4), colspan=2, rowspan=2)
            ax3.imshow(subOn[frameNum], vmin=vmin, vmax=vmax)
            ax3.set_title('Laser on')

            ax4 = plt.subplot2grid((2,10), (0,6), colspan=2, rowspan=2)
            ax4.imshow(subOff[frameNum], vmin=vmin, vmax=vmax)
            ax4.set_title('Laser off')

            diffs = i0amp*(subOn[frameNum]/sum_onIzero[frameNum]
                    -subOff[frameNum]/sum_offIzero[frameNum])
            ax5 = plt.subplot2grid((2,10), (0,8), colspan=2, rowspan=2)
            ax5.imshow(diffs)
            ax5.set_title('Laser on-off')

            plt.tight_layout()

    return motorPos, sum_onIzero, sum_offIzero, sumOnList, sumOffList, sum_imgOn, sum_imgOff


def plotROIs(runID, roiList, scanIDList=[1], i0Key=''):
    '''
    Plotting scan results on all ROIs on the same plot
    '''
    if i0Key=='':
        i0Key = 'sample-sum'

    motorPos, onIzero, offIzero, sumOnList, sumOffList, _, _ = sumScans(runID, roiList, scanIDList, plot=False, i0Key=i0Key)

    color=iter(cm.rainbow(np.linspace(0,1,len(roiList))))

    plt.figure(figsize=[9, 4])
    for i in range(len(roiList)):
        c = next(color)
        plt.plot(motorPos, sumOnList[i]/onIzero/np.nanmean(sumOnList[i][:5]), linestyle='-', marker='o', markersize=3, c=c, label=str(roiList[i]))
        plt.plot(motorPos, sumOffList[i]/offIzero/np.nanmean(sumOffList[i][:5]), linestyle='--', c=c)

    plt.legend()
    plt.title('Run {}: ROI sum / {}'.format(runID, i0Key))

    return


