#!/usr/bin/env python3

# written by grey@christoforo.net

import os
import tempfile
import argparse
import numpy as np
import mpmath
import matplotlib.pyplot as plt
from sdds import SDDS as ssds
from math import sqrt
from scipy import optimize as opt
from scipy import interpolate
from io import StringIO
from datetime import datetime
import csv
import array

class Object(object):
    pass

parser = argparse.ArgumentParser(description='Spot analysis on image data taken from SDDS files.')
parser.add_argument('--save-image', dest='saveImage', action='store_true', default=False, help="Save data .pgm images to /tmp/pgms/")
parser.add_argument('--draw-plot', dest='drawPlot', action='store_true', default=False, help="Draw data plot or each file processed")
parser.add_argument('--csv-out', type=argparse.FileType('w'), help="Save analysis data to csv file")
parser.add_argument('--correlate-proton-intensities', type=argparse.FileType('r'), help="Read proton intensities from this file")
parser.add_argument('input', type=argparse.FileType('rb'), nargs='+', help="File(s) to process")
args = parser.parse_args()

# returns a 1d vector representation of the height at position xy of a 2d gaussian surface where
# amplitude = gaussian peak height
# xo,yo is the peak's position
# sigma_x, sigma_y are the x and y standard deviations
# theta is the rotaion angle of the gaussian
# and
# offset is the surface's height offset from zero
def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x = xy[0]
    y = xy[1]
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

# calculates a 2d gaussian's height, x, y position and x and y sigma values from surface height data
def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

fieldNames = ['#File Name', 'Spot Image', 'Cycle Time', 'Acquisition Time', 'R^2', 'Peak', 'Amplitude', 'Center X', 'Center Y', 'Sigma X', 'Sigma Y', 'Rotation', 'Baseline','Screen Select','Filter Select','Acq. Counter', 'Acq. Desc.','Observables','OffsetCalSet1','OffsetCalSet2']
data = Object()
for fieldName in fieldNames:
    setattr(data, fieldName, np.array([]))

if args.csv_out is not None:
    filename,file_ext = os.path.splitext(args.csv_out.name)
    if file_ext != '.csv':
        print("Error: csv file name must end in .csv")
        exit(1)
    csvWriter = csv.DictWriter(args.csv_out,fieldnames=fieldNames)
    csvWriter.writeheader()
    args.csv_out.flush()
    
# loop through each file in the input
for f in args.input:
    fullPath = f.name
    fileName = os.path.basename(f.name)
    print('Processing', fullPath, '...')
    ds = ssds(f) # use python sdds library to parse the file
    xRes = ds.pageData[0]['parameters']['nbPtsInSet1']['value']
    yRes = ds.pageData[0]['parameters']['nbPtsInSet2']['value']
    surface1D = ds.pageData[0]['arrays']['imageSet']['value'][0] # grab the image data here
    surface2D = surface1D.reshape([yRes,xRes])

    screenSelect = ds.pageData[0]['parameters']['screenSelect']['value']
    filterSelect = ds.pageData[0]['parameters']['filterSelect']['value']
    acqCounter = ds.pageData[0]['parameters']['acqCounter']['value']
    acqDesc = ds.pageData[0]['parameters']['acqDesc']['value']
    observables = ds.pageData[0]['parameters']['observables']['value']
    offsetCalSet1 = ds.pageData[0]['arrays']['offsetCalSet1']['value'][0]
    offsetCalSet2 = ds.pageData[0]['arrays']['offsetCalSet2']['value'][0]

    # possibly save the surface to a pgm file in /tmp for inspection
    if args.saveImage:
        tmp = tempfile.gettempdir()
        saveDir = tmp + os.path.sep + 'pgms' + os.path.sep
        saveFile = saveDir+fileName+'.pgm'
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        pgmMax = surface2D.max()
        pgmHeader = 'P2 {:} {:} {:}'.format(xRes,yRes,pgmMax)    
        np.savetxt(saveFile,surface2D,header=pgmHeader,fmt='%i',comments='')
        print('Saved:', saveFile)
    else:
        saveFile = '/dev/null'

    # Create x and y grid
    xv = np.linspace(0, xRes-1, xRes)
    yv = np.linspace(0, yRes-1, yRes)
    x, y = np.meshgrid(xv, yv)
    
    # calculate some values we'll use for our initial guess
    params = moments(surface2D)
    avg = surface1D.mean()
    max = surface1D.max()
    # [amplitude, peakX, peakY, sigmaX, sigmaY, theta(rotation angle), avg (background offset level)]
    initial_guess = (max-avg, params[2], params[1], params[4], params[3], 0, avg)
    
    popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), surface1D, p0=initial_guess, maxfev=999000)
    
    fitSurface1D = twoD_Gaussian((x, y), *popt)
    fitSurface2D = fitSurface1D.reshape([yRes,xRes])
    
    # find sum of square of errors for goodness estimation
    #residuals = surface1D - fitSurface1D
    #ss_res = np.sum(residuals**2)
    #ss_tot = np.sum((surface1D - np.mean(surface1D)) ** 2)
    #r2 = 1 - (ss_res / ss_tot)
    
    # the fit parameters
    amplitude = popt[0]
    theta = popt[5]
    peakPos = (popt[1],popt[2])
    sigma = (popt[3],popt[4])    
    baseline = popt[6]
    
    # calculate evaluation lines
    length = (4*sigma[0],4*sigma[1])

    nPoints = 100
    nSigmas = 4 # line length, number of sigmas to plot in each direction
    rA = np.linspace(-nSigmas*sigma[0],nSigmas*sigma[0],nPoints) # radii (in polar coords for line A)
    AX = rA*np.cos(theta-np.pi/4) + peakPos[0] # x values for line A
    AY = rA*np.sin(theta-np.pi/4) + peakPos[1] # y values for line A
    
    rB = np.linspace(-nSigmas*sigma[1],nSigmas*sigma[1],nPoints) # radii (in polar coords for line B)
    BX = rB*np.cos(theta+np.pi/4) + peakPos[0] # x values for line B
    BY = rB*np.sin(theta+np.pi/4) + peakPos[1] # y values for line B    

    f = interpolate.interp2d(xv, yv, surface2D) # linear interpolation for data surface

    lineAData = np.array([float(f(px,py)) for px,py in zip(AX,AY)])
    lineAFit = np.array([float(twoD_Gaussian((px, py), *popt)) for px,py in zip(AX,AY)])

    lineBData = np.array([float(f(px,py)) for px,py in zip(BX,BY)])
    lineBFit = np.array([float(twoD_Gaussian((px, py), *popt)) for px,py in zip(BX,BY)])

    residuals = lineBData - lineBFit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((lineBData - np.mean(lineBData)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    twoHrTimezoneOffset = 7200 # seconds
    
    cycleTime = ds.pageData[0]['parameters']["cycleTime"]['value'].rstrip()[1:-2]
    cycleTime = datetime.strptime(cycleTime,"%Y/%m/%d %H:%M:%S.%f")
    cycleTimeStamp = cycleTime.timestamp() + twoHrTimezoneOffset
    acqTime = ds.pageData[0]['parameters']["acqTime"]['value'].rstrip()[1:-2]
    acqTime = datetime.strptime(acqTime,"%Y/%m/%d %H:%M:%S.%f")
    acqTimeStamp = acqTime.timestamp() + twoHrTimezoneOffset
    
    logMessages = StringIO()
    print('cycleTime =', cycleTime, file=logMessages)
    print('acqTime =', acqTime, file=logMessages)
    print("Green Line Cut R^2 =", r2, file=logMessages)
    peak = amplitude+baseline
    print("Peak =", peak, file=logMessages)
    print("====Fit Parameters====", file=logMessages)
    print("Amplitude =", amplitude, file=logMessages)
    print("Center X =", peakPos[0], file=logMessages)
    print("Center Y =", peakPos[1], file=logMessages)
    print("Sigma X =", sigma[0], file=logMessages)
    print("Sigma Y =", sigma[1], file=logMessages)
    print("Rotation (in rad) =", theta, file=logMessages)
    print("Baseline =", baseline, file=logMessages)
    print("", file=logMessages)
    logMessages.seek(0)
    messages = logMessages.read()
    print(messages)
    
    newValues = [fileName, '=HYPERLINK("file://'+saveFile+'")', cycleTimeStamp, acqTimeStamp, r2, peak, amplitude, peakPos[0], peakPos[1], sigma[0], sigma[1], theta, baseline, screenSelect,filterSelect,acqCounter,acqDesc,observables,offsetCalSet1,offsetCalSet2]
    valuesDict = dict(zip(fieldNames,newValues))
    
    for key,value in valuesDict.items():
        setattr(data,key,np.append(getattr(data, key),value))
    
    if args.csv_out is not None:
        csvWriter.writerow(valuesDict)
        args.csv_out.flush()
    
    if args.drawPlot:
        fig, axes = plt.subplots(2, 2,figsize=(8, 6), facecolor='w', edgecolor='k')
        fig.suptitle(fileName, fontsize=10)
        axes[0,0].imshow(surface2D, cmap=plt.cm.copper, origin='bottom',
                  extent=(x.min(), x.max(), y.min(), y.max()))
        axes[0,0].contour(x, y, fitSurface2D, 3, colors='w')
        axes[0,0].plot(AX,AY,'r') # plot line A
        axes[0,0].plot(BX,BY,'g') # plot line B
        axes[0,0].set_title("Image Data")
        axes[0,0].set_ylim([y.min(), y.max()])
        axes[0,0].set_xlim([x.min(), x.max()])

        axes[1,0].plot(rA,lineAData,'r',label='Data')
        axes[1,0].plot(rA,lineAFit,'k',label='Fit')
        axes[1,0].set_title('Red Line Cut')
        axes[1,0].set_xlabel('Distance from center of spot [pixels]')
        axes[1,0].set_ylabel('Magnitude [counts]')
        axes[1,0].grid(linestyle='--')
        handles, labels = axes[1,0].get_legend_handles_labels()
        axes[1,0].legend(handles, labels)        

        axes[1,1].plot(rB,lineBData,'g',label='Data')
        axes[1,1].plot(rB,lineBFit,'k',label='Fit')
        axes[1,1].set_title('Green Line Cut')
        axes[1,1].set_xlabel('Distance from center of spot [pixels]')
        axes[1,1].set_ylabel('Magnitude [counts]')
        axes[1,1].grid(linestyle='--')
        handles, labels = axes[1,1].get_legend_handles_labels()
        axes[1,1].legend(handles, labels)           
                
        axes[0,1].axis('off')
        axes[0,1].text(0,0,messages)
        plt.show(block=False)

if args.drawPlot:
    plt.show(block=True)
    
if args.correlate_proton_intensities is not None:
    iReader = csv.reader(args.correlate_proton_intensities)
    protonTimes = array.array('d')
    protonIntensities = array.array('d')
    for row in iReader:
        try:
            protonTime = datetime.strptime(row[0],"%Y-%m-%d %H:%M:%S.%f")
            protonTimes.append(protonTime.timestamp())
            protonIntensities.append(float(row[1]))
        except:
            pass
    protonTimes = np.array(protonTimes)
    protonIntensities = np.array(protonIntensities)
    proton = array.array('d')
    tDelta = array.array('d')
    nFilesProcessed = len(getattr(data,'Cycle Time'))
    for i in range(nFilesProcessed):
        cycleDeltas = getattr(data,'Cycle Time')[i] - protonTimes
        acqDeltas = getattr(data,'Acquisition Time')[i] - protonTimes
        cycleArgMin = np.argmin(np.abs(cycleDeltas))
        acqArgMin = np.argmin(np.abs(acqDeltas))
        #print('Match for cycle      time found at proton entry',cycleArgMin,'with delta',cycleDeltas[cycleArgMin],'s')
        #print('Match for acqusition time found at proton entry',acqArgMin,'with delta',acqDeltas[acqArgMin],'s')
        if cycleArgMin != acqArgMin:
            print("Warning cycle timestamp and acqusition timestamp don't match to the same proton intensity!")
            print("The proton intensity difference is", protonIntensities[cycleArgMin] - protonIntensities[acqArgMin])
        proton.append(protonIntensities[acqArgMin])
        tDelta.append(acqDeltas[acqArgMin])
    setattr(data,'Proton Intensity',np.array(proton))
    setattr(data,'Delta from Timber timestamp [s]',np.array(tDelta))

    if args.csv_out is not None:
        # rewrite the whole csv_out
        outName = args.csv_out.name
        args.csv_out.close()
        fieldNames.append('Proton Intensity')
        fieldNames.append('Delta from Timber timestamp [s]')        
        with open(outName, 'w', newline='') as csvfile:
            newWriter = csv.DictWriter(csvfile,fieldnames=fieldNames)
            newWriter.writeheader()
            for i in range(nFilesProcessed):
                values = []
                for field in fieldNames:
                    values.append(getattr(data,field)[i])
                valuesDict = dict(zip(fieldNames,values))
                newWriter.writerow(valuesDict)
            csvfile.flush()

if args.csv_out is not None:
    args.csv_out.close()