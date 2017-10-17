#!/usr/bin/env python3

# written by grey@christoforo.net

import os
import tempfile
import argparse
import numpy as np
#import mpmath
import matplotlib.pyplot as plt
from sdds import SDDS as ssds
#from math import sqrt
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
parser.add_argument('--save-report', dest='saveReport', action='store_true', default=False, help="Save analysis report .pdfs to /tmp/pdfs/")
parser.add_argument('--draw-plot', dest='drawPlot', action='store_true', default=False, help="Draw data plot or each file processed")
parser.add_argument('--csv-out', type=argparse.FileType('w'), help="Save analysis data to csv file")
parser.add_argument('--correlate-proton-intensities', type=argparse.FileType('r'), help="Read proton intensities from this file")
parser.add_argument('--use-parameter-files', dest='pFiles', type=argparse.FileType('r'), nargs='+', help="Read additional timestamed parameters from these files")
parser.add_argument('input', type=argparse.FileType('rb'), nargs='+', help="File(s) to process")
args = parser.parse_args()

# wheel OD mappings
mapperOD = Object()
mapperOD.one = np.array([0,0.7,1,2])
mapperOD.two = np.array([0,0.3,3,5])

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

fieldNames = ['#File Name','Fit Report', 'Spot Image', 'Device Name', 'First Lamp', 'Second Lamp', 'Cycle Time', 'Cycle Timestamp', 'Acquisition Time', 'Acquisition Timestamp','Time in Cycle [ms]', 'R^2', 'Peak', 'Amplitude', 'Effective Amplitude', 'Center X', 'Center Y', 'Sigma X', 'Sigma Y', 'Rotation', 'Baseline','Screen Select "1"','Filter Select "2"','Optical Density', 'Acq. Counter', 'Acq. Desc.','Observables','OffsetCalSet1','OffsetCalSet2', 'Video Gain']
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

if args.pFiles is not None:
    for p in args.pFiles:
        fullPath = f.name
        with open('example.csv') as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.read(2048))
            csvfile.seek(0)
            reader = csv.DictReader(csvfile,dialect=dialect)
            for row in reader:
                print(row)    
    
# loop through each file in the input
for f in args.input:
    fullPath = f.name
    fileName = os.path.basename(f.name)
    print('Processing', fullPath, '...')
    ds = ssds(f) # use python sdds library to parse the file
    
    paramValues = Object()
    # pull out parameter values and put them in paramValues
    for key,val in ds.pageData[0]['parameters'].items():
        setattr(paramValues,key,val['value'])
    
    arrayValues = Object()
    # pull out array values and put them in arrayValues
    for key,val in ds.pageData[0]['arrays'].items():
        setattr(arrayValues,key,val['value'][0])

    xRes = paramValues.nbPtsInSet1
    yRes = paramValues.nbPtsInSet2
    surface1D = arrayValues.imageSet # grab the image data here
    surface2D = surface1D.reshape([yRes,xRes])   

    # possibly save the surface to a pgm file in /tmp for inspection
    if args.saveImage:
        tmp = tempfile.gettempdir()
        saveDir = tmp + os.path.sep + 'pgms' + os.path.sep
        pgmFile = saveDir+fileName+'.pgm'
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        pgmMax = surface2D.max()
        pgmHeader = 'P2 {:} {:} {:}'.format(xRes,yRes,pgmMax)    
        np.savetxt(pgmFile,surface2D,header=pgmHeader,fmt='%i',comments='')
        print('Saved:', pgmFile)
    else:
        pgmFile = '/dev/null'

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
    
    cycleTime = paramValues.cycleTime.rstrip()[1:-2]
    cycleTime = datetime.strptime(cycleTime,"%Y/%m/%d %H:%M:%S.%f")
    cycleTimeStamp = cycleTime.timestamp() + twoHrTimezoneOffset
    acqTime = paramValues.acqTime.rstrip()[1:-2]
    acqTime = datetime.strptime(acqTime,"%Y/%m/%d %H:%M:%S.%f")
    acqTimeStamp = acqTime.timestamp() + twoHrTimezoneOffset
    
    logMessages = StringIO()
    #parametersToPrint = ('cycleTime', 'acqTime', 'screenSelect', 'filterSelect', 'deviceName')
    
    #for parameter in parametersToPrint:
    #    val = tehParams[parameter]['value']
    #    if type(val) is str:
    #        val = val.rstrip()
    #    print(parameter,'=',val, file=logMessages)
    
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
    
    
    #TODO: need to check mappings for "screen" and "filter" wheels. which is "1" and which is "2"
    OD = mapperOD.one[paramValues.screenSelect] + mapperOD.two[paramValues.filterSelect]
    T = 10**(-OD)
    effectiveBaseline = baseline/T
    effectivePeak = peak/T
    effectiveAmplitude = effectivePeak - effectiveBaseline
    
    if args.drawPlot or args.saveReport:
        fig, axes = plt.subplots(2, 2,figsize=(8, 6), facecolor='w', edgecolor='k')
        fig.suptitle(fileName, fontsize=10)
        axes[0,0].imshow(surface2D, cmap=plt.cm.copper, origin='bottom',
                  extent=(x.min(), x.max(), y.min(), y.max()))
        if len(np.unique(fitSurface2D)) is not 1: # this works around a bug in contour()
            axes[0,0].contour(x, y, fitSurface2D, 3, colors='w')
        else:
            print('Warning: contour() bug avoided')
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
        if args.drawPlot:
            plt.show(block=False)
        if args.saveReport:
            tmp = tempfile.gettempdir()
            saveDir = tmp + os.path.sep + 'pdfs' + os.path.sep
            reportFile = saveDir+fileName+'.report.pdf'
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)            
            plt.savefig(reportFile)
        else:
            reportFile = '/dev/null'
            
        newValues = [fileName,'=HYPERLINK("file://'+reportFile+'")', '=HYPERLINK("file://'+pgmFile+'")', paramValues.deviceName, paramValues.firstLamp, paramValues.secondLamp,cycleTime, cycleTimeStamp, acqTime, acqTimeStamp, arrayValues.acqTimeInCycle, r2, peak, amplitude, effectiveAmplitude, peakPos[0], peakPos[1], sigma[0], sigma[1], theta, baseline, paramValues.screenSelect, paramValues.filterSelect, OD, paramValues.acqCounter, paramValues.acqDesc, paramValues.observables, arrayValues.offsetCalSet1, arrayValues.offsetCalSet2, paramValues.videoGain]
        valuesDict = dict(zip(fieldNames,newValues))
        
        for key,value in valuesDict.items():
            setattr(data,key,np.append(getattr(data, key),value))
        
        if args.csv_out is not None:
            csvWriter.writerow(valuesDict)
            args.csv_out.flush()
    print(messages)

if args.drawPlot:
    plt.show(block=True)
    
if args.correlate_proton_intensities is not None:
    iReader = csv.reader(args.correlate_proton_intensities)
    protonTimeStamps = array.array('d')
    protonIntensities = array.array('d')
    protonTimes = []
    for row in iReader:
        try:
            protonTime = datetime.strptime(row[0],"%Y-%m-%d %H:%M:%S.%f")
            protonTimeStamps.append(protonTime.timestamp())
            protonIntensities.append(float(row[1]))
            protonTimes.append(row[0])
        except:
            pass
    protonTimeStamps = np.array(protonTimeStamps)
    protonIntensities = np.array(protonIntensities)
    proton = array.array('d')
    tDelta = array.array('d')
    tString = []
    nFilesProcessed = len(getattr(data,'Cycle Timestamp'))
    for i in range(nFilesProcessed):
        #TODO: should check for reusage of the same proton time line
        cycleDeltas = getattr(data,'Cycle Timestamp')[i] - protonTimeStamps
        acqDeltas = getattr(data,'Acquisition Timestamp')[i] - protonTimeStamps
        cycleArgMin = np.argmin(np.abs(cycleDeltas))
        acqArgMin = np.argmin(np.abs(acqDeltas))
        #print('Match for cycle      time found at proton entry',cycleArgMin,'with delta',cycleDeltas[cycleArgMin],'s')
        #print('Match for acqusition time found at proton entry',acqArgMin,'with delta',acqDeltas[acqArgMin],'s')
        if cycleArgMin != acqArgMin:
            print("Warning cycle timestamp and acqusition timestamp don't match to the same proton intensity!")
            print("The proton intensity difference is", protonIntensities[cycleArgMin] - protonIntensities[acqArgMin])
        proton.append(protonIntensities[acqArgMin])
        #tDelta.append(cycleDeltas[cycleArgMin])
        tDelta.append(acqDeltas[acqArgMin])
        tString.append(protonTimes[acqArgMin])
    setattr(data,'Proton Intensity',np.array(proton))
    setattr(data,'Delta from Timber timestamp [s]',np.array(tDelta))
    setattr(data,'Timber time',np.array(tString))

    if args.csv_out is not None:
        # rewrite the whole csv_out
        outName = args.csv_out.name
        args.csv_out.close()
        fieldNames.append('Proton Intensity')
        fieldNames.append('Delta from Timber timestamp [s]')
        fieldNames.append('Timber time')
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
