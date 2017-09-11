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

parser = argparse.ArgumentParser(description='Spot analysis on image data taken from SDDS files.')
parser.add_argument('--save-image', dest='saveImage', action='store_true', default=False, help="Save data .pgm images to /tmp/pgms/")
parser.add_argument('--draw-plot', dest='drawPlot', action='store_true', default=False, help="Draw data plot or each file processed")
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
    
    logMessages = StringIO()
    parameter = "cycleTime"
    print(parameter,'=',ds.pageData[0]['parameters'][parameter]['value'].rstrip(), file=logMessages)
    parameter = "acqTime"
    print(parameter,'=',ds.pageData[0]['parameters'][parameter]['value'].rstrip(), file=logMessages)
    print("Green Line Cut R^2 =", r2, file=logMessages)
    print("Peak =", amplitude+baseline, file=logMessages)
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

        axes[1,0].plot(rA,lineAData,'r',rA,lineAFit,'k')
        axes[1,0].set_title('Red Line Cut')
        axes[1,0].set_xlabel('Distance from center of spot [pixels]')
        axes[1,0].set_ylabel('Magnitude [counts]')
        axes[1,0].grid(linestyle='--')

        axes[1,1].plot(rB,lineBData,'g',rB,lineBFit,'k')
        axes[1,1].set_title('Green Line Cut')
        axes[1,1].set_xlabel('Distance from center of spot [pixels]')
        axes[1,1].set_ylabel('Magnitude [counts]')
        axes[1,1].grid(linestyle='--')
                
        axes[0,1].axis('off')
        axes[0,1].text(0,0,messages)
        plt.show(block=False)

if args.drawPlot:
    plt.show(block=True)
