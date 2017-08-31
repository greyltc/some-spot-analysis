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
    x = np.linspace(0, xRes-1, xRes)
    y = np.linspace(0, yRes-1, yRes)
    x, y = np.meshgrid(x, y)
    
    # calculate some values we'll use for our initial guess
    params = moments(surface2D)
    avg = surface1D.mean()
    max = surface1D.max()
    initial_guess = (max-avg, params[2], params[1], params[3], params[4], 0, avg)
    
    popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), surface1D, p0=initial_guess, maxfev=999000)
    
    fitSurface1D = twoD_Gaussian((x, y), *popt)
    fitSurface2D = fitSurface1D.reshape([yRes,xRes])
    
    # find sum of square of errors for goodness estimation
    ss_res = np.sum((surface1D-fitSurface1D) ** 2)
    ss_tot = np.sum((surface1D - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print("R^2 =", r2)
    print("Amplitude =",popt[0], "counts")
    print("")
    
    if args.drawPlot:
        fig, ax = plt.subplots(1, 1)
        #ax.hold(True)
        ax.imshow(surface2D, cmap=plt.cm.jet, origin='bottom',
                  extent=(x.min(), x.max(), y.min(), y.max()))
        ax.contour(x, y, fitSurface2D, 8, colors='w')
        plt.title(fileName)
        plt.show(block=False)

if args.drawPlot:
    plt.show(block=True)