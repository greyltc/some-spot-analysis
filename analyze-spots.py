#!/usr/bin/env python3

# written by grey@christoforo.net

import argparse
import numpy
import mpmath
import matplotlib.pyplot as plt
from sdds import SDDS as ssds
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt

parser = argparse.ArgumentParser(description='Spot analysis on image data taken from SDDS files.')

parser.add_argument('input', type=argparse.FileType('rb'), nargs='+', help="Single file or list of SDDS files to process")

args = parser.parse_args()

for f in args.input:
    ds = ssds(f)
    yRes = ds.pageData[0]['parameters']['nbPtsInSet1']['value']
    xRes = ds.pageData[0]['parameters']['nbPtsInSet2']['value']
    imageData = ds.pageData[0]['arrays']['imageSet']['value'][0]
    imageData = imageData.reshape([xRes,yRes])
    imageData = imageData/imageData.max()
    image = imageData
    
    print(f.name)
    #fig, ax = plt.subplots()
    #ax.imshow(imageData)
    #ax.axis('off')  # clear x- and y-axes
    #plt.show()
    image_gray = imageData
    #print('break')
    blobs_log = blob_log(image_gray, max_sigma=50, num_sigma=10, threshold=.1)
    
    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    
    blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
    
    blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)
    
    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ['yellow', 'lime', 'red']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
              'Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)
    
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True,
                             subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel()
    
    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image, interpolation='nearest')
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()
    
    plt.tight_layout()
    plt.show()    
