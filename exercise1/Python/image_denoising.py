# Copyright (C) 2018 Santiago Cortes, Juha Ylioinas
# This software is distributed under the GNU General Public 
# Licence (version 2 or later); please refer to the file 
# Licence.txt, included with the software, for details.

# Preparations
from matplotlib.pyplot import imread
#from skimage.transform import resize
from scipy.misc import imresize
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve as conv2
from scipy.ndimage.filters import convolve1d as conv1
from utils import imnoise, gaussian2, bilateral_filter
import warnings
warnings.filterwarnings('ignore')

## Load test images.
## Note: Must be double precision in the interval [0,1].
im = imread('einsteinpic.jpg') / 255.
im = imresize(im, (256, 256))
#im = resize(im, (256, 256))

## Add noise
## "salt and pepper" noise
imns = imnoise(im, 'salt & pepper', 0.05) * 1.
## zero-mean gaussian noise
imng = im + 0.05*np.random.randn(im.shape[0],im.shape[1])

# Display original and noise corrupted images
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,8))
ax = axes.ravel()
ax[0].imshow(im, cmap='gray')
ax[0].axis('off')
ax[1].imshow(imns, cmap='gray')
ax[1].axis('off')
ax[2].imshow(imng, cmap='gray')
ax[2].axis('off')
plt.tight_layout()
plt.suptitle("Original, 'salt and pepper' and gaussian noise corrupted", fontsize=20)
plt.subplots_adjust(top=1.2)
plt.show()

######################## Simple Separable example #################################
g1 = np.matrix('1 2 1; 2 4 2; 1 2 1')
u, s, v = np.linalg.svd(g1)
print('############## SVD ##############')
print('u=\n', u)
print('s=\n',s)
print('vh=\n',v)

print('############## SVD elements u0, sqrt(s[0]), v0 ##############')
u = u[:,0]
s = np.sqrt(s[0])
v = v[0,:]
print('u0=\n', u)
print('s_sqrt=\n',s)
print('v0=\n',v)

print('############## Resulting kernels ##############')
print('u kernel= \n', s*u)
print('v kernel= \n', s*v)
print('G filter = \n', (s*u) * (s*v))

## Apply Gaussian filter of std 2.5
sigmad = 2.5
g,_,_,_,_,_, = gaussian2(sigmad)
#gflt_imns = conv2(imns, g, mode='reflect')
#gflt_imng = conv2(imng, g, mode='reflect')

## Instead of directly filtering with g, make a separable implementation
## where you use horizontal and vertical 1D convolutions
## That is, replace the above two lines, you can use conv1 instead
## The result should not change.
## See Szeliski's Book chapter 3.2.1 Separable filtering
## Note: numpy's svd gives V as transposed

##--your-code-starts-here--##
u, s, v = np.linalg.svd(g)
u = u[:,0]
s = np.sqrt(s[0])
v = v[0,:]
u_kernel = s*u
v_kernel = s*v
gflt_imns_sep = conv1(imns, u_kernel, axis=0, mode='reflect')
gflt_imns_sep = conv1(gflt_imns_sep, v_kernel, axis=1, mode='reflect')
gflt_imns = gflt_imns_sep #replaced with separable version of filter
gflt_imng_sep = conv1(imng, u_kernel, axis=0, mode='reflect')
gflt_imng_sep = conv1(gflt_imng_sep, v_kernel, axis=1, mode='reflect')
gflt_imng = gflt_imng_sep #replaced with separable version of filter
##--your-code-ends-here--##

# Median filtering is done by extracting a local patch from the input image
# and calculating its median
def median_filter(img, wsize):
    nrows, ncols = img.shape
    output = np.zeros([nrows, ncols])
    
    k = (wsize - 1) / 2
    for i in range(nrows):
        for j in range(ncols):

            # Extract local region
            iMin = np.maximum(i - k, 0)
            iMax = np.minimum(i + k, nrows-1)
            jMin = np.maximum(j - k, 0)
            jMax = np.minimum(j + k, ncols-1)
            img_patch = img[int(iMin):int(iMax+1), int(jMin):int(jMax+1)]
            
            # Calculate the median value, e.g using numpy, from the extracted 
            # local region and store it to output using correct indexing.

            ##--your-code-starts-here--##
            output[i, j] = np.median(img_patch)
            ##--your-code-ends-here--##
    return output

## Apply median filtering, use neighborhood size 5x5
## Store the results in medflt_imns and medflt_imng
## Use the median_filter function  above
    
##--your-code-starts-here--##
medflt_imns = median_filter(imns,5)
medflt_imng = median_filter(imng,5)
##--your-code-ends-here--##

## Apply bilateral filter to each image with window size 11.
## See section 3.3.1 of Szeliski's book
## Use sigma value 2.5 for the domain kernel and 0.1 for range kernel.
## Set bilateral filter parameters. (Replace with your values)
wsize = 11
sigma_d = 2.5
sigma_r = 0.1

bflt_imns = bilateral_filter(imns, wsize, sigma_d, sigma_r)
bflt_imng = bilateral_filter(imng, wsize, sigma_d, sigma_r)

# Display filtering results
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,8))
ax = axes.ravel()
ax[0].imshow(imns, cmap='gray')
ax[0].set_title("Input image - Salt n Pepper")
ax[1].imshow(gflt_imns, cmap='gray')
#ax[1].set_title("Input image")
#ax[1].set_title("Result of gaussian filtering")
ax[2].imshow(medflt_imns, cmap='gray')
#ax[2].set_title("Result of median filtering")
ax[3].imshow(bflt_imns, cmap='gray')
#ax[3].set_title("Result of bilateral filtering")
ax[4].imshow(imng, cmap='gray')
ax[4].set_title("Input image - Gaussian Noise")
ax[5].imshow(gflt_imng, cmap='gray')
ax[5].set_title("Result of gaussian filtering")
ax[6].imshow(medflt_imng, cmap='gray')
ax[6].set_title("Result of median filtering")
ax[7].imshow(bflt_imng, cmap='gray')
ax[7].set_title("Result of bilateral filtering")
plt.suptitle("Filtering results", fontsize=12)
plt.tight_layout()
plt.show()


