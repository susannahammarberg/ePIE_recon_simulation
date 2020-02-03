# -*- coding: utf-8 -*-
"""
Created on 2020 01 28 try to get a ePIE simulation to work... with my old code

Working!

Loads an image. Saves image as object phase. Creates a flat field probe.
Defines scanning postions on the object. 
Creates an exit wave Y= object * probe in each scanning positions. Propagates 
the exit wave with a fft. Reconstructs the objec and probe by 
running the ePIE algorithm.

@author: Susanna Hammarberg

"""


import sys   #to collect system path ( to collect function from another directory)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import fft
import numpy as np
from ePIE_engine import ePIE  


#----------------------------------
# create sim probe and object.
# load image and set to object phase or amplitude values
#----------------------------------

def image():
    
    #probe = mpimg.imread('circle.png')
    probe = np.zeros((256,256))
    #probe = np.ones((128,128))
    #probe[110:147, 110:147] = 1
    probe[100:157, 100:157] = 1
    
    obj = mpimg.imread('fruit.jpg')

    return probe,obj

probe, obj = image()


# collapse to 3d matrix
try:
    probe = np.array(np.sum(probe, axis=2))
except Exception:
    print('probe not 3d. OK')
try:
    obj = np.array(np.sum(obj, axis=2))
except Exception:
    print('obj not 3d. OK')


# pad image and make them complex 
padding = 0
probe = np.pad(probe, ((padding,padding),(padding,padding)), 'constant', constant_values=(0, 0))
obj = np.pad(obj, ((padding,padding),(padding,padding)), 'constant', constant_values=(0, 0))

# normalize amplitude to 1
obj = obj /obj.max()


#make object only phase (intstead of only amplitude)
#(comment out for amplitude object)
obj = 1.0 * np.exp(obj*1j) 


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, gridspec_kw={'wspace': 0}); plt.suptitle('initial object and probe'); 
ax1.imshow((np.abs(obj)),cmap='jet'); ax1.set_title('amplitude');  ax2.imshow(np.angle(obj)); ax2.set_title('Phase')
ax3.imshow(abs(probe)); ax3.set_title('Amplitude'); ax4.imshow(np.angle(probe),cmap='jet'); ax4.set_title('Phase')

#%%
#------------------------------------------------
#define scanning positions in terms of pixels
#------------------------------------------------

origin = 20
Ny = 10 ; Nx = 10
# overlapp shild be 60% stepsize = int(probeSize * 0.4)   #for 60% overlapp
dy = 10 ; dx = dy #same
positions = np.zeros((Ny*Nx,2),dtype=np.int32)
#y positions in 1st colum
positions[:, 1] = np.tile(np.arange(Nx)*dx, Ny)
#y positions in 1st colum
positions[:, 0] = np.repeat(np.arange(Ny)*dy , Nx)  
positions += origin

print('Scanning positions defined')

#----------------------------------
# make a set of diffraction patters  
#----------------------------------

# a list of diffrac tion patterns
diff_set = []
# the indices for the area that is illuminated by the probe (should have the probes shape)
illu_indy, illu_indx = np.indices((probe.shape) )

# create a diffraction pattern at each probe position
for pos in positions:   
    # Propogate (obj*probe) to far field with a Fourier transform, then calculate the absolute square
    diff_set.append(abs(fft.fftshift(fft.fft2( obj[pos[0]+illu_indy, pos[1]+illu_indx]*probe)))**2 )
  

print('Diffraction patterns created')
plt.figure();plt.title('Example of diffraction pattern')
plt.imshow(abs(np.log10(diff_set[-1])))    
#----------------------------------
# Run reconstruction
#----------------------------------

#define object shape (needed  for ePIE function)
object_shape = obj.shape   # ex (34,34)

# number of iterations in ePIE
k = 3
# run the algoritm
objectFunc, probe_ret, err = ePIE(k,diff_set,probe,object_shape, positions,illu_indy,illu_indx) 
print('Error of last frame',err)

#----------------------------------
# image the result
#----------------------------------

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, gridspec_kw={'wspace': 0}); plt.suptitle('final object and probe'); 
ax1.imshow((np.abs(objectFunc)),cmap='jet'); ax1.set_title('amplitude obj');  ax2.imshow(np.angle(objectFunc)); ax2.set_title('Phase obj')
ax3.imshow(abs(probe_ret)); ax3.set_title('Amplitude probe'); ax4.imshow(np.angle(probe_ret),cmap='jet'); ax4.set_title('Phase probe')