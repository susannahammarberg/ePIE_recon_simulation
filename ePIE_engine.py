# -*- coding: utf-8 -*-
"""
Created on 2020 01 29

@author: Sanna

Based on "A phase retrieval algorithm for shifting illumination" J.M. Rodenburg and H.M.L Faulkner, [App. Phy. Lett.  85.20 (2004)]
"""

import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def ePIE(n, diffSet, probe, objectSize, positions,illu_indy,illu_indx ):

    # size of probe and diffraction patterns
    ysize, xsize = probe.shape
        
    # initialize object. make sure it can hold complex numbers
    objectFunc = np.ones(objectSize, dtype=np.complex64)
    
    # initalize that illuminated part of the object
    objectIlluminated = np.ones(shape=(ysize, xsize),dtype=np.complex64)
    
    # initialize algorithm wave fields (fourier and real) 
    g = np.zeros(( ysize, xsize),dtype=np.complex64)
    gprime = np.zeros(( ysize, xsize),dtype=np.complex64)
    G = np.zeros(( ysize, xsize),dtype=np.complex64)
    Gprime = np.zeros(( ysize, xsize),dtype=np.complex64)

    # define iteration counter for outer loop
    k = 0
    
    # figure for animation
    fig = plt.figure()
    
    # Initialize vector for animation data
    ims = []
    
    # initialize vector for error calculation
    # sse = np.zeros(shape=(n,1))
    
    # idex for iterating through the diffraction patterns
    diffSetIndex = 0
    
    # Start of ePIE iterations
    while k < n:
        # Start of inner loop: (where you iterate through all probe positions R)
        for pos in positions:
        
             # Cut out the part of the image that is illuminated at R(=(ypos,xpos)
             objectIlluminated = objectFunc[pos[0]+illu_indy, pos[1]+illu_indx ]

             # Guessed wave field from the object at position R 
             g = objectIlluminated * probe      
        
             # fft the wave field at position R to Fourier space
             G = (fft.fftshift(fft.fft2(g)))
            
             # make |PSI| confirm with the diffraction pattern from R
             Gprime = np.sqrt(abs(diffSet[diffSetIndex]))*np.exp(1j*np.angle(G))
             
             # inverse Fourier transform  
             gprime =  fft.ifft2(fft.ifftshift(Gprime))
     
             # update the TOTAL object function with the illuminated part
             # The update should be the differens of the last iteration and the new one 
             alpha = 1 #higher value == faster change
             objectFunc[pos[0]+illu_indy, pos[1]+illu_indx] = objectIlluminated + alpha*(gprime-g) * np.conj(probe)  / (np.max(abs(probe))**2)   #probe* annars blir det att man delar med massa nollor
             
             # update probe function
             beta = 0.01 #higher value == faster change
             probe = probe + beta * (gprime-g) * np.conj(objectIlluminated)/ (np.max(abs(objectIlluminated))**2)
             
             ########################            
             # Apply further constraints:
             # These 2 constraints are for transmission                
             ########################
            
             # constrain object amplitude to 1
#             temp_Oamp = abs(objectFunc)
#             temp_Oamp[temp_Oamp>1] = 1
#             temp = np.angle(objectFunc)
#             objectFunc = temp_Oamp * np.exp(1j* temp)
#            
#             #constraint object phase to negative or 0
#             temp_Ophase = np.angle(objectFunc)
#             temp_Ophase[temp_Ophase>0] = 0
#             objectFunc = abs(objectFunc) * np.exp(1j* temp_Ophase)
##            
             # animate
             im = plt.imshow(np.angle(objectFunc), animated=True)
             ims.append([im])

             diffSetIndex += 1
    
    
        k += 1        
        print('Iteration %d starts'%k)
        
        #SSE[0][k] =  sum(sum(abs(Gprime - diffSet[3] )**2 ))
        
        #reset inner loop index
        diffSetIndex = 0
       
    # End of iterations
    print('End of iterations')
    
    #todo calculate average error
    err = np.sum( abs(diffSet[-1]**2 - G**2 )**2)
    
    # animate reconstruction 
    ani = animation.ArtistAnimation(fig, ims, interval=150, blit=True,repeat_delay=3000)
    
    # save animation
    # .mp4 requires mencoder or ffmpeg to be installed
    #ani.save('ePIE.gif')
    #print('Saving animation')
    
    #show animation
    plt.show()
    
    return objectFunc, probe, err

if __name__ == 'main':
    
    print('main prog')