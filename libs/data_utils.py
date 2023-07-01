


import numpy as np
import math
import pdb
import scipy.misc
import scipy
import sys
import pickle
import random
from libs.image_utils import *

def create_dataset(img_size,data_dir,n_images,shape='disk',sigma=0.8,n_monte_carlo=800):

    for i in range(0,n_images):
        theta = 2*np.pi*np.random.uniform()
        _r = img_size[0]/4.0
        params = np.asarray( (_r,theta))
        img_out = monte_carlo_shape(img_size,shape,params,sigma,n_monte_carlo)
        write_image(img_out,data_dir+'/'+shape+'_img_'+str(i).zfill(5)+'.png')
    return

# params : 
# disk : radius
# disk shifted : (radius,x_shift,y_shift)
# ellipse : (y_radis,x_radius)
# triange : rotation
def monte_carlo_shape(img_size,shape,params,sigma,N):

    x_list = range( int(-img_size[1]/2.0+1), int(img_size[1]/2.0+1))
    y_list = range( int(-img_size[0]/2.0+1), int(img_size[0]/2.0+1))
    [X,Y] = np.meshgrid( y_list , x_list )

    img_out = np.zeros(img_size)
    for i in range(0,N):
        q = sigma * np.random.randn(2,1)
        x_temp = X+q[0]
        y_temp = Y+q[1]
        
        if ( shape == 'disk'):
            img_temp = (x_temp**2+y_temp**2) < params^2
        elif ( shape == 'disk_shifted'):
            img_temp = ((x_temp-params[1])**2+(y_temp-params[2])**2) < params[0]**2
        elif ( shape == 'ellipse'):
            a = params[0]
            b = params[1]
            theta = params[2]
            # carry out the inverse transformation to align the axes
            A = np.asarray([[np.cos(-theta) , -np.sin(-theta)] , [np.sin(-theta) , np.cos(-theta)] ])
            
            coords_transformed = np.matmul(A , [np.ravel(x_temp), np.ravel(y_temp)])
            x_temp = np.reshape(coords_transformed[0,:],img_size)
            y_temp = np.reshape(coords_transformed[1,:],img_size)
            
            img_temp = ( (x_temp/a)**2 + (y_temp/b)**2 ) <= 1.0
        elif ( shape == 'triangle'):
            x_centre = 0.0
            y_centre = 0.0
            r = params[0]
            theta = params[1]
            img_temp = in_triangle(x_temp, y_temp, x_centre, y_centre, r , theta).astype(float)
        else:
            print('Unknown shape')
        img_out = img_out + img_temp

    img_out = img_out/N

    return img_out


def in_triangle(x, y, x_centre, y_centre, r, theta):
    #establish the apexes of the triangle
    #   B
    # A   C
    xA = x_centre - r*np.cos(np.pi/6.0 + theta)
    yA = y_centre + r*np.sin(np.pi/6.0 + theta)
    xB = x_centre - r*np.sin(theta)
    yB = y_centre - r*np.cos(theta)
    xC = x_centre + r*np.sin(np.pi/3.0 + theta)
    yC = y_centre + r*np.cos(np.pi/3.0 + theta)

    #check if the point is on the right side of each halfplane
    #halfplane AB
    return np.logical_and(np.logical_and(in_halfplane(x, y, xA, yA, xB, yB,1), in_halfplane(x, y, xB, yB, xC, yC,1) ) , 
            in_halfplane(x, y, xC, yC, xA, yA,1) )


def in_halfplane(x, y, xA, yA, xB, yB, direction):
    sign = 0

    position = np.sign((xB - xA) * (y - yA) - (yB - yA) * (x - xA))

    return position == direction