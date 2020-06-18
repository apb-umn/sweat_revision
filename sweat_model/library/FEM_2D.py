
# coding: utf-8

# In[1]:

import numpy as np
import numba as nb

#my library
from orderedTableSearch import locate, locate_grid


# In[2]:

@nb.jit(nopython = True)
def fem2d_eleval(x, y, x0, x1, y0, y1, z0, z1, z2, z3):
# evaluate 2d element
#                
#           ||          ||
#           ||          ||
# y1 ------ z3 -------- z2 -------
#           ||          ||     
#           ||  (x, y)  ||     
#           ||    *     ||     
#           ||          ||
# y0 ------ z0 -------- z1 -------
#           ||          ||
#           ||          ||
#           x0          x1

    
    xi = (2.*x - x0 - x1)/(x1 - x0)
    eta = (2.*y - y0 - y1)/(y1 - y0)
    
    #To safely implement this part
    if x == x0:
        xi = -1.0
    elif x == x1:
        xi = 1.0
    
    if y == y0:
        eta = -1.0
    elif y == y1:
        eta = 1.0
    
    return ( (1. - xi)*(1. - eta)*z0 + 
             (1. + xi)*(1. - eta)*z1 + 
             (1. + xi)*(1. + eta)*z2 +             
             (1. - xi)*(1. + eta)*z3 ) / 4.0

@nb.jit(nopython = True)
def fem2d_peval(xval, yval, xnodes, ynodes, znodes):
    # Evaluate 2d FEM at a given point.
    # znodes is assumed to be array-like znodes = [xdim, ydim]
    # note that, if this is called for each points, it would be slow.
    
    #add a dimension check
    #if len(xnodes) != znodes.shape[0] or len(ynodes) != znodes.shape[1] 
    
    
    ix = locate(xval, xnodes)
    iy = locate(yval, ynodes)
    
    return fem2d_eleval(xval, yval, xnodes[ix], xnodes[ix+1], ynodes[iy], ynodes[iy+1],
                        znodes[ix, iy], znodes[ix+1, iy], znodes[ix+1, iy+1], znodes[ix, iy+1])

#make another function to evaluate on a grid or a line.

#!!!!!!CAUTION!!!!!! THIS FUNCTION MAY BE WRONG
@nb.jit(nopython = True)    
def fem2deval(xypoints, xnodes, ynodes, znodes): #, sort = False):

    #xypoint is array-like: xypoint[0,:] is x, xypoint[1,:] is y
    
    if xypoints.shape[0] != 2:
        print('ERROR: xypoints.shape[0] != 2')
        return None
    
    

    ans = np.zeros(xypoints.shape[1])
    
    
    #numba did not accept the following two arguments
    #ixs = np.empty(xypoints.shape[1]) 
    #iys = np.empty(xypoints.shape[1])

    ixs = locate_grid(xypoints[0,:], xnodes)
    iys = locate_grid(xypoints[1,:], ynodes)
    
    for i in range(len(ans)):
        ans[i] = fem2d_eleval(
            xypoints[0,i], xypoints[1,i], xnodes[ixs[i]], xnodes[ixs[i]+1], ynodes[iys[i]], ynodes[iys[i]+1],
            znodes[ixs[i], iys[i]], znodes[ixs[i]+1, iys[i]], znodes[ixs[i]+1, iys[i]+1], znodes[ixs[i], iys[i]+1])
    
    return ans
    
    


# In[3]:

@nb.jit(nopython = True)    
def fem2deval_mesh(xs, ys, xnodes, ynodes, znodes): #, sort = False):
    
    #I assume xs and ys are strictly accending
    
    num_x = len(xs)
    num_y = len(ys)
    
    ixs = locate_grid(xs, xnodes)
    iys = locate_grid(ys, ynodes)
    
    zs = np.ones((num_x, num_y)) #be careful with the direction
    
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            
            zs[ix, iy] = fem2d_eleval(
                x, y, xnodes[ixs[ix]], xnodes[ixs[ix]+1], ynodes[iys[iy]], ynodes[iys[iy]+1],
                znodes[ixs[ix], iys[iy]], znodes[ixs[ix]+1, iys[iy]],
                znodes[ixs[ix]+1, iys[iy]+1], znodes[ixs[ix], iys[iy]+1])

    return zs
    


