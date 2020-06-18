
# coding: utf-8

# In[1]:

import numpy as np
import numba as nb

#my library
from orderedTableSearch import locate_grid, locate_on_grids, locate


# In[2]:

@nb.jit(nopython = True)
def fem_elmeval(x, x0, x1,para0, para1):
    diff = x1 - x0
    bs0 = (x1 - x)/diff
    bs1 = (x - x0)/diff
    #if bs0 >= 0.0 and bs1 >= 0.0:
    return para0*bs0 + para1*bs1
    #else:
    #    return 0.0
    
@nb.jit(nopython=True)
def fem_peval(xval, nodes, para):
    ie = locate(xval, nodes)
    return fem_elmeval(xval, nodes[ie], nodes[ie+1], para[ie], para[ie+1])    
    
@nb.jit(nopython = True)
def fem_grideval(xvals, nodes, para):
    #I should prohibit extrapolation
            
    N = len(nodes)
    M = len(xvals)

    if N != len(para):
        print('error: N != len(para)')
        return np.ones(M)*np.nan
#         return None #this sometimes causes a problem.


    #xelements = locate_grid(xvals, nodes)
    xelements = locate_on_grids(xvals, nodes)
    ans = np.zeros(M)
    for ix,x in enumerate(xvals):
        ie = xelements[ix]

        ans[ix] = fem_elmeval(x, nodes[ie], nodes[ie+1], para[ie], para[ie+1])

    return ans

#@nb.generated_jit(nopython=True)
#def femeval(xvals, nodes, para):
#    #I should prohibit extrapolation
#    if not hasattr(xvals, "__len__"): #if xvals is scalar
#        ie = locate_grid(xvals, nodes)
#        
#        return fem_elmeval(xvals, nodes[ie], nodes[ie+1], para[ie], para[ie+1])
#    else: #if xvals is actually array-like
#            
#        return fem_grideval(xvals, nodes, para)

@nb.generated_jit(nopython=True)
def femeval(xvals, nodes, para):
    #I should prohibit extrapolation
    if isinstance(xvals, nb.types.Float) or isinstance(xvals, nb.types.Integer):
        
        return lambda xvals, nodes, para: fem_peval(xvals, nodes, para)

    else: #if xvals is actually array-like
            
        return lambda xvals, nodes, para: fem_grideval(xvals, nodes, para)

@nb.jit(nopython = True)
def dy_femeval(x, xnodes):
    """
    derivative wrt parameters (y)
    or returns only the weights on xnodes
    
    input:
    x: a point to be interpolated
    xnode: the grid
    
    output:
    ans (N-dim array)
    
    """
    
    N = len(xnodes)
    
  
    ans = np.zeros(N)
    
    ie = locate(x, xnodes) #need to think of using hunt
    
    #need to add some check
    x0 = xnodes[ie]
    x1 = xnodes[ie+1]
    diff = x1 - x0
    
    ans[ie] = (x1 - x)/diff
    ans[ie+1] = (x - x0)/diff
    
    return ans
    