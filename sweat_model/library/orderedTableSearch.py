
# coding: utf-8

# In[1]:

import numpy as np
import numba as nb


# In[2]:

@nb.jit(nopython=True)
def locate(x, grid, btm = 0, up = None):
    if up is None:
        up = len(grid)-1
    
    """
    from Numerical Repipes 2nd ed.
    input
    x: a value to be evaluated
    grid: a monotonically ordered grid
    
    return grid number 0,1,....,len(grid) -2
    """
    
    N = len(grid)
    
    
    
    if grid[N-1] > grid[0]:
        if x <= grid[0]:
            return 0
        elif x >= grid[N-1]:
            return N-2
    elif grid[N-1] < grid[0]:
        if x <= grid[N-2]:
            return N-2
        elif x >= grid[0]:
            return 0
        
    else:
        print('error: grid[N-1] == grid[0]')
    
    #Golden search
    mid = 0
    while up - btm > 1: #if not done
        mid = int((up+btm)/2) #math.floor? 
#         if (grid[N-1] > grid[0]) == (x > grid[mid]):
        if (grid[N-1] > grid[0]) == (x > grid[mid]):

            btm = mid
        else:
            up = mid

    if up - btm < 1: 
        print('locate, error: up - btm < 1, up = ', up, ', btm = ', btm, '.')
        
#         print('error: up - btm < 1')

    return btm   


@nb.jit(nopython=True)
def hunt(x, grid, init_btm):
    N = len(grid)
    ascnd = (grid[N-1] > grid[0])
    btm = init_btm
    up = 0#None does not work for numba
    
    if init_btm < 0 or init_btm > N-2:
        return locate(x,grid)
    else:
        inc = 1 #increment
        if (x > grid[btm]) is ascnd:
            while True:
                up = btm + inc
                #print('up: ',up)
                if up > N-2:
                    up = N-1
                    #print('up: ',up)
                    break
                elif (x > grid[up]) is ascnd:
                    btm = up
                    inc = inc + inc
                else:
                    break
        else:
            up = btm
            #print('up: ',up)
            while True:
                btm = up - inc
                #print('btm: ',btm)
                if btm < 0:
                    btm = 0
                    #print('btm: ',btm)
                    break
                elif (x < grid[btm]) is ascnd:
                    up = btm
                    inc = inc + inc
                    #print('up: ',up)
                else:
                    break
        #print('btm: ',btm)
        #print('up: ',up)        
        return locate(x, grid, btm, up)
    

    
#@nb.jit#(nopython=True)#hasattr is not compatible with numba

#this should accept an array-like with len == 1
@nb.jit(nopython=True)
def locate_on_grids(xvals, grid, init_btm = 0):
    M = len(xvals)
    ans = np.zeros(M, dtype = np.int64)
    ans[0]= hunt(xvals[0], grid, init_btm)

    for ix in range(1,M):
        ans[ix] = hunt(xvals[ix], grid, ans[ix-1])
        # ans[ix] = locate(xvals[ix], grid, ans[ix-1]) #this was a typo. cant set btm = ans[ix-1]
        # i donno why but it is faster

    return ans

@nb.generated_jit(nopython=True)
def locate_grid(xvals, grid, init_btm = 0, return_nparray = False):
    

    if isinstance(xvals, nb.types.Float) or isinstance(xvals, nb.types.Integer) : #if xvals is scalar
        #here, locate is converted into np.array
        #if you need just an interger, use locate instead.
        return lambda xvals, grid, init_btm, return_nparray: np.array(locate(xvals, grid))
        
        #if return_nparray is True:
        #    return lambda xvals, grid, init_btm, return_nparray: np.array(locate(xvals, grid))
        #else:
        #    return lambda xvals, grid, init_btm, return_nparray: locate(xvals, grid)
    
    else: #arraylike #maybe I should check this is arraylike
        return lambda xvals, grid, init_btm, return_nparray: locate_on_grids(xvals, grid, init_btm)


# In[4]:

if __name__ == '__main__':
    
    import time
    
    bignodes = np.linspace(-10, 100, 1000000)
    xvals = np.linspace(-1, 50, 1000000)
    t1 = time.time()
    hunt(6.6, bignodes, 3)

    t2 = time.time()

    print(' {} seconds'.format(t2 - t1))


    #comapred the speed
    t1 = time.time()
    M = len(xvals)
    ans1 = np.zeros(M, dtype = np.int64)

    for ix, x in enumerate(xvals):
        ans1[ix] = locate(x, bignodes)

    t2 = time.time()

    print(' {} seconds'.format(t2 - t1))
    
    
    
    t1 = time.time()
    M = len(xvals)
    ans2 = np.zeros(M, dtype = np.int64)

    ans2[0] = hunt(x, bignodes, 0)
    for ix in range(M):
        x = xvals[ix]
        ans2[ix] = hunt(x, bignodes, ans2[ix-1]+1)

    t2 = time.time()

    print(' {} seconds'.format(t2 - t1))
    
    
    #comapred the speed
    t1 = time.time()

    ans3 = locate_grid(xvals, bignodes)

    t2 = time.time()

    print(' {} seconds'.format(t2 - t1))


# In[ ]:



