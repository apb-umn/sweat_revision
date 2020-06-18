import numpy as np
import numba as nb


@nb.jit(nopython = True)
def unravel_index_nb(*args):
    
    length = len(args)

    
    if length < 3:
        print('error in unravel_index_nb: check the input')
        return None
    
    
    tmp = args[0]
    
    ans = np.ones(length-1, dtype = np.int64)
    
    product = 1
    for i in range(length - 1):
        product = product * args[i+1]
        
    if tmp < 0. or tmp > product - 1:
        print('error in unravel: num is beyond the range.')
        return None

    
    for i in range(length - 2):
        product = product / args[i+1]
        ans[i] = tmp // product
        tmp = tmp - ans[i] * product
       
    
    ans[-1] = tmp % args[length - 1]
        
    
    return ans