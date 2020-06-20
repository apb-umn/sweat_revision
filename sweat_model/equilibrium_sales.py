#
# Structure of this code
#
# 1, import packages
# 2. set parameters and inputs that are necessary to create an Economy instance
# 3. define an objective function for Nelder-Mead
# 4. generate shock variables and save under tmp directory
# 5. run Nelder Mead. All the intermediate steps are written in log files
#

# How to run the code
# make sure that numpy does not automatically do parallel
#
# > export MKL_NUM_THREADS=1
# 
# then, simply run
#
# > python nelder_mead_hy_ns.py
#


import numpy as np
import time
import subprocess
import pickle
import pathlib
import sys

from SCEconomy_sales import Economy, split_shock
from markov import calc_trans, Stationary
from scipy.optimize import minimize

#parameters for sales are stored under parameters_sales
sys.path.insert(0, './parameters_sales/')

case_name=sys.argv[1]
module_name='parameters_'+case_name
### import parameters @@@
exec('from '  + module_name + ' import *')


### log file destination ###
pathlib.Path('./log/').mkdir(exist_ok=True) 
nd_log_file = './log/log.txt'
detailed_output_file = './log/detail.txt'




### initial prices and parameters

# prices_init = np.array([p, rc])
# del p, rc
# GDP_guess it not included here
prices_init = np.array([p, rc, pkap])
del p, rc, pkap

### calibration target

# pure_sweat_share = 0.090 # target
# s_emp_share = 0.33 # target
# xc_share = 0.134 # target
#w*nc/GDP = 0.22



### nelder mead option
tol_nm = 1.0e-4 #if the NM returns a value large than it, the NM restarts



### define objective function for market clearing and calibration ###
dist_min = 10000000.0
econ_save = None

def target(prices):
    global dist_min
    global econ_save
    
    p_ = prices[0]
    rc_ = prices[1]
    pkap_ = prices[2]
    
    

    print('computing for the case p = {:f}, rc = {:f}'.format(p_, rc_), end = ', ')


    #create an Economy instance
    econ = Economy(alpha = alpha,
                   beta = beta,
                   chi = chi,
                   delk = delk,
                   delkap = delkap,
                   eta = eta,
                   grate = grate,
                   la = la,
                   mu = mu,
                   ome = ome,
                   phi = phi,
                   rho = rho,
                   tauc = tauc,
                   taud = taud,
                   taup = taup,
                   taucg  = taucg,
                   theta = theta,
                   veps = veps,
                   vthet = vthet,
                   zeta = zeta,
                   A = A,
                   upsilon = upsilon,
                   varpi = varpi,
                   agrid = agrid,
                   kapgrid = kapgrid,
                   prob = prob,
                   zgrid = zgrid,
                   epsgrid = epsgrid,
                   is_to_iz = is_to_iz,
                   is_to_ieps = is_to_ieps,
                   prob_yo = prob_yo,
                   iota = iota,
                   la_tilde = la_tilde,
                   tau_wo = tau_wo,
                   tau_bo = tau_bo,
                   trans_retire = trans_retire,
                   g = g,
                   yn = yn,
                   xnb = xnb,
                   taub = taub,
                   bbracket = bbracket,
                   psib = psib,
                   taun = taun,
                   nbracket = nbracket,
                   psin = psin,
                   sim_time = sim_time,
                   num_total_pop = num_total_pop,
                   num_suba_inner = num_suba_inner,
                   num_subkap_inner = num_subkap_inner,
                   path_to_data_i_s = path_to_data_i_s,
                   path_to_data_is_o = path_to_data_is_o,
                   path_to_data_sales_shock = path_to_data_sales_shock)
    
    econ.set_prices(p = p_, rc = rc_, pkap = pkap_, kapbar = kapbar)
    #kapbar is fixed now
    
    with open('econ.pickle', mode='wb') as f: pickle.dump(econ, f)
    #with open('econ.pickle', mode='rb') as f: econ = pickle.load(f)
    t0 = time.time()
    result = subprocess.run(['mpiexec', '-n', str(num_core) ,'python', 'SCEconomy_sales.py'],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    t1 = time.time()
    

    f = open(detailed_output_file, 'ab') #use byte mode
    f.write(result.stdout)
    f.close()
    
    print('etime: {:f}'.format(t1 - t0), end = ', ')

    time.sleep(1)

    with open('econ.pickle', mode='rb') as f: econ = pickle.load(f)
    
    moms = econ.moms

    # mom0 = 1. - Ecs/Eys
    # mom1 = 1. - (cc_intermediary + Ecc  + Ex+ (grate + delk)*(kc + Eks) + g + xnb - yn)/yc #modified C-goods market clearing condition
    # mom2 = 1. - (tax_rev - E_transfer - netb)/g                        
    # mom3 = 0.0
    # mom4 = Ens/En
    # mom5 = (p*Eys - (rs+delk)*Eks - w*Ens - Ex)/GDP
    # mom6 = nc
    # mom7 = 1. - EIc
    # mom8 = 1.-pkap_implied/pkap #ER/np.mean(is_s_acq) #(ER - pkap*np.mean(is_s_acq))/yc #
    # mom9 = 1. - kapbar_implied/kapbar                          

    # dist = np.sqrt(moms[0]**2.0 + moms[1]**2.0 + moms[8]**2.0) #if targets are just market clearing
    dist = np.sqrt(moms[0]**2.0 + moms[1]**2.0 + 0.1*moms[8]**2.0) #if targets are just market clearing
    print('dist = {:f}'.format(dist))

    f = open(nd_log_file, 'a')
    f.writelines(str(dist) + ', ' +str(p_) + ', ' + str(rc_) + ', ' + str(pkap_) + ', ' + str(kapbar)  + ', ' +   str(moms[0]) + ', ' + str(moms[1]) + ', ' + str(moms[2]) + ', ' + str(moms[4]) + ', ' + str(moms[5]) + ', ' + str(moms[7]) + ', ' + str(moms[8]) + ', ' + str(moms[9])  +  '\n')
  
    f.close()
    
    if dist < dist_min:
        econ_save = econ
        dist_min = dist
    return dist

if __name__ == '__main__':


    def generate_shock(prob, num_agent, num_time, buffer_time, save_dest, seed, init_state):

        np.random.seed(seed)
        data_rand = np.random.rand(num_agent, num_time+buffer_time)
        data_i = np.ones((num_agent, num_time+buffer_time), dtype = int)
        data_i[:, 0] = init_state
        calc_trans(data_i, data_rand, prob)
        data_i = data_i[:, buffer_time:]
        np.save(save_dest + '.npy' , data_i)
        split_shock(save_dest, num_agent, num_core)

        return data_i


    data_i_s = generate_shock(prob = prob,
                   num_agent = num_total_pop,
                   num_time = sim_time,
                   buffer_time = buffer_time,
                   save_dest = path_to_data_i_s,
                   seed = 0,
                   init_state = 7)

    data_is_o = generate_shock(prob = prob_yo,
                   num_agent = num_total_pop,
                   num_time = sim_time+1,
                   buffer_time = buffer_time,
                   save_dest = path_to_data_is_o,
                   seed = 2,
                   init_state = 0)

    data_sales_shock =  generate_shock(prob = np.array([[1. - pi_sell, pi_sell], [1. - pi_sell, pi_sell]]),
                   num_agent = num_total_pop,
                   num_time = sim_time,
                   buffer_time = buffer_time,
                   save_dest = path_to_data_sales_shock,
                   seed = 10,
                   init_state = 0)

    

    ### check
    f = open(nd_log_file, 'w')
    f.writelines(np.array_str(np.bincount(data_i_s[:,0]) / np.sum(np.bincount(data_i_s[:,0])), precision = 4, suppress_small = True) + '\n')
    f.writelines(np.array_str(Stationary(prob), precision = 4, suppress_small = True) + '\n')
    
    f.writelines(np.array_str(np.bincount(data_is_o[:,0]) / np.sum(np.bincount(data_is_o[:,0])), precision = 4, suppress_small = True) + '\n')
    f.writelines(np.array_str(Stationary(prob_yo), precision = 4, suppress_small = True) + '\n')
    
    f.close()

    del data_i_s, data_is_o, data_sales_shock

    f = open(nd_log_file, 'a')
    f.writelines('dist, p, rc, pkap, kapbar, mom0, mom1, mom2, mom4, mom5, mom7, mom8, mom9\n')        
    f.close()
    


    nm_result = None

    for i in range(5): # repeat up to 5 times
        nm_result = minimize(target, prices_init, method='Nelder-Mead', tol = tol_nm)

        if nm_result.fun < tol_nm: 
            break
        else:
            prices_init = nm_result.x #restart

    f = open(nd_log_file, 'a')
    f.write(str(nm_result))
    f.close()
    

    
    ###calculate other important variables###
    econ = econ_save
    with open('econ.pickle', mode='wb') as f: pickle.dump(econ, f)


    print('')

    print('agrid')
    print(econ.agrid)

    print('kapgrid')
    print(econ.kapgrid)

    print('zgrid')
    print(econ.zgrid)

    print('epsgrid')
    print(econ.epsgrid)

    print('prob')
    print(econ.prob)

    print('prob_yo')
    print(econ.prob_yo)
    print('GDP_guess = ', GDP_guess)    
    print('')    
    

    econ.print_parameters()
    econ.calc_moments()

    
    
