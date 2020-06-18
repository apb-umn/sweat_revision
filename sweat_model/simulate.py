#
# Structure of this code
#
# 1, import packages
# 2. set parameters and inputs that are necessary to create an Economy instance
# 3. save an instance as a pickle
# 4. run a main simulation code (this is a different code)
# 5. retrieve simulation result from a pickle
# 6. save simulation results
#
#


if __name__ == '__main__':
    
    #
    # import necessary packages
    #
    
    import numpy as np
    import time
    import subprocess
    import pickle
    import pathlib
    import sys


    from SCEconomy import Economy, split_shock
    from markov import calc_trans, Stationary


    # import list of parameters
    
    case_name=sys.argv[1]
    module_name='parameters_'+case_name
    ### import parameters @@@
    exec('from '  + module_name + ' import *')

    
    ### log file destination ###
    save_path = './' + 'save_data_'+ case_name + '/'
    log_output_file = save_path + 'paras_mom_' + case_name + '.txt'

    print(log_output_file)

    # create a directory to save files if it does not exist.
    pathlib.Path(save_path).mkdir(exist_ok=True) 

    
    ### generate shocks and save them ###
    #save and split shocks for istate


    
    def generate_shock(prob, num_agent, num_time, buffer_time, save_dest, seed, init_state):

        np.random.seed(seed)
        data_rand = np.random.rand(num_agent, num_time+buffer_time)
        data_i = np.ones((num_agent, num_time+buffer_time), dtype = int)
        data_i[:, 0] = init_state
        calc_trans(data_i, data_rand, prob)
        data_i = data_i[:, buffer_time:]
        np.save(save_dest + '.npy' , data_i)
        split_shock(save_dest, num_agent, num_core)

    pathlib.Path('./tmp').mkdir(exist_ok=True) 
    generate_shock(prob = prob,
                   num_agent = num_total_pop,
                   num_time = sim_time,
                   buffer_time = buffer_time,
                   save_dest = path_to_data_i_s,
                   seed = 0,
                   init_state = 7)

    generate_shock(prob = prob_yo,
                   num_agent = num_total_pop,
                   num_time = sim_time+1,
                   buffer_time = buffer_time,
                   save_dest = path_to_data_is_o,
                   seed = 2,
                   init_state = 0)


    print('Solving the model with the given prices...')
    print('Do not simulate more than one models at the same time...')
    print('GDP_guess = ', GDP_guess)


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
                   path_to_data_is_o = path_to_data_is_o)

                   

    # set prices
    econ.set_prices(p = p, rc = rc)


    # save an Economy object and pass it to another program
    with open('econ.pickle', mode='wb') as f: pickle.dump(econ, f)

    # run the main code
    t0 = time.time()
    result = subprocess.run(['mpiexec', '-n', str(num_core), 'python', 'SCEconomy.py'], stdout=subprocess.PIPE)
    t1 = time.time()


    # write output in the log file
    f = open(log_output_file, 'wb') #use byte mode
    f.write(result.stdout)
    f.close()


    # receive a simulation result
    with open('econ.pickle', mode='rb') as f: econ = pickle.load(f)

        
    # print parameters
    econ.print_parameters()
    
    ## #calculate other important variables ###
    econ.calc_sweat_eq_value() 
    econ.calc_age()
    econ.simulate_other_vars()
    econ.calc_moments()    
    econ.save_result_csv(dir_path_save = save_path) # save simulation result under ./save_data/ by default
    econ.save_moments(file_path=save_path)
    exec(open('convert_parameters_to_csv.py').read())

    
    
