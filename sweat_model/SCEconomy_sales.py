#import Yuki's library in the directory ./library
import sys
sys.path.insert(0, './library/')


import numpy as np
import numba as nb
###usage
###@nb.jit(nopython = True)


#my library
#import
#from FEM import fem_peval #1D interpolation
from orderedTableSearch import locate, hunt
from FEM import femeval
from FEM_2D import fem2d_peval, fem2deval_mesh
from markov import Stationary
from ravel_unravel_nb import unravel_index_nb


from mpi4py import MPI
comm = MPI.COMM_WORLD #retreive the communicator module
rank = comm.Get_rank() #get the rank of the process
size = comm.Get_size() #get the number of processes

import time

# %matplotlib inline
# import matplotlib as mpl
# mpl.rc("savefig",dpi=100)
# from matplotlib import pyplot as plt

class Economy:
    
    """
    class Economy stores all the economic parameters, prices, computational parameters, 
    grids information for a, kappa, eps, z, and shock transition matrix
    """
    
    def __init__(self,
                 alpha = None,                 
                 beta = None,
                 iota = None, # iota \in [0, 1], which is 
                 chi = None,
                 delk = None,
                 delkap = None,
                 eta = None,
                 g = None,
                 grate = None,
                 la = None,
                 la_tilde = None, #preserved sweat_capital if it is taken over
                 tau_wo = None, #c-productivity decline if s/he is old
                 tau_bo = None, #s-productiity decline is s/he is old
                 mu = None,
                 ome = None,
                 phi = None,
                 rho = None,
                 tauc = None,
                 taud = None,
                 taup = None,
                 taucg = None, #capital gain tax for R
                 theta = None,
                 trans_retire = None, #retirement benefit which is not included in tran
                 veps = None,
                 vthet = None,
                 xnb = None,
                 yn = None,
                 zeta = None,
                 agrid = None,
                 kapgrid = None,
                 epsgrid = None,
                 zgrid = None,
                 prob = None, #transition matrix for (eps, z)
                 prob_yo = None, #young-old transition matrix
                 prob_sales = None,
                 is_to_iz = None,
                 is_to_ieps = None,
                 num_suba_inner = None,
                 num_subkap_inner = None,
                 sim_time = None,
                 num_total_pop = None,
                 A = None,
                 upsilon = None,
                 varpi = None,
                 
                 path_to_data_i_s = None,
                 path_to_data_is_o = None,
                 path_to_data_sales_shock = None,

                 taun = None,
                 psin = None,
                 psin_fixed = None,
                 nbracket = None,
                 nbracket_fixed = None,                 
                 scaling_n = None,
                 
                 taub = None,
                 psib = None,
                 psib_fixed = None,                 
                 bbracket = None,
                 bbracket_fixed = None,                 
                 scaling_b = None):

        
        self.__set_default_parameters__()
        
        #set the parameters if designated
        #I don't know how to automate these lines
        if alpha is not None: self.alpha = alpha
        if beta is not None: self.beta = beta
        if iota is not None: self.iota = iota #added
        if chi is not None: self.chi = chi
        if delk is not None: self.delk = delk    
        if delkap is not None: self.delkap = delkap
        if eta is not None: self.eta = eta
        if g is not None: self.g = g
        if grate is not None: self.grate = grate 
        if la is not None: self.la = la
        if la_tilde is not None: self.la_tilde = la_tilde #added
        if tau_wo is not None: self.tau_wo = tau_wo #added
        if tau_bo is not None: self.tau_bo = tau_bo #added
        if mu is not None: self.mu = mu 
        if ome is not None: self.ome = ome 
        if phi is not None: self.phi = phi
        if rho is not None: self.rho = rho
        if tauc is not None: self.tauc = tauc
        if taucg is not None: self.taucg = taucg        
        if taud is not None: self.taud = taud
        if taup is not None: self.taup = taup
        if theta is not None: self.theta = theta
        if trans_retire is not None: self.trans_retire = trans_retire #added
        if veps is not None: self.veps = veps
        if vthet is not None: self.vthet = vthet
        if xnb is not None: self.xnb = xnb
        if yn is not None: self.yn = yn
        if zeta is not None: self.zeta = zeta
        if agrid is not None: self.agrid = agrid
        if kapgrid is not None: self.kapgrid = kapgrid
        if epsgrid is not None: self.epsgrid = epsgrid
        if zgrid is not None: self.zgrid = zgrid
        if prob is not None: self.prob = prob
        if prob_yo is not None: self.prob_yo = prob_yo
        if prob_sales is not None: self.prob_sales = prob_sales
        if is_to_iz is not None: self.is_to_iz = is_to_iz
        if is_to_ieps is not None: self.is_to_ieps = is_to_ieps
        if num_suba_inner is not None: self.num_suba_inner = num_suba_inner
        if num_subkap_inner is not None: self.num_subkap_inner = num_subkap_inner
        if sim_time is not None: self.sim_time = sim_time
        if num_total_pop is not None: self.num_total_pop = num_total_pop
        if A is not None: self.A = A
        if upsilon is not None: self.upsilon = upsilon
        if varpi is not None: self.varpi = varpi        

        if path_to_data_i_s is not None: self.path_to_data_i_s = path_to_data_i_s
        if path_to_data_is_o is not None: self.path_to_data_is_o = path_to_data_is_o
        
        if taun is not None: self.taun = taun
        if psin is not None: self.psin = psin
        if psin_fixed is not None: self.psin_fixed = psin_fixed
        if nbracket is not None: self.nbracket = nbracket
        if nbracket_fixed is not None: self.nbracket_fixed = nbracket_fixed                
        if scaling_n is not None: self.scaling_n = scaling_n

        if taub is not None: self.taub = taub
        if psib is not None: self.psib = psib
        if psib_fixed is not None: self.psib_fixed = psib_fixed        
        if bbracket is not None: self.bbracket = bbracket
        if bbracket_fixed is not None: self.bbracket_fixed = bbracket_fixed  
        if scaling_b is not None: self.scaling_b = scaling_b        
        

        self.__set_nltax_parameters__()
        self.__set_implied_parameters__()
    
    def __set_default_parameters__(self):

        """
        Load the baseline value
        """
        
        self.__is_price_set__ = False
        self.alpha    = 0.3
        self.beta     = 0.98
        self.iota     = 1.0 
        self.chi      = 0.0 #param for borrowing constarint
        self.delk     = 0.05
        self.delkap   = 0.05 
        self.eta      = 0.42
        self.g        = 0.234 
        self.grate    = 0.02 #gamma, growth rate for detrending
        self.la       = 0.7 #lambda
        self.la_tilde = 0.1 #added lambda_tilde
        self.tau_wo   = 0.5 
        self.tau_bo   = 0.5 
        self.mu       = 1.5 
        self.ome      = 0.6 #omega
        self.phi      = 0.15 
        self.rho      = 0.01
        self.tauc     = 0.06
        self.taud     = 0.14
        self.taup     = 0.30
        self.taucg    = 0.20
        self.theta    = 0.41
        self.trans_retire = 0.48 #added retirement benefit
        self.veps     = 0.4
        self.vthet    = 0.4
        self.xnb      = 0.185
        self.yn       = 0.451
        self.zeta     = 1.0
        self.sim_time = 1000
        self.num_total_pop = 100_000
        self.A        = 1.577707121233179 #this should give yc = 1 (approx.) z^2 case
        self.upsilon  = 0.5
        self.varpi    = 0.5

        # self.path_to_data_i_s = './input_data/data_i_s.npy'
        self.path_to_data_i_s = './tmp/data_i_s'
        self.path_to_data_is_o = './tmp/data_is_o'
        self.path_to_data_sales_shock = './tmp/data_sales_shock'        


        #nonlinear tax parameters
        self.taub = np.array([.137, .185, .202, .238, .266, .280])
        self.bbracket = np.array([0.150, 0.319, 0.824, 2.085, 2.930])
        self.scaling_b = 1.0
        self.psib_fixed = 0.03 #0.15
        self.bbracket_fixed = 2
        self.psib = None 

        self.taun = np.array([.2930, .3170, .3240, .3430, .3900, .4050, .4080, .4190])
        self.nbracket = np.array([.1760, .2196, .2710, .4432, 0.6001, 1.4566, 2.7825])
        self.scaling_n = 1.0
        self.psin_fixed = 0.03 #0.15
        self.nbracket_fixed = 5
        self.psin = None         
        

        # #grid information
        # self.agrid = np.load('./input_data/agrid.npy')
        # self.kapgrid = np.load('./input_data/kapgrid.npy')
        # self.epsgrid = np.load('./input_data/epsgrid.npy')    
        # self.zgrid = np.load('./input_data/zgrid.npy')
        

        #conbined exogenous states
        #s = (e,z)'

        #pi(t,t+1)
        # self.prob = np.load('./DeBacker/prob_epsz.npy') #default transition is taken from DeBakcer
        # self.prob_yo = np.array([[44./45., 1./45.], [3./45., 42./45.]]) #[[y -> y, y -> o], [o -> y, o ->o]]

        # self.prob = np.load('./input_data/transition_matrix.npy')
        # self.prob_yo = np.array([[0.5, 0.5], [0.5, 0.5]]) #[[y -> y, y -> o], [o -> y, o ->o]]

        # ####do we need this one here?
        # #normalization to correct rounding error.
        # for i in range(prob.shape[0]):
        #     prob[i,:] = prob[i,:] / np.sum(prob[i,:])

        # self.is_to_iz = np.load('./input_data/is_to_iz.npy')
        # self.is_to_ieps = np.load('./input_data/is_to_ieps.npy')
        
        #computational parameters
        self.num_suba_inner = 20
        self.num_subkap_inner = 30

        self.s_age = None
        self.c_age = None
        self.sind_age = None
        self.cind_age = None
        self.y_age = None
        self.o_age = None


    def __set_nltax_parameters__(self):
        
        # from LinearTax import get_consistent_phi

        
        # self.bbracket = self.bbracket * self.scaling_b
        # #set transfer term if not provided
        # if self.psib is None:
        #     self.psib = get_consistent_phi(self.bbracket, self.taub, self.psib_fixed, self.bbracket_fixed) # we need to set the last two as arguments

        tmp = self.bbracket
        self.bbracket = np.zeros(len(tmp)+2)

        self.bbracket[0] = -np.inf
        self.bbracket[-1] = np.inf
        self.bbracket[1:-1] = tmp[:]

        
        
        # self.nbracket = self.nbracket * self.scaling_n
        
        # #set transfer term if not provided
        # if self.psin is None:
        #     self.psin = get_consistent_phi(self.nbracket, self.taun, self.psin_fixed, self.nbracket_fixed) # we need to set the last two as arguments

        tmp = self.nbracket
        self.nbracket = np.zeros(len(tmp)+2)        
        self.nbracket[0] = -np.inf
        self.nbracket[-1] = np.inf
        self.nbracket[1:-1] = tmp[:]


        # self.psin = self.psin * self.scaling_n
        
        

        
    def __set_implied_parameters__(self):
        #length of grids
        self.num_a = len(self.agrid)
        self.num_kap = len(self.kapgrid)
        self.num_eps = len(self.epsgrid)
        self.num_z = len(self.zgrid)
        self.num_s = self.prob.shape[0]

        
        #implied parameters
        self.nu = 1. - self.alpha - self.phi
        self.bh = self.beta*(1. + self.grate)**(self.eta*(1. - self.mu))  #must be less than one.
        self.varrho = (1. - self.alpha - self.nu)/(1. - self.alpha) * self.vthet / (self.vthet + self.veps)

    
        if self.bh >= 1.0 or self.bh <= 0.0:
            print('Error: bh must be in (0, 1) but bh = ', self.bh)

        self.prob_st = Stationary(self.prob)
        self.prob_yo_st = Stationary(self.prob_yo)
        
        
        
    def set_prices(self, p, rc, pkap, kapbar):


        self.p = p
        self.rc = rc

        self.pkap = pkap
        self.kapbar = kapbar

        

        #assuming CRS technology for C-corp
        self.kcnc_ratio = ((self.theta * self.A)/(self.delk + self.rc))**(1./(1. - self.theta))
        self.w = (1. - self.theta)*self.A*self.kcnc_ratio**self.theta
        
        
        self.__is_price_set__ = True
        
        
        #implied prices
        self.rbar = (1. - self.taup) * self.rc
        self.rs = (1. - self.taup) * self.rc

        #set Xi-s.
        self.xi1 = ((self.ome*self.p)/(1. - self.ome))**(1./(self.rho-1.0))
        self.xi2 = (self.ome + (1. - self.ome) * self.xi1**self.rho)**(1./self.rho)
        self.xi3 = self.eta/(1. - self.eta) * self.ome / (1. + self.tauc) / self.xi2**self.rho #changed
        
        self.denom = (1. + self.p*self.xi1)*(1. + self.tauc)
        
        self.xi4 = (1. + self.rbar) / self.denom
        self.xi5 = (1. + self.grate) / self.denom
        self.xi6 = (self.yn - self.xnb) / self.denom #changed
        self.xi7 = 1./ self.denom #changed

        

        self.xi8 = ((self.alpha*self.p)/(self.rs + self.delk))**(1./(1.-self.alpha))        
        self.xi11 = self.xi7 #modified
        self.xi10 = (self.p*self.xi8**self.alpha - (self.rs + self.delk)*self.xi8)/self.denom
        self.xi13 = (self.nu*self.varpi*(self.rs + self.delk)/(self.alpha * self.w))**(1./(1.- self.upsilon))
        self.xi14 = self.w*self.xi13*(self.xi8**(1./(1.-self.upsilon)))/self.denom
        self.xi9 = (self.eta*self.ome*self.nu*(1.-self.varpi)*self.p*self.xi8**self.alpha)\
                   /((1.-self.eta)*(1.+self.tauc)*self.xi2**self.rho)
        self.xi12 = (self.vthet/self.veps)*self.nu*(1.-self.varpi)*self.p*self.xi8**self.alpha

    def save_parameters(self, file_path = './save_data/'):

        f = open(file_path + 'parameters.csv', 'w')

        f.write('Preferences')
        f.write('\n')
        f.write('Discount factor,' + str(self.beta))
        f.write('\n')        
        f.write('Leisure weight,' + str(self.eta))
        f.write('\n')        
        f.write('Intertemporal elasticity inverse,' + str(self.mu))
        f.write('\n')                
        f.write('C-corporate good share,' + str(self.ome))
        f.write('\n')                
        f.write('Love of business parameter,' + str(0.)) #this code does not have this term.
        f.write('\n')                
        f.write('Technologies')
        f.write('\n')                
        f.write('Technology growth,'+ str(self.grate))
        f.write('\n')                
        f.write('C-corporate fixed asset share,'+ str(self.theta))
        f.write('\n')                
        f.write('Private business fixed asset share,'+ str(self.alpha))
        f.write('\n')                
        f.write('Private business sweat capital share,'+ str(self.phi))
        f.write('\n')                
        f.write('Private business ces labor composite share,'+ str(self.nu))
        f.write('\n')                
        f.write('Private business employee hours share in ces labor composite ,'+ str(self.varpi))
        f.write('\n')                
        f.write('Private business hours substitution in ces labor composite ,'+ str(self.upsilon))
        f.write('\n')                
        f.write('Fixed asset depreciation,'+ str(self.delk))
        f.write('\n')                
        f.write('Sweat capital depreciation,'+ str(self.delkap))
        f.write('\n')                
        f.write('Sweat capital owner hour share,'+ str(self.vthet))
        f.write('\n')                
        f.write('Sweat capital c-good share,'+ str(self.veps))
        f.write('\n')                
        f.write('Sweat capital deterioration,'+ str(self.la))
        f.write('\n')                
        f.write('Sweat capital deterioration (bequest),'+ str(self.la_tilde))
        f.write('\n')                
        f.write('Working-capital constraint,'+ str(self.chi))
        f.write('\n')                
        
        f.write('Tax rates')
        f.write('\n')                
        f.write('Consumption tax,'+ str(self.tauc))
        f.write('\n')                
        f.write('Dividends tax,'+ str(self.taud))
        f.write('\n')                
        f.write('Profits tax,'+ str(self.taup))
        f.write('\n')                


        f.write('nbracket_left, nbracket_right, taun, psin\n')        
        for i in range(len(self.taun)):
            f.write(str(self.nbracket[i]) + ', ' + str(self.nbracket[i+1]) + ', ' +  str(self.taun[i]) + ', ' +  str(self.psin[i]))
            f.write('\n')                    

        f.write('bbracket_left, bbracket_right, taub, psib\n')        
        for i in range(len(self.taub)):
            f.write(str(self.bbracket[i]) + ', ' + str(self.bbracket[i+1]) + ', ' +  str(self.taub[i]) + ', ' +  str(self.psib[i]))
            f.write('\n')                    


        self.prob_z = np.ones((self.num_z, self.num_z))
        self.prob_eps = np.ones((self.num_eps, self.num_eps))        

        f.write('Transition of z')
        f.write('\n')
        f.write('t -> t+1')
        for iz in range(self.num_z):
            f.write(',' + str(self.zgrid[iz]))
        f.write('\n')

        for iz in range(self.num_z):
            f.write(str(self.zgrid[iz]))
            
            for izp in range(self.num_z):
                f.write(',' + str(self.prob_z[iz, izp]))

            f.write('\n')            

        f.write('Transition of eps')
        f.write('\n')
        f.write('t -> t+1')
        for iz in range(self.num_eps):
            f.write(',' + str( self.epsgrid[iz]))
        f.write('\n')

        for ieps in range(self.num_eps):
            f.write(str(self.epsgrid[ieps]))
            
            for iepsp in range(self.num_eps):
                f.write(',' + str(self.prob_eps[ieps, iepsp]))

            f.write('\n')            
            

        f.close()


        return None
        

    def print_parameters(self):

        print('')
        print('Parameters')
        print('alpha = ', self.alpha)
        print('beta = ', self.beta)
        print('chi = ', self.chi)
        print('delk = ', self.delk)
        print('delkap = ', self.delkap)
        print('eta = ', self.eta)
        print('g (govt spending) = ', self.g)
        print('grate (growth rate of the economy) = ', self.grate)
        print('la = ', self.la)
        print('mu = ', self.mu)
        print('ome = ', self.ome)
        print('upsilon = ', self.upsilon)        
        print('phi = ', self.phi)
        print('rho = ', self.rho)
        print('varpi = ', self.varpi)
        print('tauc = ', self.tauc)
        print('taud = ', self.taud)
        print('taup = ', self.taup)
        print('taucg = ', self.taucg)
        print('theta = ', self.theta)
        print('veps = ', self.veps)
        print('vthet = ', self.vthet)
        print('xnb = ', self.xnb)
        print('yn = ', self.yn)
        print('zeta = ', self.zeta)
        print('A = ', self.A)

        print('')
        print('nonlinear tax function')


        for ib, tmp in enumerate(self.taub):
            print(f'taub{ib} = {tmp}')
        for ib, tmp in enumerate(self.psib):
            print(f'psib{ib} = {tmp}')            
        for ib, tmp in enumerate(self.bbracket):
            print(f'bbracket{ib} = {tmp}')
        for i, tmp in enumerate(self.taun):
            print(f'taun{i} = {tmp}')
        for i, tmp in enumerate(self.psin):
            print(f'psin{i} = {tmp}')            
        for i, tmp in enumerate(self.nbracket):
            print(f'nbracket{i} = {tmp}')
        

        print('')
        print('Parameters specific to a lifecycle model')
        print('iota = ', self.iota) #added
        print('la_tilde = ', self.la_tilde) #added
        print('tau_wo = ', self.tau_wo) #added
        print('tau_bo = ', self.tau_bo) #added
        print('trans_retire = ', self.trans_retire)
        
        print(f'prob_yo =  {self.prob_yo[0,0]}, {self.prob_yo[0,1]}, {self.prob_yo[1,0]}, {self.prob_yo[1,1]}.') #added
        print('statinary dist of prob_yo = ', self.prob_yo_st) #added
        print('')
        
        
        if self.__is_price_set__:
            
            print('')
            print('Prices')
            print('w = ', self.w)
            print('p = ', self.p)
            print('rc = ', self.rc)
            print('pkap = ', self.pkap)
            print('kapbar = ', self.kapbar)                        
            
            print('')
            print('Implied prices')
            print('rbar = ', self.rbar)
            print('rs = ', self.rs)

            print('')
            print('Implied Parameters')
            print('nu = ', self.nu)
            print('bh (beta_tilde) = ', self.bh)
            print('varrho = ', self.varrho)


            print('')
            print('xi1 = ', self.xi1)
            print('xi2 = ', self.xi2)
            print('xi3 = ', self.xi3)
            print('xi4 = ', self.xi4)
            print('xi5 = ', self.xi5)
            print('xi6 = ', self.xi6)
            print('xi7 = ', self.xi7)
            print('xi8 = ', self.xi8)
            print('xi9 = ', self.xi9)
            print('xi10 = ', self.xi10)
            print('xi11 = ', self.xi11)
            print('xi12 = ', self.xi12)
            print('xi13 = ', self.xi13)
            print('xi14 = ', self.xi14)

            
        else:
            print('')
            print('Prices not set')
            
        print('')
        print('Computational Parameters')

        print('num_suba_inner = ', self.num_suba_inner)
        print('num_subkap_inner = ', self.num_subkap_inner)
        print('sim_time = ', self.sim_time)
        print('num_total_pop = ', self.num_total_pop)

        
    def generate_util(self):

        bh = self.bh
        eta = self.eta
        mu = self.mu
        
        @nb.jit(nopython = True)
        def util(c, l):
            if c > 0.0 and l > 0.0 and l <= 1.0:
                return (1. - bh) * (((c**eta)*(l**(1. - eta)))**(1. - mu))
            else:
                return -np.inf

        return util
    
    def generate_dc_util(self):

        bh = self.bh
        eta = self.eta
        mu = self.mu
        
        #this is in the original form
        @nb.jit(nopython = True)
        def dc_util(c, l):
            if c > 0.0 and l > 0.0 and l <= 1.0:
                return eta*c**(eta*(1.-mu) - 1.0)*((l**(1.-eta)))**(1.-mu)

            else:
                print('dc_util at c = ', c, ', l = ', l, 'is not defined.')
                print('nan will be returned.')
                return np.nan #???
            
        return dc_util
        
        
        
    def generate_cstatic(self):
        
        #load variables
        alpha = self.alpha
        beta = self.beta
        chi = self.chi
        delk = self.delk
        delkap = self.delkap
        eta = self.eta
        g = self.g
        grate = self.grate
        la = self.la
        tau_wo = self.tau_wo
        mu = self.mu
        ome = self.ome
        phi = self.phi
        rho = self.rho
        tauc = self.tauc
        taud = self.taud
        
        
        taup = self.taup
        theta = self.theta

        taun = self.taun
        psin = self.psin
        nbracket = self.nbracket

        trans_retire = self.trans_retire
        
        vthet = self.vthet
        xnb = self.xnb
        yn = self.yn
        zeta= self.zeta

        agrid = self.agrid
        kapgrid = self.kapgrid
        epsgrid = self.epsgrid
        zgrid = self.zgrid

        prob = self.prob

        is_to_iz = self.is_to_iz
        is_to_ieps = self.is_to_ieps

        num_suba_inner = self.num_suba_inner
        num_subkap_inner = self.num_subkap_inner

        num_a = self.num_a
        num_kap = self.num_kap
        num_eps = self.num_eps
        num_z = self.num_z
        num_s = self.prob.shape[0]

        nu = self.nu
        bh = self.bh
        varrho = self.varrho

        w = self.w
        p = self.p
        rc = self.rc

        rbar = self.rbar
        rs = self.rs

        xi1 = self.xi1
        xi2 = self.xi2
        xi3 = self.xi3
        xi4 = self.xi4
        xi5 = self.xi5
        xi6 = self.xi6
        xi7 = self.xi7
        xi8 = self.xi8
        xi9 = self.xi9
        xi10 = self.xi10
        xi11 = self.xi11
        xi12 = self.xi12
        #end loading 
        
        util = self.generate_util()

        @nb.jit(nopython = True)
        def get_cstatic(s):
            a = s[0]
            an = s[1]
            eps = s[2]
            is_o = s[3] # if young, this is 1. if old, 0. (or True, False)

            u = -np.inf
            cc = -1.0
            cs = -1.0
            cagg = -1.0

            l = -1.0
            n = -1.0


            #here, I directly modify epsilon
            if is_o:
                eps = tau_wo*eps #replace eps with tau_wo*eps


            #is this unique?
            #repeat until n falls in bracket nuber i (i=0,1,2,..,I-1)
            i = 0
            j = 0
            num_taun = len(taun)
            wepsn = 0.0
            for i in range(num_taun):
                n = (xi3*w*eps*(1.-taun[i]) - xi4*a + xi5*an - xi6 - xi7*(psin[i] + is_o*trans_retire))/(w*eps*(1.-taun[i])*(xi3 + xi7))
                wepsn = w*eps*n #wageincome

                j = locate(wepsn, nbracket)

                if i == j:
                    break

            obj_i = 0.
            obj_i1 = 0.

            
            #when solution is at a kink
            flag = True
            flag2 = False
            
            if i == len(taun) - 1 and i != j: #if i is not identified above
                flag = False
                flag2 = True

                for i, wepsn in enumerate(nbracket[1:-1]): #remove -inf, inf
                    #maybe it does not matter which bracket he is in? 
                    n = wepsn/w/eps
                    obj_i = n - ( (xi3*w*eps*(1.-taun[i]) - xi4*a + xi5*an - xi6 - xi7*(psin[i]+is_o*trans_retire))/(w*eps*(1.-taun[i])*(xi3 + xi7)))
                    obj_i1 = n - ( (xi3*w*eps*(1.-taun[i+1]) - xi4*a + xi5*an - xi6 - xi7*(psin[i+1]+is_o*trans_retire))/(w*eps*(1.-taun[i+1])*(xi3 + xi7)))

                    if obj_i * obj_i1 < 0:
                        flag = True
                        break
                    
            # if flag2 and flag:
            #     print('a solution is found at a corner ')
            #     print('obj_i = ', obj_i)
            #     print('obj_i1 = ', obj_i1)
            #     print('a = ', a)
            #     print('an = ', an)
            #     print('eps = ', eps)
            #     print('i = ', i)

            #     # print('j = ', j)
            #     print('n = ', n)
            #     print('wepsn = ', wepsn)
            #     print('')

            if not flag:
                print('no solution find ')
                print('err: cstatic: no bracket for n')
                print('a = ', a)
                print('an = ', an)
                print('eps = ', eps)
                print('i = ', i)
                print('j = ', j)
                print('n = ', n)
                print('wepsn = ', wepsn)
                print('')
                                        

            if n < 0.0:
                n = 0.0
                wepsn = w*eps*n
                i = locate(wepsn, nbracket)
                

            if n >= 0. and n <= 1.:

                l = 1. - n

                #cc from FOC  is wrong at the corner.
                cc = xi4*a - xi5*an + xi6 + xi7*((1.-taun[i])*w*eps*n + (psin[i] + is_o*trans_retire))
                cs = xi1*cc
                cagg = xi2*cc
                u = util(cagg, 1. - n)


            return u, cc, cs, cagg, l ,n, i, taun[i], psin[i] + is_o*trans_retire
        return get_cstatic
        

    def generate_sstatic(self):
        
        # for variable in self.__dict__ : exec(variable+'= self.'+variable)
        alpha = self.alpha
        beta = self.beta
        chi = self.chi
        delk = self.delk
        delkap = self.delkap
        eta = self.eta
        g = self.g
        grate = self.grate
        la = self.la
        mu = self.mu
        ome = self.ome
        phi = self.phi
        upsilon = self.upsilon        
        rho = self.rho
        varpi = self.varpi
        tauc = self.tauc
        taud = self.taud
        taup = self.taup
        theta = self.theta
        veps = self.veps
        vthet = self.vthet
        xnb = self.xnb
        yn = self.yn
        zeta= self.zeta

        tau_bo = self.tau_bo
        trans_retire = self.trans_retire

        taub = self.taub
        psib = self.psib
        bbracket = self.bbracket
        

        agrid = self.agrid
        kapgrid = self.kapgrid
        epsgrid = self.epsgrid
        zgrid = self.zgrid

        prob = self.prob

        is_to_iz = self.is_to_iz
        is_to_ieps = self.is_to_ieps

        num_suba_inner = self.num_suba_inner
        num_subkap_inne = self.num_subkap_inner

        num_a = self.num_a
        num_kap = self.num_kap
        num_eps = self.num_eps
        num_z = self.num_z

        nu = self.nu
        bh = self.bh


        w = self.w
        p = self.p
        rc = self.rc

        rbar = self.rbar
        rs = self.rs

        denom = self.denom
        xi1 = self.xi1
        xi2 = self.xi2
        xi3 = self.xi3
        xi4 = self.xi4
        xi5 = self.xi5
        xi6 = self.xi6
        xi7 = self.xi7
        xi8 = self.xi8
        xi9 = self.xi9
        xi10 = self.xi10
        xi11 = self.xi11
        xi12 = self.xi12
        xi13 = self.xi13
        xi14 = self.xi14        

        
        util = self.generate_util()

        
        @nb.jit(nopython = True)
        def Hy(h, alp6):
            tmp = (1. - alp6*h**((nu/(1.-alpha) - upsilon)*(upsilon/(1.-upsilon))-upsilon))
            tmp = tmp / (1. - varpi)
            tmp = (tmp**(1./upsilon))*h

            return tmp

        @nb.jit(nopython = True)
        def g(h, alp6):

            return ((h**(upsilon - nu/(1.-alpha)))*(Hy(h, alp6)**(1.-upsilon)))**(vthet/(veps+vthet))
        
        @nb.jit(nopython = True)
        def get_h_lbar(alp6):

            return alp6**(1./(upsilon + (upsilon - nu/(1.-alpha))*(upsilon/(1.-upsilon))))
       

        @nb.jit(nopython = True)
        def solve_hhyhkap(s):
            #return h, hy, hkap
            a = s[0]
            an = s[1]
            kap = s[2]
            kapn = s[3]
            z = s[4]
            is_o = s[5]            
            
            if is_o:
                z = tau_bo*z #replace eps with tau_wo*eps

            #case 0
            if (kap == 0.0 and kapn > 0.0) or (kap > 0.0 and kap < 1.0e-9 and kapn > (1. - delkap)/(1. + grate) * kap):
            #if (kap < 1.0e-10) and kapn >= 1.0e-10:

                #New version which is consistent with kap>0 version in limit                
                alp1 = eta/(1. - eta) * ome / xi2**rho / (1. + tauc) #updated
                alp2 = vthet*(xi4*a - xi5*an + xi6 + xi7*(psib + is_o*trans_retire))/(1.-taub)/veps/ ((1.+grate)*kapn/zeta)**(1./vthet) 
                alp3 = vthet/(veps*((1. + p*xi1)*(1. + tauc))) #updated

                hk_min = 1.0e-10 #used to be 0., but hk_min**(-1) = 0.**(-1.) is not ok in python grammer. in numba, it works though.
                hk_max = 1.0
                

                ####bisection start
                x_lb = ((((1. + grate)*kapn - (1. - delkap)*kap)/zeta)**(1./vthet))*hk_min**(-veps/vthet)
                ib_lb = locate(-x_lb, bbracket)
                val_lb = alp1*(1. - hk_min) - alp2[ib_lb]*hk_min**((vthet + veps)/vthet) + alp3*hk_min

                x_ub = ((((1. + grate)*kapn - (1. - delkap)*kap)/zeta)**(1./vthet))*hk_max**(-veps/vthet)                
                ib_ub = locate(-x_ub, bbracket)                
                val_ub = alp1*(1. - hk_max) - alp2[ib_ub]*hk_max**((vthet + veps)/vthet) + alp3*hk_max

                
                if val_lb *val_ub > 0.0:
                    # print('warning : no bracket')
                    return -1., -1., -1.
                
                sign = -1.0
                if val_ub > 0.:
                    sign = 1.0

                hk = (hk_max + hk_min)/2.
                x = ((((1. + grate)*kapn - (1. - delkap)*kap)/zeta)**(1./vthet))*hk**(-veps/vthet)
                ib = locate(-x, bbracket)

                it = 0
                tol = 1.0e-12
                maxit = 200
                val_m = 10000.
                diff = 1.0e23

                while it < maxit:
                    it = it + 1
                    
                    val_m = alp1*(1. - hk) - alp2[ib]*hk**((vthet + veps)/vthet) + alp3*hk


                    if sign * val_m > 0.:
                        hk_max = hk
                    elif sign * val_m < 0.:
                        hk_min = hk

                    diff = abs((hk_max + hk_min)/2 - hk)
                    hk = (hk_max + hk_min)/2.
                    x = ((((1. + grate)*kapn - (1. - delkap)*kap)/zeta)**(1./vthet))*hk**(-veps/vthet)
                    ib = hunt(-x, bbracket, ib)

                    if diff < tol:
                        break

                #convergence check
                if it == maxit or diff >= tol:
                    print('err: bisection method for hmax did not converge.')
                    print('val_m = ', val_m)
                    print('hk = ', hk)                    
                    print('hk_max = ', hk_max)
                    print('diff = ', diff)                    
                    
                ####bisection end
                

                # # mx_lb = max( (alp3*vthet/(alp2*(vthet + veps)))**(vthet/veps), (alp3/alp2) ) #typo?
                # # mx_lb = max( (alp3*vthet/(alp2*(vthet + veps)))**(vthet/veps), (alp3/alp2)**(vthet/veps) )

                # ###start newton method
                # mx = mx_lb
                # # print('mx = ', mx_lb)
                
                # it = 0
                # maxit = 100 #scipy's newton use maxit = 50
                # tol = 1.0e-15
                # dist = 10000000.

                # while it < maxit:
                #     it = it + 1
                #     res = alp1*(1. - mx) - alp2*mx**((vthet + veps)/vthet) + alp3*mx

                #     dist = abs(res)

                #     if dist < tol:
                #         break

                #     dres= -alp1 - alp2*((vthet + veps)/vthet)*mx**(veps/vthet) + alp3
                #     diff = res/dres
                #     mx = mx - res/dres

                # #convergence check
                # if it == maxit:
                #     print('err: newton method for mx did not converge.')
                #     print('mx = ', mx)

                # ans = mx    

                # ###end newton method

                return 0., 0., hk
            
            #case 1            
            elif kap == 0.0 and kapn == 0.0:

                return 0.0, 0.0, 0.0

            #case 2 -- the main case--
            elif kap > 0.0 and kapn > (1. - delkap)/(1. + grate) * kap:

                alp1 = xi9 
                alp2 = (xi4*a - xi5*an + xi6 + xi7*(psib+is_o*trans_retire))/(1.-taub)/((z*kap**phi)**(1./(1.-alpha))) #updated
                alp3 = xi10 
                alp5 = (((((1. + grate)*kapn - (1. - delkap)*kap)/zeta)**(1./vthet))/(xi12 * (z*kap**phi)**(1./(1.-alpha))))**(vthet/(vthet + veps))
                alp4 = xi11 * xi12 * alp5
                alp6 = varpi*(xi13**upsilon)*(xi8*(z*kap**phi)**(1./(1.-alpha)))**(upsilon/(1.-upsilon))
                alp7 = xi14*(z*kap**phi)**((1./(1.-alpha))*(upsilon/(1.-upsilon))) #              

                h_lbar = alp6**(1./(upsilon + (upsilon - nu/(1.-alpha))*(upsilon/(1.-upsilon))))

                ### we can do better ###
                tmp = (alp6 + 1. - varpi)
                h_hbar_plus = max(tmp**(1./upsilon), tmp**(1./(upsilon + (upsilon - nu/(1.-alpha))*(upsilon/(1.-upsilon)))) )

                hmax_lb = h_lbar                
                hmax_ub = h_hbar_plus


                ####bisection start
                val_lb = 1. - Hy(hmax_lb, alp6) - alp5*g(hmax_lb, alp6)
                val_ub = 1. - Hy(hmax_ub, alp6) - alp5*g(hmax_ub, alp6)
                
                
                if val_lb * val_ub > 0.0:
                    print('error: no bracket')
                    
                sign = -1.0
                if val_ub > 0.:
                    sign = 1.0

                hmax = (hmax_lb + hmax_ub)/2.

                it = 0
                tol = 1.0e-12
                maxit = 200
                val_m = 10000.

                diff = 1.0e23
                
                while it < maxit:
                    it = it + 1
                    val_m = 1. - Hy(hmax, alp6) - alp5*g(hmax, alp6)

                    if sign * val_m > 0.:
                        hmax_ub = hmax
                    elif sign * val_m < 0.:
                        hmax_lb = hmax

                    diff = abs((hmax_lb + hmax_ub)/2 - hmax)
                    hmax = (hmax_lb + hmax_ub)/2.

                    if diff < tol:
                        break

                #convergence check
                if it == maxit or diff >= tol:
                    print('err: bisection method for hmax did not converge.')
                    print('val_m = ', val_m)
                    print('mymax = ', hmax)
                    
                ####bisection end


                ####bisection start
                h_lb = h_lbar
                h_ub = hmax

                #check bracketting
                hk_lb = alp5*g(h_lb, alp6)
                x_lb = ((((1. + grate)*kapn - (1. - delkap)*kap)/zeta)**(1./vthet))*hk_lb**(-veps/vthet)
                ys_lb = (xi8**alpha)*(z*(kap**phi)*(h_lb**nu))**(1./(1.-alpha))
                ks_lb = xi8*(z*(kap**phi)*(h_lb**nu))**(1./(1.-alpha))
                
                if h_lb > 0.0:
                    ns_lb = xi13*(ks_lb/(h_lb**upsilon))**(1./(1.-upsilon))
                else:
                    ns_lb = 0.0

                bizinc_lb = p*ys_lb - (rs + delk)*ks_lb - x_lb - w*ns_lb
                
                
                ib_lb = locate(bizinc_lb, bbracket) 
                val_lb = alp1*(1. - Hy(h_lb, alp6) - alp5*g(h_lb, alp6))\
                         - (alp2[ib_lb]*h_lb**(-nu/(1.-alpha)) + alp3 - alp7*h_lb**((nu/(1.-alpha) - upsilon)*(upsilon/(1.-upsilon))-upsilon))*(h_lb**upsilon)*(Hy(h_lb, alp6)**(1.-upsilon)) + alp4*g(h_lb, alp6)


                hk_ub = alp5*g(h_ub, alp6)
                x_ub = ((((1. + grate)*kapn - (1. - delkap)*kap)/zeta)**(1./vthet))*hk_ub**(-veps/vthet)
                ys_ub = (xi8**alpha)*(z*(kap**phi)*(h_ub**nu))**(1./(1.-alpha))
                ks_ub = xi8*(z*(kap**phi)*(h_ub**nu))**(1./(1.-alpha))
                if h_ub > 0.0:
                    ns_ub = xi13*(ks_ub/(h_ub**upsilon))**(1./(1.-upsilon))
                else:
                    ns_ub = 0.0

                bizinc_ub = p*ys_ub - (rs + delk)*ks_ub - x_ub - w*ns_ub
                

                ib_ub = locate(bizinc_ub, bbracket) 
                val_ub = alp1*(1. - Hy(h_ub, alp6) - alp5*g(h_ub, alp6))\
                         - (alp2[ib_ub]*h_ub**(-nu/(1.-alpha)) + alp3 - alp7*h_ub**((nu/(1.-alpha) - upsilon)*(upsilon/(1.-upsilon))-upsilon))*(h_ub**upsilon)*(Hy(h_ub, alp6)**(1.-upsilon)) + alp4*g(h_ub, alp6)

                # print('val_lb = ', val_lb)
                # print('val_ub = ', val_ub)
                # print('h_lbar = ', h_lbar)
                # print('alp1 = ', alp1)
                # print('alp2 = ', alp2)
                # print('alp3 = ', alp3)
                # print('alp4 = ', alp4)
                # print('alp5 = ', alp5)
                # print('alp6 = ', alp6)                    
                
                
                if val_ub*val_lb > 0.0: 
                    # print('no bracket for h. Infer no solution')
                    return -1., -1., -1.
                
                sign = -1.0
                if val_ub > 0.:
                    sign = 1.0

                h = (h_lb + h_ub)/2.

                hk = alp5*g(h, alp6)
                x = ((((1. + grate)*kapn - (1. - delkap)*kap)/zeta)**(1./vthet))*hk**(-veps/vthet)
                ys = (xi8**alpha)*(z*(kap**phi)*(h**nu))**(1./(1.-alpha))
                ks = xi8*(z*(kap**phi)*(h**nu))**(1./(1.-alpha))
                
                if h > 0.0:
                    ns = xi13*(ks/(h**upsilon))**(1./(1.-upsilon))
                else:
                    ns = 0.0

                bizinc = p*ys - (rs + delk)*ks - x - w*ns
                
                ib = locate(bizinc, bbracket) #set brackt for h

                it = 0
                tol = 1.0e-8
                #rtol = 4.4408920985006262e-16 #this is default tolerance, but sometimes too rigid.
                rtol = 1.0e-8
                
                maxit = 400
                val_m = 10000.
                diff = 1.0e10
                
                while it < maxit:
                    it = it + 1
                    
                    # if h > 0. and h < 1.0e-6:
                    #     tol = 1.0e-20

                    val_m = alp1*(1. - Hy(h, alp6) - alp5*g(h, alp6))\
                            - (alp2[ib]*h**(-nu/(1.-alpha)) + alp3 - alp7*h**((nu/(1.-alpha) - upsilon)*(upsilon/(1.-upsilon))-upsilon))*(h**upsilon)*(Hy(h, alp6)**(1.-upsilon)) + alp4*g(h, alp6)

                    if sign * val_m > 0.:
                        h_ub = h
                    else:
                    #elif sign * val_m < 0.:
                        h_lb = h

                    diff = abs((h_lb + h_ub)/2. - h)                        
                    h = (h_lb + h_ub)/2.



                    hk = alp5*g(h, alp6)
                    x = ((((1. + grate)*kapn - (1. - delkap)*kap)/zeta)**(1./vthet))*hk**(-veps/vthet)
                    ys = (xi8**alpha)*(z*(kap**phi)*(h**nu))**(1./(1.-alpha))
                    ks = xi8*(z*(kap**phi)*(h**nu))**(1./(1.-alpha))
                    
                    if h > 0.0:
                        ns = xi13*(ks/(h**upsilon))**(1./(1.-upsilon))
                    else:
                        ns = 0.0

                    bizinc = p*ys - (rs + delk)*ks - x - w*ns
                    
                    ib = hunt(bizinc, bbracket, ib) 

                    #if diff < tol and abs(val_m) < rtol:
                    if (diff < tol and it > 100) or (diff < tol and abs(val_m) < rtol): #taking into acocunt potential jumps
                        break

                #convergence check
                if it == maxit:
                    print('err: bisection method for h did not converge.')
                    print('it = ', it)
                    print('tol = ', tol)
                    print('diff = ', diff)
                    print('alp1 = ', alp1)
                    print('alp2 = ', alp2)
                    print('alp3 = ', alp3)
                    print('alp4 = ', alp4)
                    print('alp5 = ', alp5)
                    print('alp6 = ', alp6)
                    print('alp7 = ', alp7)                                        
                    
                    print('val_m = ', val_m)
                    print('h_ub = ', h_ub)
                    print('h_lb = ', h_lb)                                        
                    print('h = ', h)
                    print('hmax = ', hmax)

                    print('val_lb = ', val_lb)
                    print('val_ub = ', val_ub)

                    print('a = ', a)
                    print('an = ', an)
                    print('kap = ', kap)
                    print('kapn = ', kapn)
                    print('z = ', z)

                ans = h
                #### bisection end ####


                if ans == 0.0:
                    print('A corner solution at 0.0 is obtianed: consider setting a smaller xtol.')
                    print('my = ', ans)


        #         if ans == mymax:
        #             #sometimes ans is extremely close to mymax
        #             #due to solver's accuracy.

                return h, Hy(h, alp6), alp5*g(h, alp6)

            else:
                #print('error: kap < 0 is not allowed.')
                return -1., -1., -1.


        @nb.jit(nopython = True)
        def get_sstatic(s):

            a = s[0]
            an = s[1]
            kap = s[2]
            kapn = s[3]
            z = s[4]
            is_o = s[5]

            if is_o :
                z = tau_bo*z #replace eps with tau_wo*eps
            

            #initialize
            u = -np.inf
            l = -2.0
            cc = -2.0
            cs = -2.0
            cagg = -2.0
            h = -2.0
            hy = -2.0
            hkap = -2.0
            x = -2.0            
            ks = -2.0
            ys = -2.0
            ns = -2.0

            ys_tmp = -2.0
            h_tmp = -2.0

            
            #if the state if feasible.
            if kapn >= (1. - delkap)/(1. + grate) * kap:
                h, hy, hkap = solve_hhyhkap(s)

                if hy >= 0. and hkap >= 0.:

                    x = -100.0

                    if hkap == 0.0: #if sweat equity production is positive
                        x = 0.0
                    else:
                        x = ((((1. + grate)*kapn - (1. - delkap)*kap)/zeta)**(1./vthet))*hkap**(-veps/vthet)


                    l = 1.0 - hy - hkap

                    ks = xi8*(z*(kap**phi)*(h**nu))**(1./(1.-alpha)) 
                    ys = (xi8**alpha)*(z*(kap**phi)*(h**nu))**(1./(1.-alpha))

                    if h > 0.0:
                        ns = xi13*(ks/(h**upsilon))**(1./(1.-upsilon))
                        ys_tmp = z*(ks**alpha)*(kap**phi)*(h**nu)
                        h_tmp = ((1. - varpi)*hy**upsilon + varpi*ns**upsilon)**(1./upsilon)
                        
                    elif h == 0.0:
                        ns = 0.0
                        ys_tmp = 0.0
                        h_tmp = 0.0

                    bizinc = p*ys - (rs + delk)*ks - x - w*ns
                    ib = locate(bizinc, bbracket)

                    
                    cc = xi4*a - xi5*an + xi6 + xi7*(psib[ib] + is_o*trans_retire) - xi11*(1.-taub[ib])*x \
                        + xi10*(1.-taub[ib])*(z*kap**phi)**(1./(1.-alpha))*(h**(nu/(1.-alpha))) \
                        - xi14*(1.-taub[ib])*(z*kap**phi)**(1./((1.-alpha)*(1.-upsilon)))*h**((nu/(1.-alpha)-upsilon)*(1./(1.-upsilon)))

                    cs = xi1 * cc
                    cagg = xi2 * cc


                    # #ys and ys_tmp are often slightly different, so tolerate small difference
                    if (np.abs(ys - ys_tmp) > 1.0e-6) and (ys >= 0.0):
                        print('err: ys does not match')
                        print('ys = ', ys)
                        print('ys_tmp = ', ys_tmp)                        

                    if (np.abs(h - h_tmp) > 1.0e-6) and (h >= 0.0):
                        print('err: h does not match')
                        print('h = ', h)
                        print('h_tmp = ', h_tmp)                        

                    if h > 0.0:
                        cc_tmp = xi9*(1.-taub[ib])*(z*kap**phi)**(1./(1.-alpha))*(h**(nu/(1.-alpha) - upsilon))*(hy**(upsilon  - 1.0))*(1. - hy - hkap)                        
                        # if (np.abs(cc - cc_tmp) > 1.0e-3):
                        #     print('err: cc does not match')
                        #     print('cc = ', cc)
                        #     print('cc_tmp = ', cc_tmp)
                        #     print('a = ', a)
                        #     print('an = ', an)
                        #     print('kap = ', kap)
                        #     print('kapn = ', kapn)
                        #     print('z = ', z)                            
                            

                    #feasibility check
                    if cagg > 0.0 and l > 0.0 and l <= 1.0 and an >= chi * ks: #the last condition varies across notes,...
                        u = util(cagg, l)


            return u, cc, cs, cagg, l, hy, hkap, h ,x, ks, ys, ns, ib, taub[ib], (psib[ib] + is_o*trans_retire)
        
        return get_sstatic
    
    
    def get_policy(self):
        for variable in self.__dict__ : exec(variable+'= self.'+variable, locals(), globals())
        
        Econ = self

        ###parameters for MPI###

        num_total_state = num_a * num_kap * num_s
        m = num_total_state // size
        r = num_total_state % size

        assigned_state_range = (rank*m+min(rank,r),(rank+1)*m+min(rank+1,r))
        num_assigned = assigned_state_range[1] - assigned_state_range[0]



        


        all_assigned_state_range = np.ones((size, 2))*(-2.)
        all_num_assigned = ()
        all_istart_assigned = ()


        for irank in range(size):

            all_istart_assigned += (int(irank*m+min(irank,r)), )
            all_assigned_state_range[irank,0] = irank*m+min(irank,r)
            all_assigned_state_range[irank,1] = (irank+1)*m+min(irank+1,r)
            all_num_assigned += (int(all_assigned_state_range[irank,1] - all_assigned_state_range[irank,0]),) 

        ###end parameters for MPI###



        @nb.jit(nopython = True)
        def unravel_ip(i_aggregated_state):

            istate, ia, ikap = unravel_index_nb(i_aggregated_state, num_s, num_a, num_kap)
            #ia, ikap, istate = unravel_index_nb(i_aggregated_state, num_a, num_kap, num_s)
            return istate, ia, ikap

        get_cstatic = Econ.generate_cstatic()

        #obtain the max...
        #cvals_supan = np.ones((num_a, num_eps)) * (-2.)
        cvals_supan = np.ones((num_a, num_eps, 2)) * (-2.)

        
        #for young c-corp workers
        for ia, a in enumerate(agrid):
                for ieps, eps in enumerate(epsgrid):

                    cvals_supan[ia, ieps, 0] = ((1. + rbar)*a + (1. - taun[0])*w*eps + psin[0] + yn - xnb)/(1. + grate)
                    
        #for old c-corp workers
        for ia, a in enumerate(agrid):
                for ieps, eps in enumerate(epsgrid):

                    cvals_supan[ia, ieps, 1] = ((1. + rbar)*a + (1. - taun[0])*tau_wo*w*eps + psin[0] + trans_retire + yn - xnb)/(1. + grate)
                    

        get_sstatic = Econ.generate_sstatic()



        ## construct subagrid and subkapgrid

        ## subagrid
        subagrid = np.ones((num_a-1)* (num_suba_inner-2) + num_a)
        num_suba = len(subagrid)

        ia = 0
        for ia in range(num_a - 1):
            subagrid[ia*(num_suba_inner) - ia : (ia+1)*(num_suba_inner) - ia] = np.linspace(agrid[ia], agrid[ia+1], num_suba_inner)



        ## subkapgrid
        subkapgrid = np.ones((num_kap - 1)* (num_subkap_inner - 2) + num_kap)

        num_subkap = len(subkapgrid)

        ikap = 0
        for ikap in range(num_kap-1):
            subkapgrid[ikap*(num_subkap_inner) - ikap : (ikap+1)*(num_subkap_inner) - ikap] = np.linspace(kapgrid[ikap], kapgrid[ikap+1], num_subkap_inner)



        ia_to_isuba = [1 for i in range(num_a)] #np.ones(num_a, dtype = int)
        for ia in range(num_a):
            ia_to_isuba[ia] = (num_suba_inner-1)*ia


        ikap_to_isubkap = [1 for i in range(num_kap)] #np.ones(num_kap, dtype = int)
        for ikap in range(num_kap):
            ikap_to_isubkap[ikap] = (num_subkap_inner-1)*ikap

        # ikap_to_isubkap_exp = [1 for i in range(num_kap+1)] #np.ones(num_kap, dtype = int)
        # for ikap in range(num_kap):
        #     ikap_to_isubkap_exp[ikap] = (num_subkap_inner-1)*ikap
        # ikap_to_isubkap_exp[-1] = len(subkapgrid) - 1



        # #check
        #subagrid[ia_to_isuba] - agrid
        #subkapgrid[ikap_to_isubkap] - kapgrid

        #export to numpy array,...sometimes we need this.
        ia_to_isuba = np.array(ia_to_isuba)
        ikap_to_isubkap = np.array(ikap_to_isubkap)
        #ikap_to_isubkap_exp = np.array(ikap_to_isubkap_exp)

        @nb.jit(nopython = True)
        def unravel_isub_mesh(i_subgrid_mesh):

            ind, ia_m, ikap_m = unravel_index_nb(i_aggregated_state, num_assigned, num_a-1, num_kap - 1)

            return ind, ia_m, ikap_m
        
        @nb.jit(nopython = True)
        def get_isub_mesh(ind, ia_m, ikap_m):

            return ind * (num_a-1)*(num_kap-1) + ia_m *(num_kap-1) + ikap_m



        #Store the S-corp utility values  the main grid

        s_util_origin_y = np.ones((num_assigned, num_a, num_kap))
        s_util_origin_o = np.ones((num_assigned, num_a, num_kap))

        @nb.jit(nopython = True)
        def get_s_util_origin(_s_util_origin_, is_o):

            for ip in range(assigned_state_range[0], assigned_state_range[1]):

                ind = ip - assigned_state_range[0]
                istate, ia, ikap = unravel_ip(ip)


                for ian in range(num_a):
                    for ikapn in range(num_kap):

                        a = agrid[ia]
                        kap = kapgrid[ikap]
                        z = zgrid[is_to_iz[istate]]
                        an = agrid[ian]
                        kapn = kapgrid[ikapn]

                        state = [a, an, kap, kapn, z, is_o]

        #                 _s_util_origin_[ind, ian, ikapn] = get_sutil_cache(ip, ia_to_isuba[ian], ikap_to_isubkap[ikapn])
                        _s_util_origin_[ind, ian, ikapn] = get_sstatic(state)[0]
                        # get_sstatic(state)[0] #


        get_s_util_origin(s_util_origin_y, 0)
        get_s_util_origin(s_util_origin_o, 1)
        
        del get_s_util_origin


        #prepare for caching data
        num_prealloc_y = int(num_assigned * (num_a-1)* (num_kap - 1) * 0.05) #assign 5%
        num_cached_y = 0
        ind_s_util_finemesh_cached_y = np.ones((num_assigned * (num_a-1)* (num_kap - 1)), dtype = int)*(-1)
        s_util_finemesh_cached_y = np.zeros((num_prealloc_y, num_suba_inner, num_subkap_inner))


        num_prealloc_o = int(num_assigned * (num_a-1)* (num_kap - 1) * 0.05) #assign 5%
        num_cached_o = 0
        ind_s_util_finemesh_cached_o = np.ones((num_assigned * (num_a-1)* (num_kap - 1)), dtype = int)*(-1)
        s_util_finemesh_cached_o = np.zeros((num_prealloc_o, num_suba_inner, num_subkap_inner))
        

        #define inner loop functions

        # @nb.jit(nopython = True)
        # def _search_on_finer_grid_2_(ian_lo, ian_hi, ikapn_lo, ikapn_hi, _EV_, ip,
        #                              _num_cached_, _is_o_,
        #                              _ind_s_util_finemesh_cached_o_ = ind_s_util_finemesh_cached_o,
        #                              _ind_s_util_finemesh_cached_y_ = ind_s_util_finemesh_cached_y,                                    
        #                              _s_util_finemesh_cached_o_ = s_util_finemesh_cached_o,
        #                              _s_util_finemesh_cached_y_ = s_util_finemesh_cached_y ):
        @nb.jit(nopython = True)
        def _search_on_finer_grid_2_(ian_lo, ian_hi, ikapn_lo, ikapn_hi, _EV_, ip,
                                     _num_cached_, _is_o_,
                                     _ind_s_util_finemesh_cached_o_,
                                     _ind_s_util_finemesh_cached_y_,
                                     _s_util_finemesh_cached_o_,
                                     _s_util_finemesh_cached_y_):
                                     

            _ind_s_util_finemesh_cached_ = _ind_s_util_finemesh_cached_y_
            _s_util_finemesh_cached_ = _s_util_finemesh_cached_y_

            if _is_o_:
                _ind_s_util_finemesh_cached_ = _ind_s_util_finemesh_cached_o_
                _s_util_finemesh_cached_ = _s_util_finemesh_cached_o_
                

            ian_c = ian_hi - 1
            ikapn_c = ikapn_hi - 1


            istate, ia, ikap = unravel_ip(ip)  
            ind = ip - assigned_state_range[0]

            a = agrid[ia]
            kap = kapgrid[ikap]
            z = zgrid[is_to_iz[istate]]

            subsubagrid = subagrid[ia_to_isuba[ian_lo] : ia_to_isuba[ian_hi]+1]
            subsubkapgrid = subkapgrid[ikap_to_isubkap[ikapn_lo] : ikap_to_isubkap[ikapn_hi]+1]

            num_prealloc = num_prealloc_y
            if _is_o_:
                num_prealloc = num_prealloc_o

            if (len(subsubagrid) != 2*num_suba_inner - 1) or (len(subsubkapgrid) != 2*num_subkap_inner - 1):
                print('error: grid number of the finer grid')



            #define a finer grid

            #s_util_fine_mesh = svals_util[ia, ikap, ia_to_isuba[ian_lo] : ia_to_isuba[ian_hi]+1, ikap_to_isubkap[ikapn_lo] : ikap_to_isubkap[ikapn_hi]+1 ,is_to_iz[istate]]

            s_util_fine_mesh = np.zeros((len(subsubagrid), len(subsubkapgrid)))


            for ian in [ian_lo, ian_hi-1]:
                for ikapn in [ikapn_lo, ikapn_hi-1]:
                    isub_mesh = get_isub_mesh(ind, ian, ikapn)

        #             print('isub_mesh = ', isub_mesh)

                    if _ind_s_util_finemesh_cached_[isub_mesh] == -1: #if not cashed


                        for ian_sub in range(ia_to_isuba[ian], ia_to_isuba[ian+1] + 1):
                            ian_ind = ian_sub - ia_to_isuba[ian_lo]

                            for ikapn_sub in range(ikap_to_isubkap[ikapn], ikap_to_isubkap[ikapn+1] + 1):
                                ikapn_ind = ikapn_sub - ikap_to_isubkap[ikapn_lo]

                                an = subagrid[ian_sub]
                                kapn = subkapgrid[ikapn_sub]

        #                         print('an = ', an)
        #                         print('kapn = ', kapn)

                                state = [a, an, kap, kapn, z, _is_o_]

                                s_util_fine_mesh[ian_ind, ikapn_ind] = get_sstatic(state)[0]

        #                         print('sutil = ', s_util_fine_mesh[ian_ind, ikapn_ind])
    

                        
                        
                        if _num_cached_ < num_prealloc:
                            ind_new_entry = _num_cached_  #this is inefficient. just keep track using another var.
                            #this should be less than something...
                            _s_util_finemesh_cached_[ind_new_entry, :, :] =\
                               s_util_fine_mesh[(ia_to_isuba[ian] - ia_to_isuba[ian_lo]):(ia_to_isuba[ian+1]+1 - ia_to_isuba[ian_lo]),\
                                                (ikap_to_isubkap[ikapn] - ikap_to_isubkap[ikapn_lo]):(ikap_to_isubkap[ikapn+1]+1 - ikap_to_isubkap[ikapn_lo])]

                            _ind_s_util_finemesh_cached_[isub_mesh] = ind_new_entry

                            _num_cached_ = _num_cached_ +1

        #                     print('cached')
        #                     print(_s_util_finemesh_cached_[ind_new_entry, :, :])
        #                     print('')
        #                     print('fine_mesh')
        #                     print(s_util_fine_mesh)





                    else: #if it is already cached, just load it

                         s_util_fine_mesh[(ia_to_isuba[ian] - ia_to_isuba[ian_lo]):(ia_to_isuba[ian+1]+1 - ia_to_isuba[ian_lo]),\
                                          (ikap_to_isubkap[ikapn] - ikap_to_isubkap[ikapn_lo]):(ikap_to_isubkap[ikapn+1]+1 - ikap_to_isubkap[ikapn_lo])] =\
                                                                                                    _s_util_finemesh_cached_[_ind_s_util_finemesh_cached_[isub_mesh], :, :]




            obj_fine_mesh = - (s_util_fine_mesh + fem2deval_mesh(subsubagrid, subsubkapgrid, agrid, kapgrid, _EV_[0, 0, :, :, istate])  )**(1./(1. - mu))




            ans_some = unravel_index_nb(np.argmin(obj_fine_mesh), len(subsubagrid), len(subsubkapgrid))



            _an_tmp_ = subsubagrid[ans_some[0]]
            _kapn_tmp_ = subsubkapgrid[ans_some[1]]
            _val_tmp_ = -obj_fine_mesh[ans_some[0], ans_some[1]] 
            _u_tmp_  = s_util_fine_mesh[ans_some[0], ans_some[1]]



            return ans_some[0], ans_some[1], _num_cached_ ,_an_tmp_, _kapn_tmp_, _val_tmp_, _u_tmp_


        @nb.jit(nopython = True)    
        def _inner_inner_loop_s_par_(ipar_loop, _EV_, _num_cached_, _is_o_,
                                     _ind_s_util_finemesh_cached_o_,
                                     _ind_s_util_finemesh_cached_y_,
                                     _s_util_finemesh_cached_o_,
                                     _s_util_finemesh_cached_y_):


            istate, ia, ikap = unravel_ip(ipar_loop)

        #     print('ia =, ', ia, ' ikap = ', ikap, ' istate = ', istate)

            a = agrid[ia]
            kap = kapgrid[ikap]

            iz = is_to_iz[istate]
            z = zgrid[iz]

            kapn_min = kap*(1. - delkap)/(1. + grate)

            #rough grid search 

            ind = ipar_loop - assigned_state_range[0]

            if _is_o_:
                s_util_origin = s_util_origin_o
            else:
                s_util_origin = s_util_origin_y
                
            obj_mesh = - (s_util_origin[ind, :, :] + _EV_[0, 0, :, :, istate]) **(1./(1. - mu))
            ans_tmp = unravel_index_nb(np.argmin(obj_mesh), num_a, num_kap)

            # if there is no feasible solution on the rough grid, pick (ia, ikap) as a guess
            if (np.all(obj_mesh == 0.0)): 
                ans_tmp = np.array([ia, ikap])
            


            an_tmp = agrid[ans_tmp[0]]
            kapn_tmp = kapgrid[ans_tmp[1]]
            val_tmp = -obj_mesh[ans_tmp[0], ans_tmp[1]] 
            u_tmp  = s_util_origin[ind, :, :][ans_tmp[0], ans_tmp[1]]


            #find surrounding grids
            ian_lo = max(ans_tmp[0] - 1, 0)
            ian_hi = min(ans_tmp[0] + 1, len(agrid) - 1)


            ikapn_lo = max(ans_tmp[1] - 1, 0)
            ikapn_hi = min(ans_tmp[1] + 1, len(kapgrid) - 1) #one is added



            if (ian_lo + 2 != ian_hi):
                if ian_lo == 0:
                    ian_hi = ian_hi + 1
                elif ian_hi == num_a -1:
                    ian_lo = ian_lo -1
                else:
                    print('error')

            if (ikapn_lo + 2 != ikapn_hi):
                if ikapn_lo == 0:
                    ikapn_hi = ikapn_hi + 1
                elif ikapn_hi == num_kap-1:
                    ikapn_lo = ikapn_lo - 1
                else:
                    print('error')



            #check if there exists mesh
            if (ian_lo + 2 != ian_hi) or (ikapn_lo + 2 != ikapn_hi):
                print('error: a finer grid was not generated.')


            if subkapgrid[ikap_to_isubkap[ikapn_hi]] <= kapn_min:
                print('subkapgrid[ikap_to_isubkap[ikapn_hi]] <= kapn_min')




            max_ian_sub = 2*num_suba_inner - 2
            max_ikapn_sub =  2*num_subkap_inner - 2


            max_iter = 200
            it_finer = 0
            while (it_finer < max_iter):
                it_finer = it_finer + 1


                ans = np.array([0, 0])

                ans[0], ans[1], _num_cached_, an_tmp, kapn_tmp, val_tmp, u_tmp =\
                        _search_on_finer_grid_2_(ian_lo, ian_hi, ikapn_lo, ikapn_hi, _EV_, ipar_loop, _num_cached_,_is_o_,
                                     _ind_s_util_finemesh_cached_o_,
                                     _ind_s_util_finemesh_cached_y_,
                                     _s_util_finemesh_cached_o_,
                                     _s_util_finemesh_cached_y_)

                # if _is_o_:
                #     ans[0], ans[1], _num_cached_, an_tmp, kapn_tmp, val_tmp, u_tmp =\
                #     _search_on_finer_grid_2_(ian_lo, ian_hi, ikapn_lo, ikapn_hi, _EV_, ipar_loop, _num_cached_,
                #                              ind_s_util_finemesh_cached_o,
                #                              s_util_finemesh_cached_o, _is_o_)
                # else:
                #     ans[0], ans[1], _num_cached_, an_tmp, kapn_tmp, val_tmp, u_tmp =\
                #     _search_on_finer_grid_2_(ian_lo, ian_hi, ikapn_lo, ikapn_hi, _EV_, ipar_loop, _num_cached_,
                #                              ind_s_util_finemesh_cached_y,
                #                              s_util_finemesh_cached_y, _is_o_)

                # stop if the best solution turns out to be infeasible one
                if u_tmp == -np.inf:
                    print('no feasible point on the finer mesh. Terminating the search. it_finer = ', it_finer, '.')
                    # # serious debugging
                    # print('agrid[ian_hi] = ', agrid[ian_hi])
                    # print('agrid[ian_lo] = ', agrid[ian_lo])                
                    # print('ian_hi = ', ian_hi)
                    # print('ian_lo = ', ian_lo)
                    # print('kapgrid[ikapn_hi] = ', kapgrid[ikapn_hi])
                    # print('kapgrid[ikapn_lo] = ', kapgrid[ikapn_lo])                
                    # print('ikapn_hi = ', ikapn_hi)
                    # print('ikapn_lo = ', ikapn_lo)
                    # print('ia = ', ia)
                    # print('ikap = ', ikap)
                    # print('istate = ', istate)                                    
                    # print('is_o = ', _is_o_)
                    # if _is_o_ == True:
                    #     print('is_o = ', True)
                    # elif _is_o_ == False:
                    #     print('is_o = ', False)
                    # else:
                    #     print('is_o = AAAAAAAAAAAA')
                    # print('ans_tmp = ', ans_tmp)
                    # print('it_finer = ', it_finer)
                    # print('')
                        
                    break


                elif ans[0] != 0 and ans[0] != max_ian_sub and ans[1] != 0 and ans[1] != max_ikapn_sub:
                    #solution
                    break
                
                elif ans[0] == 0 and (ans[1] != 0 or ans[1] != max_ikapn_sub):
                    if ian_lo == 0:
                        break
                    else:
                        ian_lo = ian_lo - 1
                        ian_hi = ian_hi - 1

                elif ans[0] == max_ian_sub and (ans[1] != 0 or ans[1] != max_ikapn_sub):
                    if ian_hi == num_a-1:
                        break
                    else:
                        ian_lo = ian_lo + 1
                        ian_hi = ian_hi + 1

                elif ans[1]  == 0 and (ans[0] != 0 or ans[0] != max_ian_sub):
                    if ikapn_lo == 0:
                        break
                    else:
                        ikapn_lo = ikapn_lo - 1
                        ikapn_hi = ikapn_hi - 1

                elif ans[1]  == max_ikapn_sub and (ans[0] != 0 or ans[0] != max_ian_sub):
                    if ikapn_hi == num_kap-1: 
                        break
                    else:
                        ikapn_lo = ikapn_lo + 1
                        ikapn_hi = ikapn_hi + 1

                elif ans[0] == 0 and ans[1] == 0:
                    if ian_lo == 0 and ikapn_lo == 0:
                        break
                    else:
                        if ian_lo != 0:
                            ian_lo = ian_lo - 1
                            ian_hi = ian_hi - 1
                        if ikapn_lo != 0:
                            ikapn_lo = ikapn_lo - 1
                            ikapn_hi = ikapn_hi - 1

                elif ans[0] == 0 and ans[1] == max_ikapn_sub:
                    if ian_lo == 0 and ikapn_hi == num_kap-1:
                        break
                    else:
                        if ian_lo != 0:
                            ian_lo = ian_lo - 1
                            ian_hi = ian_hi - 1
                        if ikapn_hi != num_kap - 1:
                            ikapn_lo = ikapn_lo + 1
                            ikapn_hi = ikapn_hi + 1

                elif ans[0] == max_ian_sub and ans[1] == 0:
                    if ian_hi == num_a-1 and ikapn_lo == 0:
                        break
                    else:
                        if ian_hi != num_a-1:
                            ian_lo = ian_lo + 1
                            ian_hi = ian_hi + 1
                        if ikapn_lo != 0:
                            ikapn_lo = ikapn_lo - 1
                            ikapn_hi = ikapn_hi - 1

                elif ans[0] == max_ian_sub and ans[1] == max_ikapn_sub:
                    if ian_hi == num_a-1 and ikapn_hi == num_kap-1:
                        break
                    else:
                        if ian_hi != num_a-1:
                            ian_lo = ian_lo + 1
                            ian_hi = ian_hi + 1
                        if ikapn_hi != num_kap-1:
                            ikapn_lo = ikapn_lo + 1
                            ikapn_hi = ikapn_hi + 1
                else:
                    print('error: ****')
        #                         print('ans[0] = {}, ans[1] = {}'.format(ans[0], ans[1]))

                if it_finer == max_iter:
                    print('error: reached the max fine grid search')
                    print('ia = ', ia, ', ikap = ', ikap, ', istate = ', istate)

                    break


            return an_tmp, kapn_tmp, val_tmp, u_tmp, _num_cached_


        @nb.jit(nopython = True) 
        def _inner_loop_s_with_range_(assigned_indexes, _EV_, _vs_an_, _vs_kapn_, _vsn_, _vs_util_, _num_cached_, _is_o_,
                                     _ind_s_util_finemesh_cached_o_,
                                     _ind_s_util_finemesh_cached_y_,
                                     _s_util_finemesh_cached_o_,
                                     _s_util_finemesh_cached_y_):

        #     for istate in range(num_s):
        #         for ia in range(num_a):
        #             for ikap in range(num_kap):

            ibegin = assigned_indexes[0]
            iend = assigned_indexes[1]



            ind = 0
            for ipar_loop in range(ibegin, iend):


                istate, ia, ikap = unravel_ip(ipar_loop)


                an_tmp = -3.0
                kapn_tmp = -3.0
                val_tmp = -3.0
                u_tmp = -3.0

                an_tmp, kapn_tmp, val_tmp, u_tmp, _num_cached_ =\
                    _inner_inner_loop_s_par_(ipar_loop, _EV_, _num_cached_, _is_o_,
                                             _ind_s_util_finemesh_cached_o_,
                                             _ind_s_util_finemesh_cached_y_,
                                             _s_util_finemesh_cached_o_,
                                             _s_util_finemesh_cached_y_)

            
                _vs_an_[ind] = an_tmp
                _vs_kapn_[ind] = kapn_tmp
                _vsn_[ind] = val_tmp
                _vs_util_[ind] = u_tmp


                ind = ind+1

            return _num_cached_

        @nb.jit(nopython = True)    
        def obj_loop_c(*args):
            _an_ = args[0]
            _EV_ = args[1]
            _ia_ = args[2]
            _ikap_ = args[3]
            _istate_ = args[4]
            _is_o_ = args[5]

            a = agrid[_ia_]
            eps = epsgrid[is_to_ieps[_istate_]]
            state = [a, _an_, eps, _is_o_]

            #possibly we need to replace state with np.array(state)
            #state = np.array([agrid[_ia_], _an_, epsgrid[is_to_ieps[_istate_]]]) 
            ##state = np.array([agrid[_ia_], _an_, epsgrid[is_to_ieps[_istate_]]], _is_o_)

            u = get_cstatic(state)[0]

            return -(u + fem2d_peval(_an_, la*kapgrid[_ikap_], agrid, kapgrid, _EV_[0, 0, :, :, _istate_]) )**(1./(1. - mu)) 

        #epsilon = np.finfo(float).eps
        @nb.jit(nopython = True)
        def _inner_inner_loop_c_(_an_sup_, _EV_, _ia_, _ikap_ ,_istate_, _is_o_):

            #arguments
            ax = 0.0
            cx = _an_sup_
            bx = 0.5*(ax + cx)

            tol=1.0e-8
            itmax=500

            #parameters
            CGOLD=0.3819660
            ZEPS=1.0e-3*2.2204460492503131e-16
            #*np.finfo(float).eps

            brent = 1.0e20
            xmin = 1.0e20

            a=min(ax,cx)
            b=max(ax,cx)
            v=bx
            w=v
            x=v
            e=0.0 
            fx= obj_loop_c(x,  _EV_, _ia_, _ikap_ ,_istate_, _is_o_) 
            fv=fx
            fw=fx

            d = 0.0

            it = 0
            for it in range(itmax):


                xm=0.5*(a+b)
                tol1=tol*abs(x)+ZEPS
                tol2=2.0*tol1

                tmp1 = tol2 - 0.5*(b-a)
                tmp2 = abs(x-xm)



                if abs(x - xm) <= tol2 - 0.5*(b - a):
                    it = itmax

                    xmin=x
                    brent=fx


                if (abs(e) > tol1):
                    r=(x-w)*(fx-fv)
                    q=(x-v)*(fx-fw)
                    p=(x-v)*q-(x-w)*r
                    q=2.0*(q-r)

                    if (q > 0.0): 
                        p=-p

                    q=abs(q)
                    etemp=e
                    e=d

                    if abs(p) >= abs(0.5*q*etemp) or  p <= q*(a-x) or p >= q*(b-x):

                        #e=merge(a-x,b-x, x >= xm )
                        if x >= xm:
                            e = a-x
                        else:
                            e = b-x
                        d=CGOLD*e

                    else:
                        d=p/q
                        u=x+d

                        if (u-a < tol2 or b-u < tol2): 
                            d= abs(tol1)*np.sign(xm - x)  #sign(tol1,xm-x)

                else:

                    if x >= xm:
                        e = a-x
                    else:
                        e = b-x

                    d=CGOLD*e

                u = 0.  #merge(x+d,x+sign(tol1,d), abs(d) >= tol1 )
                if abs(d) >= tol1:
                    u = x+d
                else:
                    u = x+abs(tol1)*np.sign(d)

                fu = obj_loop_c(u, _EV_, _ia_, _ikap_ ,_istate_, _is_o_)

                if (fu <= fx):
                    if (u >= x):
                        a=x
                    else:
                        b=x

                    #shft(v,w,x,u)
                    v = w
                    w = x
                    x = u
                    #shft(fv,fw,fx,fu)
                    fv = fw
                    fw = fx
                    fx = fu


                else:
                    if (u < x):
                        a=u
                    else:
                        b=u

                    if fu <= fw or w == x:
                        v=w
                        fv=fw
                        w=u
                        fw=fu

                    elif fu <= fv or v == x or v == w:
                        v=u
                        fv=fu

            if it == itmax-1:
                print('brent: exceed maximum iterations')

            ans = xmin

            return ans

        @nb.jit(nopython = True)
        def _inner_loop_c_with_range_(assigned_indexes, _EV_, _vc_an_, _vcn_, _vc_util_, _is_o_):

        #     for istate in range(num_s):
        #         for ia in range(num_a):
        #             for ikap in range(num_kap):

            ibegin = assigned_indexes[0]
            iend = assigned_indexes[1]

            ind = 0
            for ipar_loop in range(ibegin, iend):

                #we should replace unravel_index_nb with something unflexible one.
                istate, ia, ikap = unravel_ip(ipar_loop)

                an_sup = min(cvals_supan[ia, is_to_ieps[istate], _is_o_] - 1.e-6, agrid[-1]) #no extrapolation for aprime

                ans = -10000.

                ans =  _inner_inner_loop_c_(an_sup, _EV_, ia, ikap ,istate, _is_o_)

                _vc_an_[ind] = ans
                _vcn_[ind] = -obj_loop_c(ans, _EV_, ia, ikap, istate, _is_o_)

                state = np.array([agrid[ia], ans, epsgrid[is_to_ieps[istate]], _is_o_]) 
        #         _vcn_[ind] = -obj_loop_c(ans, _EV_, ia, ikap, istate)
        # #         _vc_util_[ind] = get_cstatic([agrid[ia], ans, epsgrid[is_to_ieps[istate]]])[0]
        
        
                _vc_util_[ind] = get_cstatic(state)[0]

                ind = ind + 1

        @nb.jit(nopython = True)
#        @nb.jit(nopython = True, parallel = True)        
        def _howard_iteration_(_vmax_y_, _vmax_o_, _bEV_yc_, _bEV_oc_, _bEV_ys_, _bEV_os_, #these vars are just data containers
                               _vn_yc_, _vn_oc_, _vn_ys_, _vn_os_, #these value functions will be updated
                               _v_yc_util_, _v_oc_util_, _v_ys_util_, _v_os_util_,
                               _v_yc_an_, _v_oc_an_, _v_ys_an_, _v_os_an_,                               
                               _v_ys_kapn_, _v_os_kapn_,
                               _howard_iter_):

            #
            _v_y_succeeded_ = np.zeros(_vmax_y_.shape)

            _vn_ys_acq_ = np.zeros(_vmax_y_.shape)
            _vn_os_acq_ = np.zeros(_vmax_o_.shape)            
            
            for it_ho in range(_howard_iter_):

                ### update vn_acq ###
                for istate in range(num_s):
                    for ia in range(num_a):
                        for ikap in range(num_kap):

                            #need to take into account borrowing constraint
                            #an_tmp = vs_an[ia, ikap,istate] - pkap*kapbar
                            # a_tmp = agrid[ia] - pkap*kapbar
                            a_tmp = agrid[ia] - pkap# *kapbar                            

                            #update ys_acq
                            if a_tmp >= 0.: 
                                _vn_ys_acq_[ia, ikap, istate] = \
                                    fem2d_peval(a_tmp,  kapgrid[ikap] + kapbar , agrid, kapgrid, _vn_ys_[:,:,istate])
                            else:
                                #if an_tmp is not feasible (an_tmp < amin)
                                _vn_ys_acq_[ia, ikap, istate] = -np.inf
                            
                            #update os_acq
                            if a_tmp >= 0.: 
                                _vn_os_acq_[ia, ikap, istate] = \
                                    fem2d_peval(a_tmp,  kapgrid[ikap] + kapbar , agrid, kapgrid, _vn_os_[:,:,istate])
                            else:
                                #if an_tmp is not feasible (an_tmp < amin)
                                _vn_os_acq_[ia, ikap, istate] = -np.inf

                ### end update vn_acq ###
                

                
                #update the value functions
                _vmax_y_[:] = np.fmax(_vn_yc_, np.fmax(_vn_ys_, _vn_ys_acq_))
                _vmax_o_[:] = np.fmax(_vn_oc_, np.fmax(_vn_os_, _vn_os_acq_))
                
            

                #numba does not support matmul for 3D or higher matrice
                for ia in range(num_a):                
#                for ia in nb.prange(num_a):
                    _bEV_yc_[0,0,ia,:,:] = prob_yo[0,0]*bh*((_vmax_y_[ia,:,:]**(1. - mu))@(prob.T)).reshape((1, 1, 1, num_kap, num_s)) +\
                                           prob_yo[0,1]*bh*((_vmax_o_[ia,:,:]**(1. - mu))@(prob.T)).reshape((1, 1, 1, num_kap, num_s))

                for ia in range(num_a):
#                for ia in nb.prange(num_a):
                    _bEV_oc_[0,0,ia,:,:] = prob_yo[1,1]*bh*((_vmax_o_[ia,:,:]**(1. - mu))@(prob.T)).reshape((1, 1, 1, num_kap, num_s)) +\
                                           prob_yo[1,0]*iota*bh*((_vmax_y_[ia,:,:]**(1. - mu))@(prob_st)).reshape((1, 1, 1, num_kap, 1)) #does this reshape work?


                _bEV_ys_[:] = _bEV_yc_[:] 
                
                for istate in range(num_s):
#                for istate in nb.prange(num_s):
                    _v_y_succeeded_[:,:,istate] = fem2deval_mesh(agrid, la_tilde*kapgrid, agrid, kapgrid, _vmax_y_[:,:,istate])

                #if one dies, productivities are drawn from the stationary one.
                for ia in range(num_a):
#                for ia in nb.prange(num_a):
                    _bEV_os_[0,0,ia,:,:] = prob_yo[1,1]*bh*((_vmax_o_[ia,:,:]**(1. - mu))@(prob.T)).reshape((1, 1, 1, num_kap, num_s)) +\
                                           prob_yo[1,0]*iota*bh*((_v_y_succeeded_[ia,:,:]**(1. - mu))@(prob_st)).reshape((1, 1, 1, num_kap, 1)) #does this reshape work?

#                for istate in nb.prange(num_s):                
                for istate in range(num_s):
                    iz = is_to_iz[istate]
                    z = zgrid[iz]


                    for ia in range(num_a):
#                        for ikap in nb.prange(num_kap):                        
                        for ikap in range(num_kap):
                            _vn_yc_[ia, ikap, istate] = (_v_yc_util_[ia, ikap, istate] +
                                                         fem2d_peval(_v_yc_an_[ia, ikap, istate], la*kapgrid[ikap], agrid, kapgrid, _bEV_yc_[0,0,:,:,istate]) )**(1./(1. - mu))
                            _vn_oc_[ia, ikap, istate] = (_v_oc_util_[ia, ikap, istate] +
                                                         fem2d_peval(_v_oc_an_[ia, ikap, istate], la*kapgrid[ikap], agrid, kapgrid, _bEV_oc_[0,0,:,:,istate]) )**(1./(1. - mu))
                            _vn_ys_[ia, ikap, istate] = (_v_ys_util_[ia, ikap, istate] +
                                                         fem2d_peval(_v_ys_an_[ia, ikap, istate], _v_ys_kapn_[ia, ikap, istate], agrid, kapgrid, _bEV_ys_[0,0,:,:,istate]) )**(1./(1. - mu))
                            _vn_os_[ia, ikap, istate] = (_v_os_util_[ia, ikap, istate] +
                                                         fem2d_peval(_v_os_an_[ia, ikap, istate], _v_os_kapn_[ia, ikap, istate], agrid, kapgrid, _bEV_os_[0,0,:,:,istate]) )**(1./(1. - mu))

        @nb.jit(nopython = True)
        def reshape_to_mat(v, val):
            for i in range(len(val)):
                istate, ia, ikap = unravel_ip(i)
                v[ia, ikap, istate] = val[i]


        #initialize variables for VFI            
        v_yc_an_tmp = np.ones((num_assigned))
        vn_yc_tmp = np.ones((num_assigned))
        v_yc_util_tmp = np.ones((num_assigned))

        v_oc_an_tmp = np.ones((num_assigned))
        vn_oc_tmp = np.ones((num_assigned))
        v_oc_util_tmp = np.ones((num_assigned))
        
        v_ys_an_tmp = np.ones((num_assigned))
        v_ys_kapn_tmp = np.ones((num_assigned))
        vn_ys_tmp = np.ones((num_assigned))
        v_ys_util_tmp = np.ones((num_assigned))

        v_os_an_tmp = np.ones((num_assigned))
        v_os_kapn_tmp = np.ones((num_assigned))
        vn_os_tmp = np.ones((num_assigned))
        v_os_util_tmp = np.ones((num_assigned))
        


        v_yc_an_full = None
        vn_yc_full = None
        v_yc_util_full = None

        if rank == 0:
            v_yc_an_full = np.ones((num_total_state))
            vn_yc_full = np.ones((num_total_state))*(-2.)
            v_yc_util_full = np.ones((num_total_state))

        v_oc_an_full = None
        vn_oc_full = None
        v_oc_util_full = None

        if rank == 0:
            v_oc_an_full = np.ones((num_total_state))
            vn_oc_full = np.ones((num_total_state))*(-2.)
            v_oc_util_full = np.ones((num_total_state))
            


        v_ys_an_full = None
        v_ys_kapn_full = None
        vn_ys_full = None
        v_ys_util_full = None

        if rank == 0:
            v_ys_an_full = np.ones((num_total_state))
            v_ys_kapn_full = np.ones((num_total_state))
            vn_ys_full = np.ones((num_total_state))*(-2.)
            v_ys_util_full = np.ones((num_total_state))


        v_os_an_full = None
        v_os_kapn_full = None
        vn_os_full = None
        v_os_util_full = None

        if rank == 0:
            v_os_an_full = np.ones((num_total_state))
            v_os_kapn_full = np.ones((num_total_state))
            vn_os_full = np.ones((num_total_state))*(-2.)
            v_os_util_full = np.ones((num_total_state))


        v_y_max = np.ones((num_a, num_kap, num_s))
        v_y_maxn = np.ones((num_a, num_kap, num_s))*100.0
        v_y_maxm1 = np.ones(v_y_max.shape)

        v_o_max = np.ones((num_a, num_kap, num_s))
        v_o_maxn = np.ones((num_a, num_kap, num_s))*100.0
        v_o_maxm1 = np.ones(v_o_max.shape)

        
        bEV_yc = np.ones((1, 1, num_a, num_kap, num_s))
        bEV_oc = np.ones((1, 1, num_a, num_kap, num_s))
        bEV_ys = np.ones((1, 1, num_a, num_kap, num_s))
        bEV_os = np.ones((1, 1, num_a, num_kap, num_s))        

        v_yc_an = np.zeros((num_a, num_kap, num_s))
        vn_yc = np.ones((num_a, num_kap, num_s))*100.0
        v_yc_util = np.ones((num_a, num_kap, num_s))*100.0

        v_oc_an = np.zeros((num_a, num_kap, num_s))
        vn_oc = np.ones((num_a, num_kap, num_s))*100.0
        v_oc_util = np.ones((num_a, num_kap, num_s))*100.0
        

        v_ys_an = np.zeros((num_a, num_kap, num_s))
        v_ys_kapn = np.zeros((num_a, num_kap, num_s))
        vn_ys = np.ones((num_a, num_kap, num_s))*100.0
        v_ys_util = np.ones((num_a, num_kap, num_s))*100.0

        v_os_an = np.zeros((num_a, num_kap, num_s))
        #v_os_kapn does not take into account succession.
        #In the sumulatin part, kap will be replaced by la_tilde kap if it is succeeded
        v_os_kapn = np.zeros((num_a, num_kap, num_s)) 
        vn_os = np.ones((num_a, num_kap, num_s))*100.0
        v_os_util = np.ones((num_a, num_kap, num_s))*100.0


        vn_os_acq = np.ones((num_a, num_kap, num_s))*100.0        
        vn_ys_acq = np.ones((num_a, num_kap, num_s))*100.0
    
        R_y = np.ones((num_a, num_kap, num_s))*100.0
        R_o = np.ones((num_a, num_kap, num_s))*100.0        
        

        

        max_iter = 20
        max_howard_iter = 50 #needs to update
        tol = 1.0e-5
        dist = 10000.0
        dist_sub = 10000.0
        it = 0

        ###record some time###
        t1, t2, t3, t4,tyc1, tyc2, tys1, tys2, toc1, toc2, tos1, tos2 = 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.
        if rank == 0:
            t1 = time.time()
            t2 = t1

        ###main VFI iteration###
        if rank == 0:
            print('starting VFI...')
            
        while it < max_iter and dist > tol:

            if rank == 0:
                it = it + 1
                print(f'it = {it}', end = ', ')

                bEV_yc[:] = prob_yo[0,0]*bh*((v_y_max**(1. - mu))@(prob.T)).reshape((1, 1, num_a, num_kap, num_s)) +\
                            prob_yo[0,1]*bh*((v_o_max**(1. - mu))@(prob.T)).reshape((1, 1, num_a, num_kap, num_s))

                bEV_oc[:] = prob_yo[1,1]*bh*((v_o_max**(1. - mu))@(prob.T)).reshape((1, 1, num_a, num_kap, num_s)) +\
                            prob_yo[1,0]*iota*bh*((v_y_max**(1. - mu))@(prob_st)).reshape((1, 1, num_a, num_kap, 1)) #does this reshape work?



                #under the current structure, the following is true. If not, it should be calculated separetely
                bEV_ys[:] = bEV_yc[:]
                
#                bEV_ys[:] = prob_yo[0,0]*bh*((v_y_max**(1. - mu))@(prob.T)).reshape((1, 1, num_a, num_kap, num_s)) +\
#                            prob_yo[0,1]*bh*((v_o_max**(1. - mu))@(prob.T)).reshape((1, 1, num_a, num_kap, num_s))

                v_y_succeeded = np.zeros(v_y_max.shape)

                for istate in range(num_s):
                    v_y_succeeded[:,:,istate] = fem2deval_mesh(agrid, la_tilde*kapgrid, agrid, kapgrid, v_y_max[:,:,istate])

                    

                #if one dies, productivities are drawn from the stationary one.
                bEV_os[:] = prob_yo[1,1]*bh*((v_o_max**(1. - mu))@(prob.T)).reshape((1, 1, num_a, num_kap, num_s)) +\
                            prob_yo[1,0]*iota*bh*((v_y_succeeded**(1. - mu))@(prob_st)).reshape((1, 1, num_a, num_kap, 1)) #does this reshape work?



            comm.Bcast([bEV_yc, MPI.DOUBLE])
            comm.Bcast([bEV_oc, MPI.DOUBLE])
            comm.Bcast([bEV_ys, MPI.DOUBLE])                        
            comm.Bcast([bEV_os, MPI.DOUBLE])            



            
            ###yc-loop begins####            	
            comm.Barrier()
            if rank == 0:
                tyc1 = time.time()

            #_is_o_ = 0
            _inner_loop_c_with_range_(assigned_state_range, bEV_yc, v_yc_an_tmp, vn_yc_tmp, v_yc_util_tmp, 0)
                    

            comm.Barrier()
            if rank == 0:
                tyc2 = time.time()
                print('time for yc = {:f}'.format(tyc2 - tyc1), end = ', ')
            ###yc-loop ends####

            ###oc-loop begins####            	
            comm.Barrier()
            if rank == 0:
                toc1 = time.time()

            #_is_o_ = 1                
            _inner_loop_c_with_range_(assigned_state_range, bEV_oc, v_oc_an_tmp, vn_oc_tmp, v_oc_util_tmp, 1)
            
                       # ind_s_util_finemesh_cached_o, ind_s_util_finemesh_cached_y, s_util_finemesh_cached_o, s_util_finemesh_cached_y) #_is_o_ = True

            comm.Barrier()
            if rank == 0:
                toc2 = time.time()
                print('time for oc = {:f}'.format(toc2 - toc1), end = ', ')
            ###oc-loop ends####

            
            ###ys-loop begins####
            comm.Barrier()
            if rank == 0:
                tys1 = time.time()


            num_cached_y = _inner_loop_s_with_range_(assigned_state_range, bEV_ys, v_ys_an_tmp ,v_ys_kapn_tmp, vn_ys_tmp, v_ys_util_tmp, num_cached_y, 0, ind_s_util_finemesh_cached_o, ind_s_util_finemesh_cached_y, s_util_finemesh_cached_o, s_util_finemesh_cached_y) #_is_o_ = False) 

            comm.Barrier()
            if rank == 0:
                tys2 = time.time()
                print('time for ys = {:f}'.format(tys2 - tys1), end = ', ')
            ###ys-loop ends####

            ###os-loop begins####
            comm.Barrier()
            if rank == 0:
                tos1 = time.time()


            num_cached_o = _inner_loop_s_with_range_(assigned_state_range, bEV_os, v_os_an_tmp ,v_os_kapn_tmp, vn_os_tmp, v_os_util_tmp, num_cached_o, 1, ind_s_util_finemesh_cached_o, ind_s_util_finemesh_cached_y, s_util_finemesh_cached_o, s_util_finemesh_cached_y) #_is_o_ = False) 

            comm.Barrier()
            if rank == 0:
                tos2 = time.time()
                print('time for os = {:f}'.format(tos2 - tos1), end = ', ')
            ###os-loop ends####

           


            ####policy function iteration starts#####

            comm.Gatherv(vn_yc_tmp,[vn_yc_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(vn_oc_tmp,[vn_oc_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])            

            comm.Gatherv(vn_ys_tmp,[vn_ys_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(vn_os_tmp,[vn_os_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            
            comm.Gatherv(v_yc_an_tmp,[v_yc_an_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(v_oc_an_tmp,[v_oc_an_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            
            comm.Gatherv(v_ys_an_tmp,[v_ys_an_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(v_os_an_tmp,[v_os_an_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])            
            
            comm.Gatherv(v_ys_kapn_tmp,[v_ys_kapn_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(v_os_kapn_tmp,[v_os_kapn_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            
            comm.Gatherv(v_yc_util_tmp,[v_yc_util_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(v_oc_util_tmp,[v_oc_util_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])            

            comm.Gatherv(v_ys_util_tmp,[v_ys_util_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(v_os_util_tmp,[v_os_util_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])            
            

            if rank == 0:
                reshape_to_mat(v_yc_an, v_yc_an_full)
                reshape_to_mat(v_oc_an, v_oc_an_full)
                
                reshape_to_mat(v_ys_an, v_ys_an_full)
                reshape_to_mat(v_os_an, v_os_an_full)
                
                reshape_to_mat(v_ys_kapn, v_ys_kapn_full)
                reshape_to_mat(v_os_kapn, v_os_kapn_full)
                
                reshape_to_mat(v_yc_util, v_yc_util_full)
                reshape_to_mat(v_oc_util, v_oc_util_full)                
                
                reshape_to_mat(v_ys_util, v_ys_util_full)
                reshape_to_mat(v_os_util, v_os_util_full)
                
                reshape_to_mat(vn_yc, vn_yc_full)
                reshape_to_mat(vn_oc, vn_oc_full)
                
                reshape_to_mat(vn_ys, vn_ys_full)
                reshape_to_mat(vn_os, vn_os_full)                

                #life cycle will be added later on
                if max_howard_iter > 0:
                    #print('Starting Howard Iteration...')
                    t3 = time.time()
                    _howard_iteration_(v_y_maxn, v_o_maxn, bEV_yc, bEV_oc, bEV_ys, bEV_os, #these vars are just data containers
                                       vn_yc, vn_oc, vn_ys, vn_os, #these value functions will be updated 
                                       v_yc_util, v_oc_util, v_ys_util, v_os_util,
                                       v_yc_an, v_oc_an, v_ys_an, v_os_an,                               
                                       v_ys_kapn, v_os_kapn,
                                       max_howard_iter)
            
                    

                #_howard_iteration_(vmaxn, vcn, vsn, vc_an, vc_util, vs_an, vs_kapn, vs_util ,max_howard_iter)

                if max_howard_iter > 0:
                    t4 = time.time()
                    print('time for HI = {:f}'.format(t4 - t3), end = ', ') 

            ####policy function iteration ends#####


            ####post_calc
            if rank == 0:


                ###calc val for acquisiiton###

                for istate in range(num_s):
                    for ia in range(num_a):
                        for ikap in range(num_kap):

                            #need to take into account borrowing constraint
                            a_tmp = agrid[ia] - pkap

                            #update ys_acq
                            if a_tmp >= 0.: 
                                vn_ys_acq[ia, ikap, istate] = \
                                    fem2d_peval(a_tmp,  kapgrid[ikap] + kapbar , agrid, kapgrid, vn_ys[:,:,istate])
                            else:
                                #if an_tmp is not feasible (an_tmp < amin)
                                vn_ys_acq[ia, ikap, istate] = -np.inf
                            
                            #update os_acq
                            if a_tmp >= 0.: 
                                vn_os_acq[ia, ikap, istate] = \
                                    fem2d_peval(a_tmp,  kapgrid[ikap] + kapbar , agrid, kapgrid, vn_os[:,:,istate])
                            else:
                                #if an_tmp is not feasible (an_tmp < amin)
                                vn_os_acq[ia, ikap, istate] = -np.inf
                                

                ###end calc val for acquisiiton###
                

                pol_yc = vn_yc > vn_ys
                pol_ys_acq = vn_ys_acq > vn_ys
                v_y_maxn[:] = np.fmax(vn_yc, np.fmax(vn_ys, vn_ys_acq))

                pol_oc = vn_oc > vn_os
                pol_os_acq = vn_os_acq > vn_os
                v_o_maxn[:] = np.fmax(vn_oc, np.fmax(vn_os, vn_os_acq))


                #we can also measure distance in terms 
                dist_y_sub = np.max(np.abs(v_y_maxn - v_y_max))
                dist_o_sub = np.max(np.abs(v_o_maxn - v_o_max))
                dist_sub = max(dist_y_sub, dist_o_sub)
                
                dist_y = np.max(np.abs(v_y_maxn - v_y_max) / (1. + np.abs(v_y_maxn)))
                dist_o = np.max(np.abs(v_o_maxn - v_o_max) / (1. + np.abs(v_o_maxn)))
                dist = max(dist_y, dist_o)
                print('')
                print('{}th loop. dist = {:f}, dist_sub = {:f}.'.format(it, dist, dist_sub), end = ',')
                v_y_maxm1[:] = v_y_max[:]
                v_o_maxm1[:] = v_o_max[:]
                v_y_max[:] = v_y_maxn[:]
                v_o_max[:] = v_o_maxn[:]                
                

            it = comm.bcast(it)
            dist = comm.bcast(dist)
            dist_sub = comm.bcast(dist_sub)

            if rank ==0:
                print(f' time = {time.time() - t2}.')
                t2 = time.time()
        ###end main VFI###

        ###post-calc###
        if rank == 0:
            t2 = time.time()
            print('Iteration terminated at {}th loop. dist = {}'.format(it, dist))
            print('Elapsed time: {:f} sec'.format(t2 - t1)) 

        ###calc fair price R(a, kap, s, age)###

        if rank == 0:

            t1 = time.time()
            from scipy.optimize import brentq
            
            for istate in range(num_s):
                for ia in range(num_a):
                    for ikap in range(num_kap):

                        #young
                        
                        obj = lambda x: femeval(agrid[ia] + (1. - taucg)*x, agrid, vn_yc[:, 0, istate]) - vn_ys[ia, ikap, istate]
                        fa = obj(0.)
                        fb = obj(agrid[-1])

                        
                        if fa*fb >= 0.:
                            if abs(fa) <= 1.0e-8 or vn_yc[ia, 0, istate] - vn_ys[ia, ikap, istate] >= 0.:
                                R_y[ia, ikap, istate] = 0.0
                            else:

                                print('a =' , agrid[ia])
                                print('kap =' , kapgrid[ikap])
                                print('istate =' , istate)
                                print('young')
                                print('fa = ', fa)
                                print('fb = ', fb)

                        else:

                            R_y[ia, ikap, istate] = brentq(obj, 0., agrid[-1])

                        
                        obj = lambda x: femeval(agrid[ia] + (1. - taucg)*x, agrid, vn_oc[:, 0, istate]) - vn_os[ia, ikap, istate]
                        fa = obj(0.)
                        fb = obj(agrid[-1])
                        if fa*fb >= 0.:
                            if abs(fa) <= 1.0e-8 or vn_oc[ia, 0, istate] - vn_os[ia, ikap, istate] >= 0.:
                                R_o[ia, ikap, istate] = 0.0
                            else:

                                print('a =' , agrid[ia])
                                print('kap =' , kapgrid[ikap])
                                print('istate =' , istate)
                                print('old')
                                print('fa = ', fa)
                                print('fb = ', fb)

                        else:

                            R_o[ia, ikap, istate] = brentq(obj, 0., agrid[-1])

            t2 = time.time()

            print(f'Elapsed time for calculating R(s) = {t2 -t1}')
                            

        ###end calc fair price R(a, kap, s, age)###
            



        #return value and policy functions

        ###Bcast the result###

        comm.Bcast([v_yc_an, MPI.DOUBLE])
        comm.Bcast([v_oc_an, MPI.DOUBLE])
        comm.Bcast([v_ys_an, MPI.DOUBLE])
        comm.Bcast([v_os_an, MPI.DOUBLE])

        #There is no y_yc_kapn or y_oc_kapn because we analytically know the transition.
        #if we change the transition, we may need to add them
        comm.Bcast([v_ys_kapn, MPI.DOUBLE])
        comm.Bcast([v_os_kapn, MPI.DOUBLE])        
        comm.Bcast([vn_ys, MPI.DOUBLE])
        comm.Bcast([vn_os, MPI.DOUBLE])
        comm.Bcast([vn_ys_acq, MPI.DOUBLE])
        comm.Bcast([vn_os_acq, MPI.DOUBLE])        
        comm.Bcast([vn_yc, MPI.DOUBLE])
        comm.Bcast([vn_oc, MPI.DOUBLE])
        comm.Bcast([R_y, MPI.DOUBLE])
        comm.Bcast([R_o, MPI.DOUBLE])        



        #return policy function
        self.v_yc_an = v_yc_an
        self.v_oc_an = v_oc_an        
        self.v_ys_an = v_ys_an
        self.v_os_an = v_os_an        
        self.v_ys_kapn = v_ys_kapn
        self.v_os_kapn = v_os_kapn        
        self.vn_yc = vn_yc
        self.vn_oc = vn_oc        
        self.vn_ys = vn_ys
        self.vn_os = vn_os
        self.vn_ys_acq = vn_ys_acq
        self.vn_os_acq = vn_os_acq        

        self.R_y = R_y
        self.R_o = R_o

        self.bEV_yc = bEV_yc
        self.bEV_oc = bEV_oc
        self.bEV_ys = bEV_ys 
        self.bEV_os = bEV_os 
        

    def generate_shocks(self):

        """
        return:
        data_is_o
        data_i_s
        """

        prob = self.prob
        prob_yo = self.prob_yo
        prob_st = self.prob_st
        num_s = self.num_s

        #simulation parameters
        sim_time = self.sim_time
        num_total_pop = self.num_total_pop
        
        
        
        ###codes to generate shocks###
        @nb.jit(nopython = True)
        def transit(i, r, _prob_):
            num_s = _prob_.shape[0]

            if r <= _prob_[i,0]:
                return 0

            for j in range(1, num_s):

                #print(np.sum(_prob_[i,0:j]))
                if r <= np.sum(_prob_[i,0:j]):
                    return j - 1

            if r > np.sum(_prob_[i,0:-1]) and r <= 1.:
                return num_s - 1

            print('error')

            return -1

        @nb.jit(nopython = True)
        def draw(r, _prob_st_):
            num_s = len(_prob_st_)

            if r <= _prob_st_[0]:
                return 0

            for j in range(1, num_s):

                #print(np.sum(_prob_st_[0:j]))
                if r <= np.sum(_prob_st_[0:j]):
                    return j - 1

            if r > np.sum(_prob_st_[0:-1]) and r <= 1.:
                return num_s - 1

            print('error')

            return -1    
        

        np.random.seed(1) #fix the seed
        data_rnd_s = np.random.rand(num_total_pop, 2*sim_time+1) 
        data_rnd_yo = np.random.rand(num_total_pop,2*sim_time+1)
        
        @nb.jit(nopython = True, parallel = True)
        #@nb.jit(nopython = True)        
        def calc_trans(_data_, _rnd_, _prob_):
            num_entity, num_time = _rnd_.shape

#            for i in range(num_entity):
            for i in nb.prange(num_entity):                
                for t in range(1, num_time):                    
                    
                    _data_[i, t] = transit(_data_[i, t-1], _rnd_[i, t], _prob_)

        print('generating is_o...')
        data_is_o = np.zeros((num_total_pop, 2*sim_time+1), dtype = int) #we can't set dtype = bool
        calc_trans(data_is_o, data_rnd_yo, prob_yo)
        print('done')

        #transition of s = (eps, z) depends on young-old transition.
        @nb.jit(nopython = True, parallel = True)
#        @nb.jit(nopython = True)        
        def calc_trans_shock(_data_, _data_is_o_, _rnd_s_, _prob_s_, _prob_s_st_):
            num_entity, num_time = _rnd_s_.shape

            for i in nb.prange(num_entity):
#            for i in range(num_entity):
                for t in range(1, num_time):
                    is_o = _data_is_o_[i,t]
                    is_o_m1 = _data_is_o_[i,t-1]

                    if is_o_m1 and not (is_o): #if s/he dies and reborns, prod. shocks are drawn from the stationary dist
                        _data_[i, t] = draw(_rnd_s_[i, t], _prob_s_st_)
                        
                    else:
                        _data_[i, t] = transit(_data_[i, t-1], _rnd_s_[i, t], _prob_s_)
        print('generating i_s...')
        data_i_s = np.ones((num_total_pop, 2*sim_time+1), dtype = int) * (num_s // 2)
        calc_trans_shock(data_i_s, data_is_o, data_rnd_s, prob, prob_st)
        print('done')
        
        return data_i_s[:, sim_time:2*sim_time], data_is_o[:, sim_time:2*sim_time+1]
   
        
    #def get_obj(w, p, rc, vc_an, vs_an, vs_kapn, vcn, vsn):
    def simulate_model(self):
        Econ = self

        for variable in self.__dict__ : exec(variable+'= self.'+variable, locals(), globals())        
        #load the value functions

        v_yc_an = self.v_yc_an
        v_oc_an = self.v_oc_an        
        v_ys_an = self.v_ys_an
        v_os_an = self.v_os_an        
        v_ys_kapn = self.v_ys_kapn
        v_os_kapn = self.v_os_kapn
        
        vn_yc = self.vn_yc
        vn_oc = self.vn_oc        
        vn_ys = self.vn_ys
        vn_os = self.vn_os
        vn_ys_acq = self.vn_ys_acq
        vn_os_acq = self.vn_os_acq

        R_y = self.R_y
        R_o = self.R_o        
        

        vn_y = np.fmax(vn_yc, np.fmax(vn_ys, vn_ys_acq))
        vn_o = np.fmax(vn_oc, np.fmax(vn_os, vn_os_acq))

        #for c-corp guys, it is always true that kapn = la*kap
        bEV_yc = prob_yo[0,0]*bh*((vn_y**(1. - mu))@(prob.T)).reshape((1, 1, num_a, num_kap, num_s)) +\
                 prob_yo[0,1]*bh*((vn_o**(1. - mu))@(prob.T)).reshape((1, 1, num_a, num_kap, num_s))

        bEV_oc = prob_yo[1,1]*bh*((vn_o**(1. - mu))@(prob.T)).reshape((1, 1, num_a, num_kap, num_s)) +\
                 prob_yo[1,0]*iota*bh*((vn_y**(1. - mu))@(prob_st)).reshape((1, 1, num_a, num_kap, 1)) #?

        #under the current structure, the following is true. If not, it should be calculated separetely
        bEV_ys = bEV_yc.copy()
                
        v_y_succeeded = np.zeros(vn_y.shape)

        for istate in range(num_s):
            v_y_succeeded[:,:,istate] = fem2deval_mesh(agrid, la_tilde*kapgrid, agrid, kapgrid, vn_y[:,:,istate])

            
        bEV_os = prob_yo[1,1]*bh*((vn_o**(1. - mu))@(prob.T)).reshape((1, 1, num_a, num_kap, num_s)) +\
                 prob_yo[1,0]*iota*bh*((v_y_succeeded**(1. - mu))@(prob_st)).reshape((1, 1, num_a, num_kap, 1)) #?


        @nb.jit(nopython = True)
        def unravel_ip(i_aggregated_state):

            istate, ia, ikap = unravel_index_nb(i_aggregated_state, num_s, num_a, num_kap)
            #ia, ikap, istate = unravel_index_nb(i_aggregated_state, num_a, num_kap, num_s)
            return istate, ia, ikap

        get_cstatic = Econ.generate_cstatic()
        get_sstatic = Econ.generate_sstatic()
        ### start parameters for MPI ###
        m = num_total_pop // size
        r = num_total_pop % size

        assigned_pop_range =  (rank*m+min(rank,r)), ((rank+1)*m+min(rank+1,r))
        num_pop_assigned = assigned_pop_range[1] - assigned_pop_range[0]


        all_assigned_pop_range = np.ones((size, 2))*(-2.)
        all_num_pop_assigned = ()
        all_istart_pop_assigned = ()

        for irank in range(size):

            all_istart_pop_assigned += (int(irank*m+min(irank,r)), )
            all_assigned_pop_range[irank,0] = irank*m+min(irank,r)
            all_assigned_pop_range[irank,1] = (irank+1)*m+min(irank+1,r)
            all_num_pop_assigned += (int(all_assigned_pop_range[irank,1] - all_assigned_pop_range[irank,0]),) 
        ### end parameters for MPI ###    

        #data container for each node
        data_a_elem = np.ones((num_pop_assigned, sim_time))*4.0
        data_kap_elem = np.ones((num_pop_assigned, sim_time))*0.0
        data_kap0_elem = np.ones((num_pop_assigned, sim_time))*0.0        
        data_i_s_elem = np.ones((num_pop_assigned, sim_time), dtype = int)*7
        data_is_c_elem = np.zeros((num_pop_assigned, sim_time), dtype = bool) 
        data_is_c_elem[0:int(num_pop_assigned*0.7), 0] = True
        data_is_o_elem = np.zeros((num_pop_assigned, sim_time+1), dtype = bool)
        data_sales_shock_elem = np.zeros((num_pop_assigned, sim_time), dtype = bool)
        data_option_elem = np.ones((num_pop_assigned, sim_time), dtype = int)*(-1)
        data_R_elem = np.ones((num_pop_assigned, sim_time))*(-10000.)        


        data_a = None
        data_kap = None
        data_kap0 = None
        data_i_s = None
        data_is_c = None
        data_is_o = None
        data_R = None

        data_sales_shock = None
        data_option = None



        if rank == 0:
            data_a = np.zeros((num_total_pop, sim_time))
            data_kap = np.zeros((num_total_pop, sim_time))
            data_kap0 = np.zeros((num_total_pop, sim_time))            
            data_i_s = np.zeros((num_total_pop, sim_time), dtype = int)
            data_is_c = np.zeros((num_total_pop, sim_time), dtype = bool)
            data_is_o = np.zeros((num_total_pop, sim_time+1), dtype = bool)
            data_sales_shock = np.zeros((num_total_pop, sim_time), dtype = bool)
            data_option = np.zeros((num_total_pop, sim_time), dtype = int)
            data_R = np.zeros((num_total_pop, sim_time))*(-10000.)        

    
        ###load productivity shock data###

        
        data_i_s_elem[:] = np.load(self.path_to_data_i_s + '_' + str(rank) + '.npy')
        data_is_o_elem[:] = np.load(self.path_to_data_is_o + '_' + str(rank) + '.npy')
        data_sales_shock_elem[:] = np.load(self.path_to_data_sales_shock + '_' + str(rank) + '.npy')        
        
        # #check the dimension of data_is_o
        # #I assume that data_is_o.shape[1] >  sim_time+1

        # if rank == 0:

        #     if data_is_o.shape[1]  <  sim_time+1:
        #         print('data_is_o.shape[1]  <  sim_time+1:')
        #         print('code will be terminated...')
        #         return None
        


        @nb.jit(nopython = True)
        def calc(data_a_, data_kap_, data_kap0_ ,data_i_s_, data_is_o_ ,data_is_c_, data_sales_shock_, data_option_, data_R_):

            
            for i in range(num_pop_assigned):
                for t in range(1, sim_time):
                    a = data_a_[i, t-1]
                    kap = data_kap0_[i, t-1]

                    is_o = data_is_o_[i,t]
                    is_o_m1 = data_is_o_[i,t-1]
                    is_c_m1 = data_is_c_[i,t-1] 
                    istate = data_i_s_[i, t]
                    is_sold = data_sales_shock_[i,t]

                    eps = epsgrid[is_to_ieps[istate]]
                    z = zgrid[is_to_iz[istate]]

                    #replace kap with la_tilde kap if it was succeeded from old S-guy
                    #that is, if he was old in t-1 and S-guy, he is young in t.
                    if is_o_m1 and (not is_o) and (not is_c_m1):
                        kap = la_tilde * kap

                    #save kap here.
                    data_kap_[i, t-1] = kap


                    a_acq = a - pkap #can be negative. need to check feasibility.
                    kap_acq = kap + kapbar


                    
                    R = -100.
                    if is_o:
                        #this linear interpolation is a bit dangerous.
                        #it is better to find an optimal R everytime.
                        R = fem2d_peval(a, kap, agrid, kapgrid, R_o[:,:,istate])

                    else:
                        R = fem2d_peval(a, kap, agrid, kapgrid, R_y[:,:,istate])

                    

                    a_sale = a + (1.-taucg)*R
                    kap_sale = 0.

                    option = -1

                        
                        
                    if not is_o: #if young

                        
                        an_c = fem2d_peval(a, kap, agrid, kapgrid, v_yc_an[:,:,istate])
                        kapn_c = la*kap #fem2d_peval(a, kap, agrid, kapgrid, vc_kapn[:,:,istate]) #or lambda * kap
                        #kapn_c = fem2d_peval(a, kap, agrid, kapgrid, vc_kapn[:,:,istate])

                        an_s = fem2d_peval(a, kap, agrid, kapgrid, v_ys_an[:,:,istate])
                        kapn_s = fem2d_peval(a, kap, agrid, kapgrid, v_ys_kapn[:,:,istate])
                        #if we dont 'want to allow for extraplation
                        #kapn_s = max((1. - delkap)/(1.+grate)*kap, fem2d_peval(a, kap, agrid, kapgrid, vs_kapn[:,:,istate]))


                        an_c_sale = fem2d_peval(a_sale, kap_sale, agrid, kapgrid, v_yc_an[:,:,istate])
                        kapn_c_sale = la*kap_sale

                        an_acq = fem2d_peval(a_acq, kap_acq, agrid, kapgrid, v_ys_an[:,:,istate])
                        kapn_acq = fem2d_peval(a_acq, kap_acq, agrid, kapgrid, v_ys_kapn[:,:,istate])


                        val_c = (get_cstatic([a, an_c, eps, is_o])[0] + fem2d_peval(an_c, kapn_c, agrid, kapgrid, bEV_yc[0, 0, :, :, istate]))**(1./(1.- mu))
                        val_s = (get_sstatic([a, an_s, kap, kapn_s, z, is_o])[0] + fem2d_peval(an_s, kapn_s, agrid, kapgrid, bEV_ys[0, 0, : ,: ,istate]))**(1./(1.- mu))

                        val_acq = -np.inf
                        if a_acq >= 0.0:
                            val_acq = (get_sstatic([a_acq, an_acq, kap_acq, kapn_acq, z, is_o])[0] + fem2d_peval(an_acq, kapn_acq, agrid, kapgrid, bEV_ys[0, 0, : ,: ,istate]))**(1./(1.- mu))

                    else: #elif old

                        an_c = fem2d_peval(a, kap, agrid, kapgrid, v_oc_an[:,:,istate])
                        kapn_c = la*kap #fem2d_peval(a, kap, agrid, kapgrid, vc_kapn[:,:,istate]) #or lambda * kap
                        #kapn_c = fem2d_peval(a, kap, agrid, kapgrid, vc_kapn[:,:,istate])

                        an_s = fem2d_peval(a, kap, agrid, kapgrid, v_os_an[:,:,istate])
                        kapn_s = fem2d_peval(a, kap, agrid, kapgrid, v_os_kapn[:,:,istate]) #this kapn does not take into account
                        #if we dont 'want to allow for extraplation
                        #kapn_s = max((1. - delkap)/(1.+grate)*kap, fem2d_peval(a, kap, agrid, kapgrid, vs_kapn[:,:,istate]))


                        an_acq = fem2d_peval(a_acq, kap_acq, agrid, kapgrid, v_os_an[:,:,istate])
                        kapn_acq = fem2d_peval(a_acq, kap_acq, agrid, kapgrid, v_os_kapn[:,:,istate])

                        an_c_sale = fem2d_peval(a_sale, kap_sale, agrid, kapgrid, v_oc_an[:,:,istate])
                        kapn_c_sale = la*kap_sale
                        

                        val_c = (get_cstatic([a, an_c, eps, is_o])[0] + fem2d_peval(an_c, kapn_c, agrid, kapgrid, bEV_oc[0, 0, :,:,istate])) **(1./(1.- mu))
                        val_s = (get_sstatic([a, an_s, kap, kapn_s, z, is_o])[0] + fem2d_peval(an_s, kapn_s, agrid, kapgrid, bEV_os[0, 0, :,:,istate])) **(1./(1.- mu))

                        val_acq = -np.inf
                        if a_acq >= 0.0:
                            val_acq = (get_sstatic([a_acq, an_acq, kap_acq, kapn_acq, z, is_o])[0] + fem2d_peval(an_acq, kapn_acq, agrid, kapgrid, bEV_os[0, 0, : ,: ,istate]))**(1./(1.- mu))
                        

                        

                    #if val_c == val_s:
                    #    print('error: val_c == val_s')

                    is_c = False
                    is_s = False
                    is_acq = False

                    max_posi = np.argmax(np.array([val_c, val_s, val_acq]))
                    
                    if max_posi == 0:
                        is_c = True
                    elif max_posi == 1:
                        is_s = True
                    elif max_posi == 2:
                        is_acq = True
                    else:
                        print('error')

                    # del max_posi #del is not implemented

                    if is_s:
                        if is_sold:
                            # switching this case.
                            option = 0                            
                        else:
                            option = 2
                    elif is_c:
                        option = 1
                    elif is_acq:
                        option = 3
                    else:
                        print('error in determining option')


                    if is_s and is_sold:
                        is_s = False
                        is_c = True

                    # if is_c:
                    #     if is_sold:
                    #         option = 0
                    #     else:
                    #         option = 1
                    # elif is_s:
                    #     option = 2
                    # elif is_acq:
                    #     option = 3
                    # else:
                    #     print('error in determining option')
                        
                    

                    an = is_c * an_c + is_s * an_s + is_acq * an_acq
                    kapn = is_c * kapn_c + is_s * kapn_s + is_acq * kapn_acq

                    if is_c and is_sold:
                        an = an_c_sale
                        kapn = kapn_c_sale

                    
                    data_a_[i, t] = an
                    # data_kap_[i, t] = kapn # we do not save here,...
                    data_kap0_[i, t] = kapn
                    data_is_c_[i, t] = is_c
                    data_option_[i,t] = option
                    data_R_[i, t] = R

                # for t = sim_time+1:
                # we need this part since decrease in kap due to succeession depends on realization of a elderly shock

                t = sim_time

                kap = data_kap0_[i, t-1]
                is_o = data_is_o_[i,t]
                is_o_m1 = data_is_o_[i,t-1]
                is_c_m1 = data_is_c_[i,t-1] 
            
                #replace kap with la_tilde kap if it was succeeded from old S-guy
                #that is, if he was old in t-1 and S-guy, he is young in t.
                if is_o_m1 and (not is_o) and (not is_c_m1):
                    kap = la_tilde * kap

                #save
                data_kap_[i, t-1] = kap

            
        calc(data_a_elem, data_kap_elem, data_kap0_elem ,data_i_s_elem, data_is_o_elem,
             data_is_c_elem, data_sales_shock_elem, data_option_elem, data_R_elem)


        comm.Gatherv(data_a_elem, [data_a, all_num_pop_assigned, all_istart_pop_assigned,  MPI.DOUBLE.Create_contiguous(sim_time).Commit() ])
        comm.Gatherv(data_kap_elem, [data_kap, all_num_pop_assigned, all_istart_pop_assigned,  MPI.DOUBLE.Create_contiguous(sim_time).Commit() ])
        comm.Gatherv(data_kap0_elem, [data_kap0, all_num_pop_assigned, all_istart_pop_assigned,  MPI.DOUBLE.Create_contiguous(sim_time).Commit() ])        
        comm.Gatherv(data_i_s_elem, [data_i_s, all_num_pop_assigned, all_istart_pop_assigned,  MPI.LONG.Create_contiguous(sim_time).Commit() ])   
        comm.Gatherv(data_is_c_elem, [data_is_c, all_num_pop_assigned, all_istart_pop_assigned,  MPI.BOOL.Create_contiguous(sim_time).Commit() ])
        comm.Gatherv(data_is_o_elem, [data_is_o, all_num_pop_assigned, all_istart_pop_assigned,  MPI.BOOL.Create_contiguous(sim_time+1).Commit() ])
        comm.Gatherv(data_sales_shock_elem, [data_sales_shock, all_num_pop_assigned, all_istart_pop_assigned,  MPI.BOOL.Create_contiguous(sim_time).Commit() ])
        comm.Gatherv(data_option_elem, [data_option, all_num_pop_assigned, all_istart_pop_assigned,  MPI.LONG.Create_contiguous(sim_time).Commit() ])
        comm.Gatherv(data_R_elem, [data_R, all_num_pop_assigned, all_istart_pop_assigned,  MPI.DOUBLE.Create_contiguous(sim_time).Commit() ])        

    

        #calculate other variables

        data_ss = None

        if rank == 0:

            data_ss = np.ones((num_total_pop, 22)) * (-2.0)

            t =  sim_time - 1 #don't set t = -1 since data_is_o has different length
            for i in range(num_total_pop):

                #need to check the consistency within variables... there may be errors...
#                 if np.isin(data_option[i, t], [0,1]):
                if data_is_c[i, t]:                    

                    a = data_a[i, t-1]
                    kap = data_kap[i, t-1]
                    an = data_a[i, t]
                    kapn = data_kap0[i, t] #this must be kap0, not kap
                    eps = epsgrid[is_to_ieps[data_i_s[i, t]]]
                    is_o = data_is_o[i, t]
                    R = data_R[i,t]

                    atilde = a
                    kaptilde = kap
                    
                    #R can be negative
                    if data_option[i,t] == 0 and R > 0.: #if the option is 0 (C-and sale kappa) and R has a positive value
                        atilde = a + (1.-taucg)*R
                        kaptilde = 0.0
                        

                    data_ss[i,0] = data_option[i,t]
                    data_ss[i,1] = a
                    data_ss[i,2] = kap
                    data_ss[i,3] = an
                    data_ss[i,4] = kapn
                    data_ss[i,5] = eps*(1.-is_o) + tau_wo*eps*is_o

                    tmp = get_cstatic(np.array([atilde, an, eps, float(is_o)]))
                    data_ss[i,6:11] = tmp[1:6]
                    data_ss[i,17:20] = tmp[6:9]
                    data_ss[i,20] = is_o
                    data_ss[i,21] = R

#               elif np.isin(data_option[i, t], [2,3]):                    
                else:

                    a = data_a[i, t-1]
                    kap = data_kap[i, t-1]
                    an = data_a[i, t]
                    kapn = data_kap0[i, t] #this must be kap0, not kap
                    z = zgrid[is_to_iz[data_i_s[i, t]]]
                    is_o = data_is_o[i, t]

                    atilde = a
                    kaptilde = kap
                    
                    if data_option[i,t] == 3:
                        atilde = a - pkap
                        kaptilde = kap + kapbar
                        if atilde < 0.:
                            print('simulation error: atilde is negative')
                    

                    data_ss[i,0] = data_option[i,t]
                    data_ss[i,1] = a
                    data_ss[i,2] = kap
                    data_ss[i,3] = an
                    data_ss[i,4] = kapn
                    data_ss[i,5] = z*(1. - is_o) + tau_bo*z*is_o
                    
                    tmp = get_sstatic(np.array([atilde, an, kaptilde, kapn, z, float(is_o)]))
                    data_ss[i,6:20] = tmp[1:]
                    data_ss[i,20] = is_o                    
                    data_ss[i,21] = -1000000.




        self.data_a = data_a
        self.data_kap = data_kap
        self.data_kap0 = data_kap0        
        self.data_i_s = data_i_s
        self.data_is_c = data_is_c
        self.data_is_o = data_is_o
        self.data_option = data_option
        self.data_sales_shock = data_sales_shock
        self.data_R = data_R
        self.data_ss = data_ss


        # self.calc_moments()

        return


    # def calc_kap_bornwith(self):
    #     #kap_bornwith is kap when the agent is/was age zero.

    #     #simulation parameters
    #     sim_time = self.sim_time
    #     num_total_pop = self.num_total_pop

    #     #load main simlation result
    #     data_is_c = self.data_is_c
    #     data_is_s = ~data_is_c


    #     data_is_o = self.data_is_o
    #     data_is_y = ~data_is_o

    #     data_kap = self.data_kap

    #     data_kap_bornwith = np.full(data_kap.shape, -1.0)

    #     @nb.jit(nopython = True, parallel = True)
    #     def _calc_kap_bornwith_(data_kap_bornwith_, data_kap_, data_is_o_):

    #         for i in nb.prange(num_total_pop):

    #             t = 0
    #             data_kap_bornwith_[i, t] = data_kap_[i, t]
    #             data_kap_bornwith_[i, t+1] = data_kap_[i, t+1]

    #             for t in range(1, sim_time):

    #                 #if reborn
    #                 if data_is_o_[i, t] and not data_is_o_[i, t+1]:
    #                     data_kap_bornwith_[i, t] = data_kap_[i, t]
    #                 else:
    #                     data_kap_bornwith_[i, t] = data_kap_bornwith_[i, t-1]
            
        
    #     _calc_kap_bornwith_(data_kap_bornwith, data_kap, data_is_o)

    #     self.data_kap_bornwith = data_kap_bornwith

    #     return
    
        
    #this calculate age of S-corp. 
    def calc_age(self):

        #import data from Econ

        #simulation parameters
        sim_time = self.sim_time
        num_total_pop = self.num_total_pop

        #load main simlation result
        data_is_c = self.data_is_c
        data_is_s = ~data_is_c

        data_is_o = self.data_is_o
        data_is_y = ~data_is_o

        data_is_born = np.zeros((num_total_pop, sim_time+1), dtype = bool)
        data_is_born[:,1:] = (~data_is_o[:,1:])*(data_is_o[:,:-1])

        self.data_is_born = data_is_born
        
        s_age = np.ones(num_total_pop, dtype = int) * -1
        c_age = np.ones(num_total_pop, dtype = int) * -1

        sind_age = np.ones(num_total_pop, dtype = int) * -1
        cind_age = np.ones(num_total_pop, dtype = int) * -1
        
        o_age = np.ones(num_total_pop, dtype = int) * -1
        y_age = np.ones(num_total_pop, dtype = int) * -1

        

        ind_age = np.ones(num_total_pop, dtype = int) * -1
        

        @nb.jit(nopython = True, parallel = True)
        def _calc_age_(_data_is_, _age_):
            for i in nb.prange(num_total_pop):
                if not _data_is_[i,-1]:
                    _age_[i] = -1
                else:
                    t = 0
                    while t < sim_time:
                        if _data_is_[i, -t - 1]:
                            _age_[i] = t
                        else:
                            break
                        t = t + 1


        @nb.jit(nopython = True, parallel = True)
#        @nb.jit(nopython = True)
        def _calc_ind_age_(_data_is_born_, _age_):
            for i in nb.prange(num_total_pop):
                t = 0

                while t < sim_time:
                    if _data_is_born_[i, -t - 1]:
                        _age_[i] = t
                        break
                    else:
                        t = t+1

        @nb.jit(nopython = True, parallel = True)
        def _calc_age_adjust_born_(_data_is_, _data_is_born_, _age_):
            for i in nb.prange(num_total_pop):
                if not _data_is_[i,-1]:
                    _age_[i] = -1
                else:
                    t = 0
                    while t < sim_time:
                        if _data_is_[i, -t-1] and not _data_is_born_[i, -t-1]: #as long as they stay the same occupation and NOT age 0
                            _age_[i] = t
                            
                        elif _data_is_[i, -t-1] and _data_is_born_[i, -t-1]:
                            _age_[i] = t
                            break
                    
                        else:
                            break

                        t = t + 1
                        
                        

        _calc_age_(data_is_c, c_age)                
        _calc_age_(data_is_s, s_age)                                        
        _calc_age_(data_is_y[:,0:-1], y_age)
        _calc_age_(data_is_o[:,0:-1], o_age)
        _calc_age_adjust_born_(data_is_c, data_is_born[:,0:-1], cind_age)
        _calc_age_adjust_born_(data_is_s, data_is_born[:,0:-1], sind_age)        
        _calc_ind_age_(data_is_born[:,0:-1], ind_age)

        
        # for i in range(num_total_pop):
        #     if data_is_c[i,-1]:
        #         s_age[i] = -1
        #     else:
        #         t = 0
        #         while t < sim_time:
        #             if data_is_s[i, -t - 1]:
        #                 s_age[i] = t
        #             else:
        #                 break
        #             t = t + 1


        # for i in range(num_total_pop):
        #     if data_is_s[i,-1]:
        #         c_age[i] = -1
        #     else:
        #         t = 0
        #         while t < sim_time:
                    
        #             if data_is_c[i, -t - 1]:
        #                 c_age[i] = t
        
        #             else:
        #                 break
        #             t = t + 1
        

        self.s_age = s_age
        self.c_age = c_age
        self.sind_age = sind_age
        self.cind_age = cind_age
        self.y_age = y_age
        self.o_age = o_age
        self.ind_age = ind_age        
        
        return
        
    def calc_moments(self):

        for variable in self.__dict__ : exec(variable+'= self.'+variable, locals(), globals())

        # #load main simlation result
        # data_a = self.data_a
        # data_kap = self.data_kap
        # data_i_s = self.data_i_s
        # data_is_c = self.data_is_c
        # data_ss = self.data_ss


        #print moments

        mom0 = None
        mom1 = None
        mom2 = None
        mom3 = None
        mom4 = None
        mom5 = None
        mom6 = None
        mom7 = None
        mom8 = None
        mom9 = None
        

        if rank == 0:
            print('max of a in simulation = {}'.format(np.max(data_a)))
            print('min of a in simulation = {}'.format(np.min(data_a)))
            print('max of kap in simulation = {}'.format(np.max(data_kap)))
            print('min of kap in simulation = {}'.format(np.min(data_kap)))

            print('')
            print(f'min of agrid = {agrid[0]}')
            print(f'max of agrid = {agrid[-1]}')            
            print(f'min of kapgrid = {kapgrid[0]}')
            print(f'max of kapgrid = {kapgrid[-1]}')
            print('')

            
            t = sim_time - 1

            # data_ss
            # 0: option
            # 1: a
            # 2: kap
            # 3: an
            # 4: kapn
            # 5: eps or z (after adjusting tau_wo and tau_bo)
            # 6: cc
            # 7: cs
            # 8: cagg
            # 9: l
            # 10: n or hy
            # 11: hkap
            # 12: h
            # 13: x
            # 14: ks
            # 15: ys
            # 16: ns
            # 17: i_bracket
            # 18: taub[] or taun[]
            # 19: psib[] or psin[] + is_o*trans_retire
            # 20: is_o
            # 21: R



            is_c = np.isin(data_ss[:, 0], [0, 1])
            is_c_sale = np.isin(data_ss[:, 0], [0])
            is_c_nonsale = np.isin(data_ss[:, 0], [1])
            
            is_s = np.isin(data_ss[:, 0], [2, 3])            
            is_s_nonacq = np.isin(data_ss[:, 0], [2])
            is_s_acq = np.isin(data_ss[:, 0], [3])                        
            
            
            EIc = np.mean(is_c)
            Ea = np.mean(data_ss[:,1])
            Ekap = np.mean(data_ss[:,2])
            Ecc = np.mean(data_ss[:,6])
            Ecs = np.mean(data_ss[:,7])
            El = np.mean(data_ss[:,9])
            En = np.mean(data_ss[:,5]* data_ss[:,10] * is_c)
            
            Ex = np.mean(data_ss[:,13] * is_s)
            Eks = np.mean(data_ss[:,14] * is_s)
            Eys = np.mean(data_ss[:,15] * is_s)
            Ehkap = np.mean(data_ss[:,11] * is_s)
            Ehy = np.mean(data_ss[:,10] * is_s)
            Eh = np.mean(data_ss[:,12] * is_s)
            Ens = np.mean(data_ss[:,16] * is_s)


            Ecagg_c = np.mean((data_ss[:,6] + p*data_ss[:,7] )* is_c)
            Ecagg_s = np.mean((data_ss[:,6] + p*data_ss[:,7] )* is_s)

            wepsn_i = w*data_ss[:,5]*data_ss[:,10]*is_c
            ETn = np.mean((data_ss[:,18]*wepsn_i - data_ss[:,19])*is_c)

            # ETm = np.mean((taum*(p*data_ss[:,15] - (rs + delk)*data_ss[:,14] - w*data_ss[:,16] - data_ss[:,13]) - tran)*(1. - data_ss[:,0]) )
            
            bizinc_i = (p*data_ss[:,15] - (rs + delk)*data_ss[:,14] - w*data_ss[:,16] - data_ss[:,13])*is_s
            ETm = np.mean((data_ss[:,18]*(bizinc_i) - data_ss[:,19])*is_s)

            E_transfer = np.mean(data_ss[:,19]) #E_transfer includes E_transfer
            ETr = np.mean(trans_retire*data_ss[:,20]) # for now,E_transfer includes ETr, so we don't need include this term.
            ER = np.mean(data_ss[:,21]*is_c_sale)
            
            ETcg = taucg*ER
            pkap_implied = ER/np.mean(is_s_acq)
            kapbar_implied = np.mean(data_ss[:,2][is_c_sale])
            #kapbar_implied = np.mean(data_ss[:,2]*is_c_sale) * 10.
                          
            # yc = 1.0 #we can do this by choosing C-corp firm productivity A

            #here we impose labor market clearing. need to be clearful.
            #(I think) unless we inpose yc = 1, nc or kc or yc can't be identified from prices,
            nc = En - Ens
            
            self.nc = nc
            self.En = En
            self.Ens = Ens
        

            kc = nc*kcnc_ratio
            
            # kc = ((w/(1. - theta)/A)**(1./theta))*nc
            

            yc = A * (kc**theta)*(nc**(1.-theta))
            yc_sub = Ecc + Ex+ (grate + delk)*(kc + Eks) + g + xnb - yn

            Tc = tauc*(Ecc + p*Ecs)
            Tp = taup*(yc - w*nc - delk*kc)
            Td = taud*(yc - w*nc - (grate + delk)*kc - Tp)

            #b = (Tc + ETn + ETm + Tp + Td - g)/(rbar - grate) #old def
            b = Ea - (1. - taud)*kc - Eks
    #         netb = (grate + delk)*b ##typo
            netb = (rbar - grate)*b
            tax_rev = Tc + ETn + ETm + Td + Tp + E_transfer + ETcg

            GDP = yc + yn + p*Eys - Ex
            C = Ecc + p*Ecs
            xc = (grate + delk)*kc
            Exs = (grate + delk)*Eks

            def gini(array):
                """Calculate the Gini coefficient of a numpy array."""
                # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
                # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
                array = array.flatten() #all values are treated equally, arrays must be 1d
                if np.amin(array) < 0:
                    array -= np.amin(array) #values cannot be negative
                array += 0.0000001 #values cannot be 0
                array = np.sort(array) #values must be sorted
                index = np.arange(1,array.shape[0]+1) #index per array element
                n = array.shape[0]#number of array elements
                return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient




    #         print()
    #         print('yc = {}'.format(yc)) 
    #         print('nc = {}'.format(nc))
    #         print('kc = {}'.format(kc))
    #         print('Tc = {}'.format(Tc))
    #         print('Tp = {}'.format(Tp))
    #         print('Td = {}'.format(Td))
    #         print('b = {}'.format(b))

            print('')
            print('RESULT')
            print('Simulation Parameters')
            print('Simulation Periods = ', sim_time)
            print('Simulation Pops = ', num_total_pop)
            print('')
            print('Prices')
                          

            print('Wage (w) = {}'.format(w))
            print('S-good price (p) = {}'.format(p))
            print('Interest rate (r_c) = {}'.format(rc))

            #instead of imposing labor market clearing condition
            # mom0 = 1. - (1. - theta)*yc/(w*nc)

            cc_intermediary = pkap*np.mean(is_s_acq) - ER
            mom0 = 1. - Ecs/Eys
            # mom1 = 1. - (Ecc  + Ex+ (grate + delk)*(kc + Eks) + g + xnb - yn)/yc
            mom1 = 1. - (cc_intermediary + Ecc  + Ex+ (grate + delk)*(kc + Eks) + g + xnb - yn)/yc #modified C-goods market clearing condition
            mom2 = 1. - (tax_rev - E_transfer - netb)/g            
            print('')

            # print('1-(1-thet)*yc/(E[w*eps*n]) = {}'.format(mom0))
            print('1-E(cs)/E(ys) = {}'.format(mom0))
            print('1-(Ecc+Ex+(grate+delk)*(kc + Eks)+ g + xnb - yn)/yc = {}'.format(mom1))            
            #print('1-((1-taud)kc+E(ks)+b)/Ea = {}'.format(1. - (b + (1.- taud)*kc + Eks)/Ea))
            print('1-(tax-tran-netb)/g = {}'.format(mom2))


            print('')
            print('Important Moments')
            print('  Financial assets(Ea) = {}'.format(Ea))
            print('  Sweat assets(Ekap) = {}'.format(Ekap))
            print('  Govt. debt(b) = {}'.format(b))
            print('  S-Corp Rental Capital (ks) =  {}'.format(Eks))
            print('  Ratio of C-corp workers(EIc) = {}'.format(EIc))
            print('  GDP(yc + yn + p*Eys - Ex) = {}'.format(GDP))

            print('')
            print('C-corporation production:')
            print('  Consumption(Ecc) = {}'.format(Ecc))
            print('  Investment((grate+delk)*kc) = {}'.format(kc*(grate + delk)))
            print('  Govt purchase(g) = {}'.format(g))
            print('  Output(yc) = {}'.format(yc))
            print('  Capital(kc) = {}'.format(kc))
            print('  Eff. Worked Hours Demand(nc) = {}'.format(nc)) 

            print('')
            print('S-corporation production:')
            print('  Consumption(Ecs) = {}'.format(Ecs))
            print('  Output(Eys) = {}'.format(Eys))
            print('    Output in prices(p*Eys) = {}'.format(p*Eys))            
            print('  investment, sweat(Ex (or Ee)) = {}'.format(Ex))
            print('  Hours, production(Ehy) = {}'.format(Ehy))
            print('  Hours, sweat(Ehkap) = {}'.format(Ehkap))
            print('  Sweat equity (kap) = {}'.format(Ekap))
            print('  Eff. Worked Hours Demand(Ens) = {}'.format(Ens))             

            print('')
            print('National Income Shares (./GDP):')
            print('  C-wages (w*nc) = {}'.format((w*nc)/GDP))
            print('  Rents(rc*kc + rs*Eks) = {}'.format((rc*kc + rs*Eks)/GDP))
            print('  Sweat(p*Eys - (rs+delk)*Eks - Ex)) = {}'.format((p*Eys - (rs+delk)*Eks - Ex)/GDP))
            print('      Pure Sweat(p*Eys - (rs+delk)*Eks - w*Ens - Ex)) = {}'.format((p*Eys - (rs+delk)*Eks - w*Ens - Ex)/GDP))
            print('      S-wage(w*Ens)) = {}'.format((w*Ens)/GDP))                        
            print('  Deprec.(delk*(kc+Eks)) = {}'.format((delk*(kc+Eks))/GDP))
            print('  NonBusiness income(yn) = {}'.format(yn/GDP))
            print('    Sum = {}'.format((w*nc + rc*kc + rs*Eks+ p*Eys - (rs+delk)*Eks + delk*(kc+Eks) + yn)/GDP))

            self.puresweat = (p*Eys - (rs+delk)*Eks - w*Ens)/GDP
            

            print('')
            print('National Product Shares (./GDP):')
            print('  Consumption(C) = {}'.format(C/GDP))
            print('  physical Investments(xc) = {}'.format((xc)/GDP))
            print('  physical Investments(Exs) = {}'.format((Exs)/GDP))
            print('  Govt Consumption(g) = {}'.format(g/GDP))
            print('  Nonbusiness investment(xnb) = {}'.format((xnb)/GDP))
            print('  sweat    Investment(Ex) = {}'.format(Ex))
            print('    Sum = {}'.format((C+xc+Exs+g+xnb + Ex)/GDP))

            print('')
            print('Govt Budget:')
            print('  Public Consumption(g) = {}'.format(g))
            print('  Net borrowing(netb) = {}'.format(netb))
            print('  Transfer  = {}'.format(E_transfer))
            print('  Tax revenue(tax_rev) = {}'.format(tax_rev))

            print('')
            print('Gini Coefficients:')
            print('  Financial Assets = {}'.format(gini(data_ss[:,1])))
            print('  Sweats Assets = {}'.format(gini(data_ss[:,2])))
            print('  C-wages (wepsn) (S\' wage is set to zero)= {}'.format(gini(wepsn_i)))
            print('  C-wages (wepsn) (conditional on C)= {}'.format(gini(wepsn_i[is_c])))            
            print('  S-inc (pys - (rs +delk) - wns - x) (C\'s is set to zero )= {}'.format(gini(bizinc_i*(is_s))))
            print('  S-inc (pys - (rs +delk) - wns - x) (conditional on S)= {}'.format(gini(bizinc_i[is_s])))            
        #     print('  S-income = {}'.format(gini((p*data_ss[:,14]) * (1. - data_ss[:,0]))))
        #     print('  Total Income'.format('?'))
        #     print()


            print('')
            print('Characteristics of Owners:')
            print('  Frac of S-corp owner = {}'.format(1. - EIc))
            print('  Switched from S to C = {}'.format(np.mean((1. - data_is_c[:, t-1]) * (data_is_c[:, t]))))
            print('  Switched from C to S = {}'.format(np.mean(data_is_c[:, t-1] * (1. - data_is_c[:, t]) )))


            print('')
            print('Acquired N years ago:')

            for t in range(40):
                print(' N is ', t, ', with = {:f}'.format((np.mean(np.all(data_is_c[:,-(t+2):-1] == False, axis = 1)) - np.mean(np.all(data_is_c[:,-(t+3):-1] == False, axis = 1)) )/ np.mean(is_s))) 

            t = sim_time -1
            
            print('')
            print('Labor Market')
            print('  Labor Supply(En)   = {}'.format(En))
            print('  Labor Demand of C(nc) = {}'.format(nc))
            print('  Labor Demand of S(Ens) = {}'.format(Ens))
            print('')


            print('')
            print('Additional Moments for the Lifecycle version model')
            print('  Frac of Old         = {}'.format(np.mean(data_ss[:,20])))
            print('  Frac of Old (check) = {}'.format(np.mean(data_is_o[:,t-1])))
            #add double check
            print('  Frac of Young who was Young, Old = {}, {} '.format(np.mean( (1.-data_is_o[:,t])*(1.-data_is_o[:,t-1])),
                                                                        np.mean( (1.-data_is_o[:,t])*data_is_o[:,t-1])))
            print('  Frac of Old   who was Young, Old = {}, {} '.format(np.mean( (data_is_o[:,t])*(1.-data_is_o[:,t-1])),
                                                                        np.mean( (data_is_o[:,t])*data_is_o[:,t-1])))
            print('  Frac of Young who is  C  ,   S   = {}, {} '.format(np.mean( (1.-data_is_o[:,t])*(data_is_c[:,t])),
                                                                        np.mean( (1.-data_is_o[:,t])*(1.-data_is_c[:,t]))))
            print('  Frac of Old   who is  C  ,   S   = {}, {} '.format(np.mean( (data_is_o[:,t])*(data_is_c[:,t])),
                                                                        np.mean( (data_is_o[:,t])*(1.-data_is_c[:,t]))))
            print('  Deterioraton of kapn due to inheritance = {}'.format(np.mean(data_kap0[:,t]) - np.mean(data_kap[:,t])))
            print('  Deterioraton of kap  due to inheritance = {}'.format(np.mean(data_kap0[:,t-1]) - np.mean(data_kap[:,t-1])))

            print('  Transfer                = {}'.format(E_transfer))
            print('    Transfer (Non-retire) = {}'.format(E_transfer - ETr))
            print('    Transfer (retire)     = {}'.format(ETr))
            

            print('')
            print('Additional Moments')
            print('  E(phi p ys - x)       = {}'.format(phi*p*Eys - Ex))
            print('  E(phi p ys - x)/GDP   = {}'.format((phi*p*Eys - Ex)/GDP))
            print('  E(ks)                 = {}'.format(Eks))
            print('  E(ks)/GDP             = {}'.format(Eks/GDP))
            print('  E(nu p ys - w ns)     = {}'.format((nu*p*Eys - w*Ens)))                        
            print('  E(nu p ys - w ns)/GDP = {}'.format((nu*p*Eys - w*Ens)/GDP))



            print('')
            print('Additional Moments for Kappa-Sales')
            print('ER           = {}'.format(ER))
            print('pkap_implied = {}'.format(pkap_implied))
            print('pkap (param) = {}'.format(pkap))
            print('kapbar_implied = {}'.format(kapbar_implied))
            print('kapbar (param) = {}'.format(kapbar))

            print('')
            print('Moments for sold kappa')
            print('   Mean      = {}'.format(np.mean(data_ss[:,2][is_c_sale])))
            print('   STD       = {}'.format(np.std(data_ss[:,2][is_c_sale])))
            print('   Median    = {}'.format(np.median(data_ss[:,2][is_c_sale])))                        

            print('')
            print('Moments for active owners kappa')
            print('   Mean      = {}'.format(np.mean(data_ss[:,2][is_s])))
            print('   STD       = {}'.format(np.std(data_ss[:,2][is_s])))
            print('   Median    = {}'.format(np.median(data_ss[:,2][is_s])))                        
            
            

            print('')
            print('Additional Moments for Occupational Choice')
            print('   Option 1 (c, sale  ) = {}'.format(np.mean(is_c_sale)))
            print('   Option 2 (c, nosale) = {}'.format(np.mean(is_c_nonsale)))
            print('   Option 3 (s, nonacq) = {}'.format(np.mean(is_s_nonacq)))
            print('   Option 4 (s, acq) = {}'.format(np.mean(is_s_acq)))            



            if (self.s_age is not None) and (self.c_age is not None):
                print('')
                print('Average age/tenure of worker/passthru')

                tmp = self.s_age
                print('  All   Passthru = {}'.format(tmp[tmp != -1].mean()))
                tmp = self.c_age
                print('  All   Worker   = {}'.format(tmp[tmp != -1].mean()))
                tmp = self.s_age[self.data_is_o[:,t-1]]
                print('  Old   Passthru = {}'.format(tmp[tmp != -1].mean()))
                tmp = self.s_age[~self.data_is_o[:,t-1]]            
                print('  Young Passthru = {}'.format(tmp[tmp != -1].mean()))           
                tmp = self.c_age[self.data_is_o[:,t-1]]
                print('  Old   Worker   = {}'.format(tmp[tmp != -1].mean()))
                tmp = self.c_age[~self.data_is_o[:,t-1]]            
                print('  Young Worker   = {}'.format(tmp[tmp != -1].mean()))

            if (self.sind_age is not None) and (self.cind_age is not None):
                print('')
                print('Average age/tenure of worker/passthru within their lifetime')

                tmp = self.sind_age
                print('  All   Passthru = {}'.format(tmp[tmp != -1].mean()))
                tmp = self.cind_age
                print('  All   Worker   = {}'.format(tmp[tmp != -1].mean()))
                tmp = self.sind_age[self.data_is_o[:,t-1]]
                print('  Old   Passthru = {}'.format(tmp[tmp != -1].mean()))
                tmp = self.sind_age[~self.data_is_o[:,t-1]]            
                print('  Young Passthru = {}'.format(tmp[tmp != -1].mean()))           
                tmp = self.cind_age[self.data_is_o[:,t-1]]
                print('  Old   Worker   = {}'.format(tmp[tmp != -1].mean()))
                tmp = self.cind_age[~self.data_is_o[:,t-1]]            
                print('  Young Worker   = {}'.format(tmp[tmp != -1].mean()))           
                
            
            
            # if self.data_val_sweat is not None:
            #     EVb_sdicount = np.mean(data_val_sweat[:,-1])
            #     print('  EVb (bh un_c/u_c)              = {}'.format(EVb_sdicount))
            #     print('  EVb/GDP (bh un_c/u_c)          = {}'.format(EVb_sdicount/GDP))
                
            # if self.data_val_sweat_1gR is not None:
            #     EVb_1gR = np.mean(data_val_sweat[:,-1])
            #     print('  EVb  ((1+grate)/(1+rbar))      = {}'.format(EVb_1gR))
            #     print('  EVb/GDP ((1+grate)/(1+rbar))   = {}'.format(EVb_1gR/GDP))                            

            # mom0 = 1. - Ecs/Eys
            # mom1 = 1. - (cc_intermediary + Ecc  + Ex+ (grate + delk)*(kc + Eks) + g + xnb - yn)/yc #modified C-goods market clearing condition
            # mom2 = 1. - (tax_rev - E_transfer - netb)/g            
            
            mom3 = 0.0
            mom4 = Ens/En
            mom5 = (p*Eys - (rs+delk)*Eks - w*Ens - Ex)/GDP
            mom6 = nc
            mom7 = 1. - EIc
            mom8 = 1. - pkap_implied/pkap # ER/np.mean(is_s_acq) #(ER - pkap*np.mean(is_s_acq))/yc
            mom9 = xc/GDP                 # 1. - kapbar_implied/kapbar                          
            
        mom0 = comm.bcast(mom0) # 1. - Ecs/Eys
        mom1 = comm.bcast(mom1) # 1. - (Ecc  + Ex+ (grate + delk)*(kc + Eks) + g + xnb - yn)/yc
        mom2 = comm.bcast(mom2) # 1. - (tax_rev - tran - netb)/g
        mom3 = comm.bcast(mom3) # 0.0
        mom4 = comm.bcast(mom4) # Ens/En
        mom5 = comm.bcast(mom5) # (p*Eys - (rs+delk)*Eks - w*Ens - Ex)/GDP
        mom6 = comm.bcast(mom6) # nc
        mom7 = comm.bcast(mom7) # 1. - EIc
        mom8 = comm.bcast(mom8) # 1. - pkap_implied/pkap
        mom9 = comm.bcast(mom9) # 1. - kapbar_implied/kapbar
        

        self.moms = [mom0, mom1, mom2, mom3, mom4, mom5, mom6, mom7, mom8, mom9]

        return

    def save_moments(self, file_path = './save_data/'):

        if rank == 0:

            # for variable in self.__dict__ : exec(variable+'= self.'+variable, locals(), globals())        
            f = open(file_path  + 'moments.csv', 'w')
            data_ss = self.data_ss
            data_is_c = self.data_is_c            

            w = self.w
            p = self.p
            rs = self.rs
            pkap = self.pkap
            kapbar = self.kapbar
            
            delk = self.delk
            trans_retire = self.trans_retire
            A = self.A
            theta = self.theta
            g = self.g
            xnb = self.xnb
            yn = self.yn

            rc = self.rc            
            rbar = self.rbar
            grate = self.grate
            
            tauc = self.tauc
            taup = self.taup            
            taud = self.taud
            taucg = self.taucg
            kcnc_ratio = self.kcnc_ratio

            # data_ss
            # 0: is_c
            # 1: a
            # 2: kap
            # 3: an
            # 4: kapn
            # 5: eps or z (after adjusting tau_wo and tau_bo)
            # 6: cc
            # 7: cs
            # 8: cagg
            # 9: l
            # 10: n or hy
            # 11: hkap
            # 12: h
            # 13: x
            # 14: ks
            # 15: ys
            # 16: ns
            # 17: i_bracket
            # 18: taub[] or taun[]
            # 19: psib[] or psin[] + is_o*trans_retire
            # 20: is_o


            is_c = np.isin(data_ss[:, 0], [0, 1])
            is_c_sale = np.isin(data_ss[:, 0], [0])
            is_c_nonsale = np.isin(data_ss[:, 0], [1])
            
            is_s = np.isin(data_ss[:, 0], [2, 3])            
            is_s_nonacq = np.isin(data_ss[:, 0], [2])
            is_s_acq = np.isin(data_ss[:, 0], [3])                        
            
            
            EIc = np.mean(is_c)
            Ea = np.mean(data_ss[:,1])
            Ekap = np.mean(data_ss[:,2])
            Ecc = np.mean(data_ss[:,6])
            Ecs = np.mean(data_ss[:,7])
            El = np.mean(data_ss[:,9])
            En = np.mean(data_ss[:,5]* data_ss[:,10] * is_c)
            
            Ex = np.mean(data_ss[:,13] * is_s)
            Eks = np.mean(data_ss[:,14] * is_s)
            Eys = np.mean(data_ss[:,15] * is_s)
            Ehkap = np.mean(data_ss[:,11] * is_s)
            Ehy = np.mean(data_ss[:,10] * is_s)
            Eh = np.mean(data_ss[:,12] * is_s)
            Ens = np.mean(data_ss[:,16] * is_s)


            Ecagg_c = np.mean((data_ss[:,6] + p*data_ss[:,7] )* is_c)
            Ecagg_s = np.mean((data_ss[:,6] + p*data_ss[:,7] )* is_s)

            wepsn_i = w*data_ss[:,5]*data_ss[:,10]*is_c
            ETn = np.mean((data_ss[:,18]*wepsn_i - data_ss[:,19])*is_c)

            # ETm = np.mean((taum*(p*data_ss[:,15] - (rs + delk)*data_ss[:,14] - w*data_ss[:,16] - data_ss[:,13]) - tran)*(1. - data_ss[:,0]) )
            
            bizinc_i = (p*data_ss[:,15] - (rs + delk)*data_ss[:,14] - w*data_ss[:,16] - data_ss[:,13])*is_s
            ETm = np.mean((data_ss[:,18]*(bizinc_i) - data_ss[:,19])*is_s)

            E_transfer = np.mean(data_ss[:,19]) #E_transfer includes E_transfer
            ETr = np.mean(trans_retire*data_ss[:,20]) # for now,E_transfer includes ETr, so we don't need include this term.
            ER = np.mean(data_ss[:,21]*is_c_sale)
            
            ETcg = taucg*ER
            pkap_implied = ER/np.mean(is_s_acq)
            kapbar_implied = np.mean(data_ss[:,2][is_c_sale])
            

            # yc = 1.0 #we can do this by choosing C-corp firm productivity A

            #here we impose labor market clearing. need to be clearful.
            #(I think) unless we inpose yc = 1, nc or kc or yc can't be identified from prices,
            nc = En - Ens
            kc = nc*kcnc_ratio
            
            # kc = ((w/(1. - theta)/A)**(1./theta))*nc
            

            yc = A * (kc**theta)*(nc**(1.-theta))
            yc_sub = Ecc + Ex+ (grate + delk)*(kc + Eks) + g + xnb - yn

            Tc = tauc*(Ecc + p*Ecs)
            Tp = taup*(yc - w*nc - delk*kc)
            Td = taud*(yc - w*nc - (grate + delk)*kc - Tp)

            #b = (Tc + ETn + ETm + Tp + Td - g)/(rbar - grate) #old def
            b = Ea - (1. - taud)*kc - Eks
            #netb = (grate + delk)*b ##typo
            netb = (rbar - grate)*b
            tax_rev = Tc + ETn + ETm + Td + Tp + E_transfer + ETcg

            GDP = yc + yn + p*Eys - Ex
            C = Ecc + p*Ecs
            xc = (grate + delk)*kc
            Exs = (grate + delk)*Eks

            # instead of imposing labor market clearing condition
            # mom0 = 1. - (1. - theta)*yc/(w*nc)

            cc_intermediary = pkap*np.mean(is_s_acq) - ER
            
            mom0 = 1. - Ecs/Eys
            mom1 = 1. - (cc_intermediary + Ecc  + Ex+ (grate + delk)*(kc + Eks) + g + xnb - yn)/yc #modified C-goods market clearing condition            
            mom2 = 1. - (tax_rev - E_transfer - netb)/g
            
            self.puresweat = (p*Eys - (rs+delk)*Eks - w*Ens)/GDP


            f.write('moment, value')
            f.write('\n')            
            
            f.write('Wage (w)  ,' + str(w))
            f.write('\n')            
            f.write('S-good price (p)  ,' + str(p))
            f.write('\n')            
            f.write('Interest rate (rc)  ,' + str(rc))
            f.write('\n')            
            f.write('pkap  ,' + str(pkap))
            f.write('\n')
            f.write('kapbar  ,' + str(kapbar))
            f.write('\n')            
            

            
            # f.write('1-(1-thet)*yc/(E[w*eps*n])  ,' + str(mom0))
            f.write('1-E(cs)/E(ys)  ,' + str(mom0))
            f.write('\n')            
            f.write('1-(cc_intermediary + Ecc  + Ex+ (grate + delk)*(kc + Eks) + g + xnb - yn)/yc  ,' + str(mom1))
            f.write('\n')            
            # f.write('1-((1-taud)kc+E(ks)+b)/Ea  ,' + str(1. - (b + (1.- taud)*kc + Eks)/Ea))
            f.write('1-(tax-tran-netb)/g  ,' + str(mom2))
            f.write('\n')            


            #f.write('')
            #f.write('Important Moments')
            f.write('  Financial assets(Ea)  ,' + str(Ea))
            f.write('\n')            
            f.write('  Sweat assets(Ekap)  ,' + str(Ekap))
            f.write('\n')            
            f.write('  Govt. debt(b)  ,' + str(b))
            f.write('\n')            
            f.write('  S-Corp Rental Capital (ks)   ,' + str(Eks))
            f.write('\n')            
            f.write('  Ratio of C-corp workers(EIc)  ,' + str(EIc))
            f.write('\n')            
            f.write('  GDP(yc + yn + p*Eys - Ex)  ,' + str(GDP))
            f.write('\n')            

            # f.write('')
            # f.write('C-corporation production:')
            f.write('  Consumption(Ecc)  ,' + str(Ecc))
            f.write('\n')            
            f.write('  Investment((grate+delk)*kc)  ,' + str(kc*(grate + delk)))
            f.write('\n')            
            f.write('  Govt purchase(g)  ,' + str(g))
            f.write('\n')            
            f.write('  Output(yc)  ,' + str(yc))
            f.write('\n')            
            f.write('  Capital(kc)  ,' + str(kc))
            f.write('\n')            
            f.write('  Eff. Worked Hours Demand(nc)  ,' + str(nc))
            f.write('\n')            

            # f.write('')
            # f.write('S-corporation production:')
            f.write('  Consumption(Ecs)  ,' + str(Ecs))
            f.write('\n')            
            f.write('  Output(Eys)  ,' + str(Eys))
            f.write('\n')            
            f.write('    Output in prices(p*Eys)  ,' + str(p*Eys))
            f.write('\n')            
            f.write('  investment sweat(Ex (or Ee))  ,' + str(Ex))
            f.write('\n')            
            f.write('  Hours production(Ehy)  ,' + str(Ehy))
            f.write('\n')            
            f.write('  Hours sweat(Ehkap)  ,' + str(Ehkap))
            f.write('\n')            
            f.write('  Sweat equity (kap)  ,' + str(Ekap))
            f.write('\n')            
            f.write('  Eff. Worked Hours Demand(Ens)  ,' + str(Ens))
            f.write('\n')            

            # f.write('')
            # f.write('National Income Shares (./GDP):')
            f.write('  C-wages (w*nc)  ,' + str((w*nc)/GDP))
            f.write('\n')            
            f.write('  Rents(rc*kc + rs*Eks)  ,' + str((rc*kc + rs*Eks)/GDP))
            f.write('\n')            
            f.write('  Sweat(p*Eys - (rs+delk)*Eks - Ex))  ,' + str((p*Eys - (rs+delk)*Eks - Ex)/GDP))
            f.write('\n')            
            f.write('      Pure Sweat(p*Eys - (rs+delk)*Eks - w*Ens - Ex))  ,' + str((p*Eys - (rs+delk)*Eks - w*Ens - Ex)/GDP))
            f.write('\n')            
            f.write('      S-wage(w*Ens))  ,' + str((w*Ens)/GDP))
            f.write('\n')            
            f.write('  Deprec.(delk*(kc+Eks))  ,' + str((delk*(kc+Eks))/GDP))
            f.write('\n')            
            f.write('  NonBusiness income(yn)  ,' + str(yn/GDP))
            f.write('\n')            
            f.write('    Sum  ,' + str((w*nc + rc*kc + rs*Eks+ p*Eys - (rs+delk)*Eks + delk*(kc+Eks) + yn)/GDP))
            f.write('\n')            


            

            # f.write('')
            # f.write('National Product Shares (./GDP):')
            f.write('  Consumption(C)  ,' + str(C/GDP))
            f.write('\n')            
            f.write('  physical Investments(xc)  ,' + str((xc)/GDP))
            f.write('\n')            
            f.write('  physical Investments(Exs)  ,' + str((Exs)/GDP))
            f.write('\n')
            f.write('  Govt Consumption(g)  ,' + str(g/GDP))
            f.write('\n')            
            f.write('  Nonbusiness investment(xnb)  ,' + str((xnb)/GDP))
            f.write('\n')            
            f.write('  sweat    Investment(Ex)  ,' + str(Ex))
            f.write('\n')            
            f.write('    Sum  ,' + str((C+xc+Exs+g+xnb + Ex)/GDP))
            f.write('\n')            

            # f.write('')
            # f.write('Govt Budget:')
            f.write('  Public Consumption(g)  ,' + str(g))
            f.write('\n')            
            f.write('  Net borrowing(netb)  ,' + str(netb))
            f.write('\n')            
            f.write('  Transfer   ,' + str(E_transfer))
            f.write('\n')            
            f.write('  Tax revenue(tax_rev)  ,' + str(tax_rev))
            f.write('\n')            


            # f.write('')
            # f.write('Characteristics of Owners:')
            f.write('  Frac of S-corp owner  ,' + str(1. - EIc))
            f.write('\n')            
            # f.write('  Switched from S to C  ,' + str(np.mean((1. - data_is_c[:, t-1]) * (data_is_c[:, t]))))
            # f.write('  Switched from C to S  ,' + str(np.mean(data_is_c[:, t-1] * (1. - data_is_c[:, t]) )))


            
            # f.write('')
            # f.write('Labor Market')
            f.write('\n')            
            f.write('  Labor Supply(En)    ,' + str(En))
            f.write('\n')            
            f.write('  Labor Demand of C(nc)  ,' + str(nc))
            f.write('\n')            
            f.write('  Labor Demand of S(Ens)  ,' + str(Ens))
            f.write('\n')            
            # f.write('  Transfer                 ,' + str(E_transfer))
            f.write('    Transfer (Non-retire)  ,' + str(E_transfer - ETr))
            f.write('\n')            
            f.write('    Transfer (retire)      ,' + str(ETr))
            f.write('\n')

            f.write('Additional Moments for Kappa-Sales')
            f.write('\n')            
            f.write('ER           ,' + str(ER))
            f.write('\n')            
            f.write('pkap_implied ,' + str(pkap_implied))
            f.write('\n')            
            f.write('pkap (param) ,' + str(pkap))
            f.write('\n')            
            f.write('kapbar_implied ,' + str(kapbar_implied))
            f.write('\n')            
            f.write('kapbar (param) ,' + str(kapbar))
            


            f.close()



            f = open(file_path  + 'started_n_periods_ago.csv', 'w')

            f.write('N periods ago,')
            f.write('\n')                        

            for t in range(80):#randomly large numbers
                f.write(str(t) + ','  + str((np.mean(np.all(data_is_c[:,-(t+2):-1] == False, axis = 1)) - np.mean(np.all(data_is_c[:,-(t+3):-1] == False, axis = 1)) )/ np.mean(is_s)))
                f.write('\n')                            

            f.close()
            
            f = open(file_path  + 'income_shares.csv', 'w')
            f.write('moment, value')
            f.write('\n')            
            f.write('Business Income, ' + str(1. - yn/GDP))
            f.write('\n')                        
            f.write('Sweat labor income, ' + str((p*Eys - (rs+delk)*Eks - w*Ens - Ex)/GDP))
            f.write('\n')                        
            f.write('Non-sweat labor income, ' + str((w*nc + w*Ens)/GDP))
            f.write('\n')                        
            f.write('Non-sweat labor income C corporations, ' + str((w*nc)/GDP))
            f.write('\n')                        
            f.write('Non-sweat labor income pass through business, ' + str((w*Ens)/GDP))
            f.write('\n')                        
            f.write('Non-sweat capital income, ' + str((rc*kc + rs*Eks + delk*kc + delk*Eks)/GDP))
            f.write('\n')                        
            f.write('Non-sweat capital income rents, ' + str((rc*kc + rs*Eks)/GDP))
            f.write('\n')                        
            f.write('Non-sweat capital income depreciation, ' + str((delk*kc + delk*Eks)/GDP))
            f.write('\n')                        
            f.write('Non-business Income, ' + str(yn/GDP))
            f.write('\n')                        
            f.close()

            
            f = open(file_path  + 'product_shares.csv', 'w')
            f.write('moment, value')
            f.write('\n')            
            f.write('Private Consumption, ' + str((Ecc + p*Ecs)/GDP))
            f.write('\n')                        
            f.write('Government Consumption, ' + str((g)/GDP))
            f.write('\n')                        
            f.write('Business Investments C corporations, ' + str((xc)/GDP))
            f.write('\n')                        
            f.write('Business Investments pass through business, ' + str((Exs)/GDP))
            f.write('\n')                        
            f.write('Non-business Investments, ' + str((xnb)/GDP))
            f.write('\n')
            
            f.close()

            f = open(file_path  + 'hours_use.csv', 'w')
            f.write('moment, value')
            f.write('\n')            
            f.write('C corporations emp.,' + str(nc))
            f.write('\n')                        
            f.write('Pass Through   emp.,' + str(Ens))
            f.write('\n')                        
            f.write('Pass Through   owner,' + str(Ehy + Ehkap))
            f.write('\n')                        
            f.write('Pass Through   owner production,' + str(Ehy))
            f.write('\n')                        
            f.write('Pass Through   owner sweat,' + str( Ehkap))
            f.write('\n')
            
            f.close()

            f = open(file_path  + 'fixed_asset_div_gdp.csv', 'w')
            f.write('moment, value')
            f.write('\n')            
            
            
            f.write('fixed asset C corporations, ' + str((kc)/GDP))
            f.write('\n')                        
            f.write('fixed asset pass through business, ' + str((Eks)/GDP))
            f.write('\n')
            
            f.close()
            

            # if sweat equity was already calc'd

            if (self.data_val_sweat_dyna_sdf is not None ) and (self.data_val_sweat_dyna_fix is not None):
                
                f = open(file_path  + 'sweat_equity_div_gdp.csv', 'w')
                f.write('moment, value')
                f.write('\n')            
                f.write('discounted by beta_tilde, ' + str((np.mean(self.data_val_sweat_dyna_fix[:, self.sim_time - 1]))/GDP))
                f.write('\n')                            
                f.write('discounted by owner SDF , ' + str((np.mean(self.data_val_sweat_dyna_sdf[:, self.sim_time - 1]))/GDP))
                f.write('\n')
                
                f.close()

                

        return
        


    def calc_sweat_eq_value(self):

        """
        
        This method solve the value function

        V(a, \kappa, s) = d(a, \kappa, s) + E[m*V(a', \kappa' ,s')]

        where d = dividend from sweat equity
        m = beta_hat*u'_c/u_c

        return d, val
        
        """

        #load variables
        alpha = self.alpha
        beta = self.beta
        chi = self.chi
        delk = self.delk
        delkap = self.delkap
        eta = self.eta
        g = self.g
        grate = self.grate
        la = self.la
        la_tilde = self.la_tilde        
        mu = self.mu
        ome = self.ome
        phi = self.phi
        rho = self.rho
        tauc = self.tauc
        taud = self.taud
        taun = self.taun
        taup = self.taup
        taucg = self.taucg
        theta = self.theta

        veps = self.veps 
        vthet = self.vthet
        xnb = self.xnb
        yn = self.yn
        zeta= self.zeta

        agrid = self.agrid
        kapgrid = self.kapgrid
        epsgrid = self.epsgrid
        zgrid = self.zgrid

        prob = self.prob
        prob_st = self.prob_st
        prob_yo = self.prob_yo
        prob_sales = self.prob_sales                        

        is_to_iz = self.is_to_iz
        is_to_ieps = self.is_to_ieps

        num_a = self.num_a
        num_kap = self.num_kap
        num_eps = self.num_eps
        num_z = self.num_z
        num_s = self.num_s

        nu = self.nu
        bh = self.bh
        varrho = self.varrho

        w = self.w
        p = self.p
        rc = self.rc
        pkap = self.pkap
        kapbar = self.kapbar                

        rbar = self.rbar
        rs = self.rs

        iota = self.iota

        #load the value functions
        v_yc_an = self.v_yc_an
        v_oc_an = self.v_oc_an        
        v_ys_an = self.v_ys_an
        v_os_an = self.v_os_an        
        v_ys_kapn = self.v_ys_kapn
        v_os_kapn = self.v_os_kapn        
        vn_yc = self.vn_yc
        vn_oc = self.vn_oc        
        vn_ys = self.vn_ys
        vn_os = self.vn_os        

        #we may need to change the dimension
        bEV_yc = self.bEV_yc[0,0,:,:,:]
        bEV_oc = self.bEV_oc[0,0,:,:,:]
        bEV_ys = self.bEV_ys[0,0,:,:,:]
        bEV_os = self.bEV_os[0,0,:,:,:]

        R_y = self.R_y
        R_o = self.R_o        
        

        get_cstatic = self.generate_cstatic()
        get_sstatic = self.generate_sstatic()
        dc_util = self.generate_dc_util()




        ###obtain dividends and the stochastic discount factor###
        d = np.ones((num_a, num_kap, num_s, 2, 2)) * (-1.0e20)
    #     d_after_tax = np.ones((num_a, num_kap, num_s)) * (-2.0)

        dc_u = np.zeros((num_a, num_kap, num_s, 2, 2))
        dc_up = np.zeros((num_a, num_kap, num_s, 2, 2, num_s, 2, 2))

        
        data_an = np.zeros((num_a, num_kap, num_s, 2, 2))
        data_kapn0 = np.zeros((num_a, num_kap, num_s, 2, 2))

#        an2 = np.zeros((num_a, num_kap, num_s, 2,  num_s, 2))
#        kapn2 = np.zeros((num_a, num_kap, num_s, 2, num_s, 2))

        be_s = np.zeros((num_a, num_kap, num_s, 2, 2), dtype = bool)

        #to be parallelized but nb.prange does not work here.
        @nb.jit(nopython = True, parallel = True)
        def _pre_calc_(d, dc_u, dc_up, data_an, data_kapn0, be_s):
            # for ia , a in enumerate(agrid):
            for ia in nb.prange(num_a):
                a = agrid[ia]
                
                for ikap, kap in enumerate(kapgrid):
                    for istate in range(num_s):
                        z = zgrid[is_to_iz[istate]]
                        eps = epsgrid[is_to_ieps[istate]]

                        #if young

                        an = None
                        kapn0 = None

                        # for is_o in range(2):
                        for is_o, to_sales in [(0,0), (0,1), (1,0), (1,1)]:

                            #if young
                            if is_o == 0:

                                an_s = v_ys_an[ia, ikap, istate]
                                kapn_s = v_ys_kapn[ia, ikap, istate]

                                an_c = v_yc_an[ia, ikap, istate]
                                kapn_c = la * kap

                                #need to check feasibility a - pkap >= 0.
                                an_acq = fem2d_peval(a - pkap, kap + kapbar, agrid, kapgrid, v_ys_an[:, :, istate])
                                kapn_acq = fem2d_peval(a - pkap, kap + kapbar, agrid, kapgrid, v_ys_kapn[:, :, istate])

                                an_sale = femeval(a + (1. - taucg)*R_y[ia, ikap, istate], agrid, v_yc_an[:, 0, istate])
                                kapn_sale = 0.
                                
                                val_s = (get_sstatic(np.array([a, an_s, kap, kapn_s, z, is_o]))[0]  +  fem2d_peval(an_s, kapn_s, agrid, kapgrid, bEV_ys[:,:,istate])) **(1./(1.- mu))
                                val_c = (get_cstatic(np.array([a, an_c, eps, is_o]))[0]  +  fem2d_peval(an_c, kapn_c, agrid, kapgrid, bEV_yc[:,:,istate])) **(1./(1.- mu))
                                val_acq = -np.inf
                                if a - pkap >= 0.0:
                                    val_acq = (get_sstatic(np.array([a - pkap, an_acq, kap + kapbar, kapn_acq, z, is_o]))[0]  +  fem2d_peval(an_acq, kapn_acq, agrid, kapgrid, bEV_ys[:,:,istate])) **(1./(1.- mu))
                                    
                                val_sale = (get_cstatic(np.array([a + (1. - taucg)*R_y[ia, ikap, istate], an_sale, eps, is_o]))[0]  +  fem2d_peval(an_sale, kapn_sale, agrid, kapgrid, bEV_yc[:,:,istate])) **(1./(1.- mu))

                                # it should be the case that |val_acq - val_sale| < small


                            #else if old
                            elif is_o == 1:

                                an_s = v_os_an[ia, ikap, istate]
                                kapn_s = v_os_kapn[ia, ikap, istate]

                                an_c = v_oc_an[ia, ikap, istate]
                                kapn_c = la * kap

                                #need to check feasibility a - pkap >= 0.
                                an_acq = fem2d_peval(a - pkap, kap + kapbar, agrid, kapgrid, v_os_an[:, :, istate])
                                kapn_acq = fem2d_peval(a - pkap, kap + kapbar, agrid, kapgrid, v_os_kapn[:, :, istate])

                                an_sale = femeval(a + (1. - taucg)*R_o[ia, ikap, istate], agrid, v_oc_an[:, 0, istate])
                                kapn_sale = 0.
                                
                                val_s = (get_sstatic(np.array([a, an_s, kap, kapn_s, z, is_o]))[0]  +  fem2d_peval(an_s, kapn_s, agrid, kapgrid, bEV_os[:,:,istate])) **(1./(1.- mu))
                                val_c = (get_cstatic(np.array([a, an_c, eps, is_o]))[0]  +  fem2d_peval(an_c, kapn_c, agrid, kapgrid, bEV_oc[:,:,istate])) **(1./(1.- mu))
                                val_acq = -np.inf
                                if a - pkap >= 0.0:
                                    val_acq = (get_sstatic(np.array([a - pkap, an_acq, kap + kapbar, kapn_acq, z, is_o]))[0]  +  fem2d_peval(an_acq, kapn_acq, agrid, kapgrid, bEV_os[:,:,istate])) **(1./(1.- mu))
                                    
                                val_sale = (get_cstatic(np.array([a + (1. - taucg)*R_o[ia, ikap, istate], an_sale, eps, is_o]))[0]  +  fem2d_peval(an_sale, kapn_sale, agrid, kapgrid, bEV_oc[:,:,istate])) **(1./(1.- mu))

                            else:
                                print('error')


                            # print('val_s = ', val_s)
                            # print('val_sale = ', val_sale)                            
                            # decide the option at the first date
                            option = np.argmax(np.array([val_c, val_s, val_acq])) + 1

                            if option == 2 and to_sales:
                                option = 0

                                
                            is_s = None
                            # set an and kapn0
                            if option == 0:
                                an = an_sale
                                kapn0 = kapn_sale
                                is_s = False
                            elif option == 1:
                                an = an_c
                                kapn0 = kapn_c
                                is_s = False                                
                            elif option == 2:
                                an = an_s
                                kapn0 = kapn_s
                                is_s = True                                
                            elif option == 3:
                                an = an_acq
                                kapn0 = kapn_acq
                                is_s = True                                
                            else:
                                print('error: option must be either 0, 1, 2, 3.')

                            # if c
                            if not is_s:
                                be_s[ia, ikap, istate, is_o, to_sales] = False

                                if option == 0:
                                    if is_o:
                                        u, cc, cs, cagg, l ,n, i, taunn, psinn = get_cstatic([a + (1. - taucg)*R_o[ia, ikap, istate], an, eps, is_o])
                                        
                                    else:
                                        u, cc, cs, cagg, l ,n, i, taunn, psinn = get_cstatic([a + (1. - taucg)*R_y[ia, ikap, istate], an, eps, is_o])
                                        
                                elif option == 1:
                                    u, cc, cs, cagg, l ,n, i, taunn, psinn = get_cstatic([a, an, eps, is_o])
                                else:
                                    print('error: option should be 0 or 1.')
                                
                                tmp = dc_util(cagg, l)
                                dc_u[ia, ikap, istate, is_o, to_sales] = tmp

                                if np.isnan(tmp):
                                    print('error: du_c is nan, C. is_o = ', is_o, ' to_sales = ', to_sales, ', option = ', option, ', val_sale = ', val_sale, ', val_s = ', val_s)

                                if option == 0:
                                    if is_o:
                                        d[ia, ikap, istate, is_o, to_sales] = (1. - taucg)*R_o[ia, ikap, istate]
                                    else:
                                        d[ia, ikap, istate, is_o, to_sales] = (1. - taucg)*R_y[ia, ikap, istate]
                                        
                                elif option == 1:
                                    d[ia, ikap, istate, is_o, to_sales] = 0.
                                else:
                                    print('error: option should be 0 or 1.')

                            else:
                                be_s[ia, ikap, istate, is_o, to_sales] = True

                                if option == 2:
                                    u, cc, cs, cagg, l, hy, hkap, h ,x, ks, ys, ns, ib, taubb, psibb = get_sstatic([a, an, kap, kapn0, z, is_o])
                                    
                                elif option == 3:
                                    u, cc, cs, cagg, l, hy, hkap, h ,x, ks, ys, ns, ib, taubb, psibb = get_sstatic([a-pkap, an, kap+kapbar, kapn0, z, is_o])
                                    if a-pkap < 0.0:
                                        print('error: a-pkap < 0.')
                                else:
                                    print('error: option should be 2 or 3.')
                                    

                                tmp = dc_util(cagg, l)
                                dc_u[ia, ikap, istate, is_o, to_sales] = tmp

                                if np.isnan(tmp):
                                    print('error: du_c is nan, S. is_o = ', is_o, ' to_sales = ', to_sales, ', option = ', option, ', val_sale = ', val_sale, ', val_s = ', val_s)                                    
                            
                                                                

                                if option == 2:
                                    d[ia, ikap, istate, is_o, to_sales] = phi * p * ys - x
                                    
                                elif option == 3:
                                    d[ia, ikap, istate, is_o, to_sales] = phi * p * ys - x - pkap
                                    
                                else:
                                      print('error: option should be 2 or 3.')
                                    

                            data_an[ia, ikap, istate, is_o, to_sales] = an
                            data_kapn0[ia, ikap, istate, is_o, to_sales] = kapn0

                            for istate_n in range(num_s):
                                
                                
                                anp = None
                                kapnp0 = None

                                zp = zgrid[is_to_iz[istate_n]]
                                epsp = epsgrid[is_to_ieps[istate_n]]
                                

                                for is_o_n, to_sales_n in [(0,0), (0,1), (1,0), (1,1)]:

                                    #if young
                                    if is_o_n == 0:
                                        
                                        kapn = kapn0
                                        #if kappa is succeeded from a S guy, kapn must be depreciate
                                        if is_s and is_o: #and not is_o_n 
                                            kapn = la_tilde*kapn0

                                        anp_s = fem2d_peval(an, kapn, agrid, kapgrid, v_ys_an[:, :, istate_n])
                                        kapnp_s = fem2d_peval(an, kapn, agrid, kapgrid, v_ys_kapn[:, :, istate_n])

                                        anp_c = fem2d_peval(an, kapn, agrid, kapgrid, v_yc_an[:, :, istate_n])
                                        kapnp_c = la*kapn
                                        
                                        #need to check feasibility a - pkap >= 0.
                                        anp_acq = fem2d_peval(an - pkap, kapn + kapbar, agrid, kapgrid, v_ys_an[:, :, istate_n])
                                        kapnp_acq = fem2d_peval(an - pkap, kapn + kapbar, agrid, kapgrid, v_ys_kapn[:, :, istate_n])

                                        Rp = fem2d_peval(an, kapn, agrid, kapgrid, R_y[:, :, istate_n])
                                        anp_sale = femeval(an + (1. - taucg)*Rp, agrid, v_yc_an[:, 0, istate_n])
                                        kapnp_sale = 0.
                                        
                                        val_n_s = (get_sstatic([an, anp_s, kapn, kapnp_s, zp, is_o_n])[0]  +  fem2d_peval(anp_s, kapnp_s, agrid, kapgrid, bEV_ys[:,:,istate_n])) **(1./(1.- mu))
                                        val_n_c = (get_cstatic([an, anp_c, epsp, is_o_n])[0]  +  fem2d_peval(anp_c, kapnp_c, agrid, kapgrid, bEV_yc[:,:,istate_n])) **(1./(1.- mu))
                                        val_n_acq = -np.inf
                                        if an - pkap >= 0.0:
                                            val_n_acq = (get_sstatic([an - pkap, anp_acq, kapn + kapbar, kapnp_acq, zp, is_o_n])[0]  +  fem2d_peval(anp_acq, kapnp_acq, agrid, kapgrid, bEV_ys[:,:,istate_n])) **(1./(1.- mu))
                                            
                                        val_n_sale = (get_cstatic([an + (1. - taucg)*Rp, anp_sale, epsp, is_o_n])[0]  +  fem2d_peval(anp_sale, kapnp_sale, agrid, kapgrid, bEV_yc[:,:,istate_n])) **(1./(1.- mu))


                                    #else if old
                                    elif is_o_n == 1:
                                        
                                        anp_s = fem2d_peval(an, kapn0, agrid, kapgrid, v_os_an[:, :, istate_n])
                                        kapnp_s = fem2d_peval(an, kapn0, agrid, kapgrid, v_os_kapn[:, :, istate_n])

                                        anp_c = fem2d_peval(an, kapn, agrid, kapgrid, v_oc_an[:, :, istate_n])
                                        kapnp_c = la*kapn

                                        anp_acq = fem2d_peval(an - pkap, kapn + kapbar, agrid, kapgrid, v_os_an[:, :, istate_n])
                                        kapnp_acq = fem2d_peval(an - pkap, kapn + kapbar, agrid, kapgrid, v_os_kapn[:, :, istate_n])

                                        Rp = fem2d_peval(an, kapn, agrid, kapgrid, R_o[:, :, istate_n])
                                        anp_sale = femeval(an + (1. - taucg)*Rp, agrid, v_oc_an[:, 0, istate_n])
                                        kapnp_sale = 0.

                                        val_n_s = (get_sstatic([an, anp_s, kapn, kapnp_s, zp, is_o_n])[0]  +  fem2d_peval(anp_s, kapnp_s, agrid, kapgrid, bEV_os[:,:,istate_n])) **(1./(1.- mu))
                                        val_n_c = (get_cstatic([an, anp_c, epsp, is_o_n])[0]  +  fem2d_peval(anp_c, kapnp_c, agrid, kapgrid, bEV_oc[:,:,istate_n])) **(1./(1.- mu))
                                        val_n_acq = -np.inf
                                        if an - pkap >= 0.0:
                                            val_n_acq = (get_sstatic([an - pkap, anp_acq, kapn, kapnp_acq, zp, is_o_n])[0]  +  fem2d_peval(anp_acq, kapnp_acq, agrid, kapgrid, bEV_os[:,:,istate_n])) **(1./(1.- mu))
                                            
                                        val_n_sale = (get_cstatic([an + (1. - taucg)*Rp, anp_sale, epsp, is_o_n])[0]  +  fem2d_peval(anp_sale, kapnp_sale, agrid, kapgrid, bEV_oc[:,:,istate_n])) **(1./(1.- mu))
                                        

                                    else:
                                        print('error')

                                    anp = None
                                    kapnp0 = None
                                    is_s_n = None

                                    # decide the option at the first date
                                    option_n = np.argmax(np.array([val_n_c, val_n_s, val_n_acq])) + 1
                                    
                                    if option_n == 2 and to_sales_n:

                                        option_n = 0
                                        
                                        
                                    is_s_n = None
                                    # set an and kapn0
                                    if option_n == 0:
                                        anp = anp_sale
                                        kapnp0 = kapnp_sale
                                        is_s_n = False
                                        
                                    elif option_n == 1:
                                        anp = anp_c
                                        kapnp0 = kapnp_c
                                        is_s_n = False
                                        
                                    elif option_n == 2:
                                        anp = anp_s
                                        kapnp0 = kapnp_s
                                        is_s_n = True
                                        
                                    elif option_n == 3:
                                        anp = anp_acq
                                        kapnp0 = kapnp_acq
                                        is_s_n = True
                                        
                                    else:
                                        print('error: option must be either 0, 1, 2, 3.')
                                                

                                    # if C
                                    if not is_s_n:
                                        if option_n == 0:
                                            u, cc, cs, cagg, l ,n, i, taunn, psinn = get_cstatic([an + (1. - taucg)*Rp , anp, epsp, is_o_n])
                                        elif option_n == 1:
                                            u, cc, cs, cagg, l ,n, i, taunn, psinn = get_cstatic([an, anp, epsp, is_o_n])
                                        else:
                                            print('error.')

                                    # if S
                                    else:
                                        if option_n == 2:
                                            u, cc, cs, cagg, l, hy, hkap, h ,x, ks, ys, ns, ib, taubb, psibb = get_sstatic([an, anp, kapn, kapnp0, zp, is_o_n])
                                        elif option_n == 3:
                                            u, cc, cs, cagg, l, hy, hkap, h ,x, ks, ys, ns, ib, taubb, psibb = get_sstatic([an - pkap, anp, kapn + kapbar, kapnp0, zp, is_o_n])
                                        else:
                                            print('error.')
                                            
                                     
                                    tmp = dc_util(cagg, l)
                                    dc_up[ia, ikap, istate, is_o, to_sales, istate_n, is_o_n, to_sales_n] = tmp

                                    if np.isnan(tmp):
                                        print('error: du_c is nan. is_o = ', is_o, ', to_sales = ', to_sales, ' is_o_n = ', is_o_n, ' to_sales_n = ', to_sales_n, ', option = ', option,  ', option_n = ', option_n, ', val_n_sales = ', val_n_sale, ', val_n_s = ', val_n_s, '.')




            ###end obtain dividends and the stochastic discount factor###


        print('calculating d and dc,...')
        _pre_calc_(d, dc_u, dc_up, data_an, data_kapn0, be_s)
        print('done.')

        
        @nb.jit(nopython = True, parallel = True)
        def _inner_get_sweat_equity_value_pp_(_val_,#output
                                              _d_, _an_, _kapn0_,  _dc_u_, _dc_up_, _be_s_, is_dynasty = True, discount = None):
            #valb = np.full((num_a, num_kap, num_s, 2), -1000.)

            val_tmp = _val_.copy()
            bEval = np.zeros(_val_.shape)

            tol = 1.0e-8
            dist = 1.0e100
            maxit = 500
            it = 0

            while it < maxit and dist > tol:
                it = it+1
                bEval[:] = 0.0 #initialize


                ###for-s###
                for ia in nb.prange(num_a):
                    a = agrid[ia]
                    for ikap in range(num_kap):
                        kap = kapgrid[ikap]
                        for istate in range(num_s):
                            for is_o, to_sales in [(0,0), (0,1), (1,0), (1,1)]:
                                dc_u = _dc_u_[ia, ikap, istate, is_o, to_sales]

                                for istate_n in range(num_s):
                                    for is_o_n, to_sales_n in [(0,0), (0,1), (1,0), (1,1)]:


                                        an = _an_[ia, ikap, istate, is_o, to_sales]
                                        kapn = _kapn0_[ia, ikap, istate, is_o, to_sales]

                                        #if kappa_n is succeeded from old S guy, kapn must be depreciated.
                                        #this adjustment is necessary to calc val_p

                                        if is_o and _be_s_[ia, ikap, istate, is_o, to_sales] and not is_o_n:
                                            kapn = la_tilde * kapn
                                        

                                        dc_un = _dc_up_[ia, ikap, istate, is_o, to_sales, istate_n, is_o_n, to_sales_n]
                                        val_p = fem2d_peval(an, kapn, agrid, kapgrid, _val_[:, :, istate_n, is_o_n, to_sales_n])


                                        if discount is None:
                                            #if o -> y, s is drawn from unconditional dist
                                            #else, use the transition matrix
                                            if is_o and not is_o_n:
                                                if is_dynasty:
                                                    bEval[ia, ikap, istate, is_o, to_sales] += bh*iota*prob_sales[to_sales_n]*prob_yo[is_o, is_o_n]*prob_st[istate_n] * dc_un * val_p / dc_u
                                                else:
                                                    pass #if the evaluation is not dynastic, there is no future value
                                                
                                            else:
                                                bEval[ia, ikap, istate, is_o, to_sales] += bh*prob_sales[to_sales_n]*prob_yo[is_o, is_o_n]*prob[istate, istate_n] * dc_un * val_p / dc_u
                                        else: #use a given discount factor

                                            if is_o and not is_o_n:
                                                if is_dynasty:
                                                    bEval[ia, ikap, istate, is_o, to_sales] += discount*iota*prob_sales[to_sales_n]*prob_yo[is_o, is_o_n]*prob_st[istate_n] *val_p 
                                                else:
                                                    pass #if the evaluation is not dynastic, there is no future value
                                                
                                            else:
                                                bEval[ia, ikap, istate, is_o, to_sales] += discount*prob_sales[to_sales_n]*prob_yo[is_o, is_o_n]*prob[istate, istate_n] * val_p
                                            


                ###end for-s###


                #update the results
                val_tmp[:] = _val_[:]

                _val_[:] = _d_ + bEval

                dist = np.max(np.abs(_val_ - val_tmp))



            #after calc evaluation
            if it >= maxit:
                print('iteration terminated at maxit')

            ###debug code###

            print('converged at it = ', it)
            print('dist = ', dist)

            ###end debug code###                
                
                

        
        #main loop to calculate values
        val_dyna_sdf = d / (rbar - grate) #initial value
        _inner_get_sweat_equity_value_pp_(val_dyna_sdf, d, data_an, data_kapn0, dc_u, dc_up, be_s, True)

        val_dyna_fix = d / (rbar - grate) #initial value
        _inner_get_sweat_equity_value_pp_(val_dyna_fix, d, data_an, data_kapn0, dc_u, dc_up, be_s, True, (1. + grate)/(1. + rs))
        

        # val_life = d / (rbar - grate)
        # _inner_get_sweat_equity_value_pp_(val_life, d, data_an, data_kapn0, dc_u, dc_up, be_s, False)        


        self.sweat_div = d
        self.sweat_val_dyna_sdf = val_dyna_sdf
        self.sweat_val_dyna_fix = val_dyna_fix        
        

    def simulate_other_vars(self):

        
        for variable in self.__dict__ : exec(variable+'= self.'+variable, locals(), globals())

        data_a = self.data_a
        data_kap = self.data_kap
        data_kap0 = self.data_kap0
        data_i_s = self.data_i_s
        data_is_c = self.data_is_c
        data_is_o = self.data_is_o
        data_option = self.data_option
        data_sales_shock = self.data_sales_shock
        data_R = self.data_R        
        

        #simulation parameters
        sim_time = self.sim_time
        num_total_pop = self.num_total_pop

        get_cstatic = self.generate_cstatic()
        get_sstatic = self.generate_sstatic()


        @nb.jit(nopython = True, parallel = True)
        def calc_all(data_a_, data_kap_, data_kap0_, data_i_s_, data_is_c_, data_is_o_, data_option_, data_R_,
                     data_atilde_, data_kaptilde_, data_u_, data_cc_, data_cs_, data_cagg_, data_l_, data_n_, data_hy_, data_hkap_, data_h_, data_x_, data_ks_, data_ys_, data_ns_, data_i_tax_bracket_):

            for i in nb.prange(num_total_pop):
                for t in range(1, sim_time):


                    option = data_option_[i,t]
                    istate = data_i_s_[i, t]
                    eps = epsgrid[is_to_ieps[istate]]
                    z = zgrid[is_to_iz[istate]]

                    u = np.nan
                    cc = np.nan
                    cs = np.nan
                    cagg = np.nan
                    l = np.nan
                    n = np.nan
                    
                    hy = np.nan
                    hkap = np.nan
                    h = np.nan
                    ks = np.nan
                    ys = np.nan
                    ns = np.nan

                    a = data_a_[i, t-1]
                    kap = data_kap_[i, t-1] #this should be kap, not kap0

                    an = data_a_[i, t]
                    kapn = data_kap0_[i, t] #this should be kap0

                    is_c = data_is_c_[i, t]
                    is_o = data_is_o_[i, t]

                    atilde = np.nan
                    kaptilde = np.nan

                    R = data_R_[i,t]
                    
                    if option == 0: #C and Sell all the kappa

                        atilde = a + (1.-taucg)*R
                        kaptilde = 0.

                        u, cc, cs, cagg, l , n, i_bra, tau, psi = get_cstatic([atilde, an, eps, is_o])

                    elif option == 1: #C and keep kappa

                        atilde = a
                        kaptilde = kap

                        u, cc, cs, cagg, l , n, i_bra, tau, psi = get_cstatic([atilde, an, eps, is_o])
                        
                    elif option == 2: #S and no purchase of p
                        atilde = a
                        kaptilde = kap

                        u, cc, cs, cagg, l, hy, hkap, h, x, ks, ys, ns, i_bra, tau, psi = get_sstatic([atilde, an, kap, kapn, z, is_o])

                    elif option == 3: #S and acquisition
                        atilde = a - pkap
                        kaptilde = kap + kapbar

                        u, cc, cs, cagg, l, hy, hkap, h, x, ks, ys, ns, i_bra, tau, psi = get_sstatic([atilde, an, kap, kapn, z, is_o])

                    else:
                        print('error: option value is ', option)

                        

                    # data_ss
                    # 0: is_c
                    # 1: a
                    # 2: kap
                    # 3: an
                    # 4: kapn
                    # 5: eps or z
                    # 6: cc
                    # 7: cs
                    # 8: cagg
                    # 9: l
                    # 10: n or hy
                    # 11: hkap
                    # 12: h
                    # 13: x
                    # 14: ks
                    # 15: ys
                    # 16: ns
                    # 17: i_bracket
                    # 18: taub[] or taun[]
                    # 19: psib[] or psin[] + is_o*trans_retire
                    # 20: is_old
            

                    data_atilde_[i, t-1] = atilde
                    data_kaptilde_[i, t-1] = kaptilde

                    data_u_[i, t] = u
                    data_cc_[i, t] = cc
                    data_cs_[i, t] = cs
                    data_cagg_[i, t] = cagg
                    data_l_[i, t] = l
                    data_n_[i, t] = n
                    data_hy_[i, t] = hy
                    data_hkap_[i, t] = hkap
                    data_h_[i, t] = h                 
                    data_x_[i, t] = x
                    data_ks_[i, t] = ks
                    data_ys_[i, t] = ys
                    data_ns_[i, t] = ns
                    data_i_tax_bracket_[i, t] = i_bra

        data_atilde = np.zeros(data_a.shape)
        data_kaptilde = np.zeros(data_a.shape)        
        data_u = np.zeros(data_a.shape)
        data_cc = np.zeros(data_a.shape)
        data_cs = np.zeros(data_a.shape)
        data_cagg = np.zeros(data_a.shape)
        data_l = np.zeros(data_a.shape)
        data_n = np.zeros(data_a.shape)
        data_hy = np.zeros(data_a.shape)
        data_hkap = np.zeros(data_a.shape)
        data_h = np.zeros(data_a.shape)        
        data_x = np.zeros(data_a.shape)
        data_ks = np.zeros(data_a.shape)
        data_ys = np.zeros(data_a.shape)
        data_ns = np.zeros(data_a.shape)
        data_i_tax_bracket = np.zeros(data_a.shape)

        #note that this does not store some impolied values,,,, say div or value of sweat equity
        calc_all(data_a, data_kap, data_kap0, data_i_s, data_is_c, data_is_o, data_option, data_R,##input
                 data_atilde, data_kaptilde, data_u, data_cc, data_cs, data_cagg, data_l, data_n, data_hy, data_hkap, data_h, data_x, data_ks, data_ys, data_ns, data_i_tax_bracket ##output
            )


        #value of dividends are calc'd here
        @nb.jit(nopython = True, parallel = True)        
        def calc_val_seq( _data_val_, #output;
                         _data_a_, _data_kap_,  _data_i_s_, _data_is_o_, _data_sales_shock_, _data_sweat_val_):

            for i in nb.prange(num_total_pop):
                for t in range(1, sim_time):
                    
                    istate = _data_i_s_[i,t]
                    is_o = int(_data_is_o_[i,t])
                    is_sold = int(_data_sales_shock_[i,t])
#                    eps = epsgrid[is_to_ieps[istate]]
#                    z = zgrid[is_to_iz[istate]]
                                    
                    a = _data_a_[i, t-1]
                    kap = _data_kap_[i, t-1] #this is not kap0

                    #these data record beginning of the period info of dividends and their discounted values
                    #that is, all the values are taken after they observe the shocks.

                    _data_val_[i, t] = fem2d_peval(a, kap, agrid, kapgrid, _data_sweat_val_[:,:, istate, is_o, is_sold])


        data_div_sweat = np.zeros(data_a.shape)
        data_val_sweat_dyna_sdf = np.zeros(data_a.shape)
        data_val_sweat_dyna_fix = np.zeros(data_a.shape)

        calc_val_seq(data_div_sweat, data_a, data_kap,  data_i_s, data_is_o, data_sales_shock, self.sweat_div)        
        calc_val_seq(data_val_sweat_dyna_sdf, data_a, data_kap,  data_i_s, data_is_o, data_sales_shock,self.sweat_val_dyna_sdf)
        calc_val_seq(data_val_sweat_dyna_fix, data_a, data_kap,  data_i_s, data_is_o, data_sales_shock,self.sweat_val_dyna_fix)

        self.data_atilde = data_atilde
        self.data_kaptilde = data_kaptilde
        self.data_u = data_u
        self.data_cc = data_cc
        self.data_cs = data_cs
        self.data_cagg = data_cagg
        self.data_l = data_l
        self.data_n = data_n
        self.data_hy = data_hy
        self.data_hkap = data_hkap
        self.data_h = data_h
        self.data_x = data_x
        self.data_ks = data_ks
        self.data_ys = data_ys
        self.data_ns = data_ns
        self.data_i_tax_bracket = data_i_tax_bracket
        
        self.data_div_sweat = data_div_sweat
        self.data_val_sweat_dyna_sdf = data_val_sweat_dyna_sdf
        self.data_val_sweat_dyna_fix = data_val_sweat_dyna_fix        
        
        

        return

        
    def save_result(self, dir_path_save = './save_data/'):
        if rank == 0:
            print('Saving results under ', dir_path_save, '...')

            
            np.save(dir_path_save + 'agrid', self.agrid)
            np.save(dir_path_save + 'kapgrid', self.kapgrid)
            np.save(dir_path_save + 'zgrid', self.zgrid)
            np.save(dir_path_save + 'epsgrid', self.epsgrid)

            np.save(dir_path_save + 'prob', self.prob)
            np.save(dir_path_save + 'is_to_iz', self.is_to_iz)
            np.save(dir_path_save + 'is_to_ieps', self.is_to_ieps)                        

            np.save(dir_path_save + 'taub', self.taub)
            np.save(dir_path_save + 'psib', self.psib)
            np.save(dir_path_save + 'bbracket', self.bbracket)                                                             

            np.save(dir_path_save + 'taun', self.taun)
            np.save(dir_path_save + 'psin', self.psin)
            np.save(dir_path_save + 'nbracket', self.nbracket)                                                             

            np.save(dir_path_save + 'data_option', self.data_option[:, -100:])                        
            np.save(dir_path_save + 'data_is_o', self.data_is_o[:, -100-1:])            
            np.save(dir_path_save + 'data_a', self.data_a[:, -100:])
            np.save(dir_path_save + 'data_atilde', self.data_atilde[:, -100:])            
            np.save(dir_path_save + 'data_kap', self.data_kap[:, -100:])
            np.save(dir_path_save + 'data_kaptilde', self.data_kaptilde[:, -100:])            
            np.save(dir_path_save + 'data_kap0', self.data_kap0[:, -100:])            
            np.save(dir_path_save + 'data_i_s', self.data_i_s[:, -100:])
            np.save(dir_path_save + 'data_is_c', self.data_is_c[:, -100:])
            np.save(dir_path_save + 'data_u', self.data_u[:, -100:])
            np.save(dir_path_save + 'data_cc', self.data_cc[:, -100:])
            np.save(dir_path_save + 'data_cs', self.data_cs[:, -100:])
            np.save(dir_path_save + 'data_cagg', self.data_cagg[:, -100:])
            np.save(dir_path_save + 'data_l', self.data_l[:, -100:])
            np.save(dir_path_save + 'data_n', self.data_n[:, -100:])
            np.save(dir_path_save + 'data_hy', self.data_hy[:, -100:])
            np.save(dir_path_save + 'data_hkap', self.data_hkap[:, -100:])
            np.save(dir_path_save + 'data_h', self.data_h[:, -100:])            
            np.save(dir_path_save + 'data_x', self.data_x[:, -100:])
            np.save(dir_path_save + 'data_ks', self.data_ks[:, -100:])
            np.save(dir_path_save + 'data_ys', self.data_ys[:, -100:])
            np.save(dir_path_save + 'data_ns', self.data_ns[:, -100:])
            np.save(dir_path_save + 'data_i_tax_bracket', self.data_i_tax_bracket[:, -100:])
            np.save(dir_path_save + 'data_R', self.data_R[:, -100:])            
            
            np.save(dir_path_save + 'v_yc_an', self.v_yc_an)
            np.save(dir_path_save + 'v_oc_an', self.v_oc_an)            
            np.save(dir_path_save + 'v_ys_an', self.v_ys_an)
            np.save(dir_path_save + 'v_os_an', self.v_os_an)
            np.save(dir_path_save + 'v_ys_kapn', self.v_ys_kapn)
            np.save(dir_path_save + 'v_os_kapn', self.v_os_kapn)
            np.save(dir_path_save + 'vn_yc', self.vn_yc)
            np.save(dir_path_save + 'vn_oc', self.vn_oc)            
            np.save(dir_path_save + 'vn_ys', self.vn_ys)
            np.save(dir_path_save + 'vn_os', self.vn_os)            

            np.save(dir_path_save + 'sweat_div', self.sweat_div)
            np.save(dir_path_save + 'sweat_val_dyna', self.sweat_val_dyna)
            np.save(dir_path_save + 'sweat_val_life', self.sweat_val_life)

            np.save(dir_path_save + 'data_div_sweat', self.data_div_sweat[:, -100:])
            np.save(dir_path_save + 'data_val_sweat_dyna', self.data_val_sweat_dyna[:, -100:])
            np.save(dir_path_save + 'data_val_sweat_life', self.data_val_sweat_life[:, -100:])

            np.save(dir_path_save + 'sind_age', self.sind_age)
            np.save(dir_path_save + 'cind_age', self.cind_age)
            np.save(dir_path_save + 's_age', self.s_age)
            np.save(dir_path_save + 'c_age', self.c_age)
            np.save(dir_path_save + 'y_age', self.y_age)
            np.save(dir_path_save + 'o_age', self.o_age)
            np.save(dir_path_save + 'ind_age', self.ind_age)

    def save_result_csv(self, dir_path_save = './save_data/'):
        if rank == 0:
            print('Saving results under ', dir_path_save, '...')

            
            np.savetxt(dir_path_save + 'agrid.csv', self.agrid)
            np.savetxt(dir_path_save + 'kapgrid.csv', self.kapgrid)
            np.savetxt(dir_path_save + 'zgrid.csv', self.zgrid)
            np.savetxt(dir_path_save + 'epsgrid.csv', self.epsgrid)

            np.savetxt(dir_path_save + 'prob.csv', self.prob)
            np.savetxt(dir_path_save + 'is_to_iz.csv', self.is_to_iz)
            np.savetxt(dir_path_save + 'is_to_ieps.csv', self.is_to_ieps)                        

            np.savetxt(dir_path_save + 'taub.csv', self.taub)
            np.savetxt(dir_path_save + 'psib.csv', self.psib)
            np.savetxt(dir_path_save + 'bbracket.csv', self.bbracket)                                                             

            np.savetxt(dir_path_save + 'taun.csv', self.taun)
            np.savetxt(dir_path_save + 'psin.csv', self.psin)
            np.savetxt(dir_path_save + 'nbracket.csv', self.nbracket)                                                             

            np.savetxt(dir_path_save + 'data_option', self.data_option[:, -100:])
            np.savetxt(dir_path_save + 'data_option.csv', self.data_option[:, -100:])                        
            np.savetxt(dir_path_save + 'data_is_o.csv', self.data_is_o[:, -100-1:])            
            np.savetxt(dir_path_save + 'data_a.csv', self.data_a[:, -100:])
            np.savetxt(dir_path_save + 'data_kap.csv', self.data_kap[:, -100:])
            np.savetxt(dir_path_save + 'data_kap0.csv', self.data_kap0[:, -100:])            
            np.savetxt(dir_path_save + 'data_i_s.csv', self.data_i_s[:, -100:])
            np.savetxt(dir_path_save + 'data_is_c.csv', self.data_is_c[:, -100:])
            np.savetxt(dir_path_save + 'data_u.csv', self.data_u[:, -100:])
            np.savetxt(dir_path_save + 'data_cc.csv', self.data_cc[:, -100:])
            np.savetxt(dir_path_save + 'data_cs.csv', self.data_cs[:, -100:])
            np.savetxt(dir_path_save + 'data_cagg.csv', self.data_cagg[:, -100:])
            np.savetxt(dir_path_save + 'data_l.csv', self.data_l[:, -100:])
            np.savetxt(dir_path_save + 'data_n.csv', self.data_n[:, -100:])
            np.savetxt(dir_path_save + 'data_hy.csv', self.data_hy[:, -100:])
            np.savetxt(dir_path_save + 'data_hkap.csv', self.data_hkap[:, -100:])
            np.savetxt(dir_path_save + 'data_h.csv', self.data_h[:, -100:])            
            np.savetxt(dir_path_save + 'data_x.csv', self.data_x[:, -100:])
            np.savetxt(dir_path_save + 'data_ks.csv', self.data_ks[:, -100:])
            np.savetxt(dir_path_save + 'data_ys.csv', self.data_ys[:, -100:])
            np.savetxt(dir_path_save + 'data_ns.csv', self.data_ns[:, -100:])
            np.savetxt(dir_path_save + 'data_i_tax_bracket.csv', self.data_i_tax_bracket[:, -100:])
            np.savetxt(dir_path_save + 'data_R.csv', self.data_R[:, -100:])
            
            np.save(dir_path_save + 'v_yc_an', self.v_yc_an)
            np.save(dir_path_save + 'v_oc_an', self.v_oc_an)            
            np.save(dir_path_save + 'v_ys_an', self.v_ys_an)
            np.save(dir_path_save + 'v_os_an', self.v_os_an)
            np.save(dir_path_save + 'v_ys_kapn', self.v_ys_kapn)
            np.save(dir_path_save + 'v_os_kapn', self.v_os_kapn)
            np.save(dir_path_save + 'vn_yc', self.vn_yc)
            np.save(dir_path_save + 'vn_oc', self.vn_oc)            
            np.save(dir_path_save + 'vn_ys', self.vn_ys)
            np.save(dir_path_save + 'vn_os', self.vn_os)

            np.save(dir_path_save + 'R_y', self.R_y)
            np.save(dir_path_save + 'R_o', self.R_o)

            # these won't work due to dimension
            # np.savetxt(dir_path_save + 'sweat_div.csv', self.sweat_div)
            # np.savetxt(dir_path_save + 'sweat_val_dyna_sdf.csv', self.sweat_val_dyna_sdf)
            # np.savetxt(dir_path_save + 'sweat_val_dyna_fix.csv', self.sweat_val_dyna_fix)

            np.savetxt(dir_path_save + 'data_div_sweat.csv', self.data_div_sweat[:, -100:])
            np.savetxt(dir_path_save + 'data_val_sweat_dyna_sdf.csv', self.data_val_sweat_dyna_sdf[:, -100:])
            np.savetxt(dir_path_save + 'data_val_sweat_dyna_fix.csv', self.data_val_sweat_dyna_fix[:, -100:])            


            np.savetxt(dir_path_save + 'sind_age.csv', self.sind_age)
            np.savetxt(dir_path_save + 'cind_age.csv', self.cind_age)
            np.savetxt(dir_path_save + 's_age.csv', self.s_age)
            np.savetxt(dir_path_save + 'c_age.csv', self.c_age)
            np.savetxt(dir_path_save + 'y_age.csv', self.y_age)
            np.savetxt(dir_path_save + 'o_age.csv', self.o_age)
            np.savetxt(dir_path_save + 'ind_age.csv', self.ind_age)                             
            

import pickle        
import os.path
from sys import platform

def import_econ(name = 'econ.pickle'):

    if platform == 'darwin':
        from MacOSFile import pickle_load
        return pickle_load(name)
    else:
        with open(name, mode='rb') as f: econ = pickle.load(f)
        return econ


def export_econ(econ, name = 'econ.pickle'):


    if rank == 0:
        if platform == 'darwin':
            from MacOSFile import pickle_dump
            pickle_dump(econ, name)
        else:
            with open(name, mode='wb') as f: pickle.dump(econ, f)   

    return

#here we need to split two kinds of shock -- i_istate and is_old
def split_shock(path_to_data_shock, num_total_pop, size):


    m = num_total_pop // size
    r = num_total_pop % size

    data_shock = np.load(path_to_data_shock + '.npy')
    

    for rank in range(size):
        assigned_pop_range =  (rank*m+min(rank,r)), ((rank+1)*m+min(rank+1,r))
        np.save(path_to_data_shock + '_' + str(rank) + '.npy', data_shock[assigned_pop_range[0]:assigned_pop_range[1], :])

    return


if __name__ == '__main__':
    econ = import_econ()

    econ.get_policy()

    if rank == 0:
        econ.print_parameters()
                    
    econ.simulate_model()
    econ.calc_moments()
    # #econ.calc_age()

    export_econ(econ)
