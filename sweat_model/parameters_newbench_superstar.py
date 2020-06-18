import sys
sys.path.insert(0, './library/')


import numpy as np
from quantecon.markov.approximation import tauchen
from PiecewiseLinearTax import get_consistent_phi



### paths
save_data_folder='save_data_newbench_superstar'

### number of cores utilize
num_core = 640  # number of cores for parallel

# prices 
p=2.3881100774613797
rc=0.06551480433432233 

# S-corp production function
alpha    = 0.3 # capital share parameter for S-corp
phi      = 0.15 # sweat capital share for S-corp
# composite labor share parameter is defined by nu = 1. - alpha - phi


# capital depreciation parameters
delk     = 0.041 # physical capital depreciation rate
delkap   = 0.0579 # sweat ccapitaldepreciation rate for S-owners
la       = 0.20 # 1-la sweat capital depreciation rate for C-workers

# household preference
beta     = 0.98 # discount rate
eta      = 0.42 # utility weight on consumption
mu       = 1.5  # risk aversion coeff. of utility

# final good aggregator
rho      = 0.01 # Elasticity of substitutions parameter between C-S goods
#ome      = 0.448704166754681 # benchmark
ome=0.4484629603536501

# linear tax
tauc     = 0.065 # consumption tax
taud     = 0.133 # dividend tax
taup     = 0.36 # profit tax 

# C-corp production function
#theta    = 0.5000702399881483 #capital share parameter for C-corporation Cobb-Douglas technology
A        = 1.577707121233179 #TFP parameter for C-corporation Cobb-Douglas technology
#theta    = 0.502257346263903 #benchmark
theta=0.5223943297749546

# parameters for sweat capital production function
veps     = 0.408 # owner's time share
vthet    = 1.0 - veps # C-good share
zeta     = 1.0 # TFP term 

# CES aggregator 
upsilon  = 0.5 #elasticity parameter between owner's labor and employee's labor
#varpi    = 0.575016619113812     #newwageprocess, lambda and better gdp guess
varpi=0.5220690802661696


# other parameters
chi      = 0.0 # borrowing constraint parameter a' >= chi ks    
grate    = 0.02 # Growth rate of the economy


# state space grids
# state space grid requires four parameters
# min, max, curvature, number of grid point
# if curvature is 1, the grid is equi-spaced.
# if curvature is larger than one, it puts more points near min


amin = 0.0 
amax = 1200.
acurve = 2.0
num_a = 40

kapmin = 0.0
kapmax = 25.0
kapcurve = 2.0
num_kap = 30

# a function that generates non equi-spaced grid
def curvedspace(begin, end, curve, num=100):
    ans = np.linspace(0., (end - begin)**(1.0/curve), num) ** (curve) + begin
    ans[-1] = end #so that the last element is exactly end
    return ans

agrid = curvedspace(amin, amax, acurve, num_a) 
kapgrid = curvedspace(kapmin, kapmax, kapcurve, num_kap)


# productivity shock

rho_z = 0.7
sig_z = 0.1
num_z = 5

rho_eps = 0.70446      # Estimates based on:
sig_eps = 0.1598256    # Low, Meghir, Pistaferri (2011)
num_eps = 5

mc_z   = tauchen(rho = rho_z  , sigma_u = sig_z  , m = 3, n = num_z) # discretize z
mc_eps = tauchen(rho = rho_eps, sigma_u = sig_eps, m = 3, n = num_eps) # discretize eps


# prob_z_filepath = './DeBacker/debacker_prob_z.csv'
# prob_eps_filepath = './DeBacker/debacker_prob_eps.csv'
# prob_z   = np.loadtxt(prob_z_filepath)# read transition matrix from DeBacker
# prob_eps = np.loadtxt(prob_eps_filepath) # read transition matrix from DeBacker

prob_z=np.array([[6.11518699e-01, 1.70401029e-01, 9.83162307e-02, 6.45004050e-02,
        5.52636369e-02, 0.00000000e+00],
       [1.72256453e-01, 5.50902581e-01, 1.87292256e-01, 6.43230641e-02,
        2.52256453e-02, 0.00000000e+00],
       [9.86771708e-02, 1.90740195e-01, 4.75423031e-01, 1.90404047e-01,
        4.47555561e-02, 0.00000000e+00],
       [5.99202977e-02, 5.46976541e-02, 1.63699901e-01, 5.58095098e-01,
        1.63587049e-01, 0.00000000e+00],
       [4.54759607e-02, 9.43063646e-03, 3.42224662e-02, 1.35067007e-01,
        7.74914190e-01, 8.89739922e-04],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        2.22222222e-02, 9.77777778e-01]])


#prob_eps = np.array([
#    [7.677448011412900675e-01, 1.871142489300404721e-01, 3.514094992866936135e-02, 9.999999999999998473e-03, 0.000000000000000000e+00],
#    [1.517699300484544878e-01, 6.718485353616006073e-01, 1.562200948768088515e-01, 2.016143971313614711e-02, 0.000000000000000000e+00],
#    [3.965872604973709470e-02, 1.082936302486854629e-01, 6.738055411740797584e-01, 1.630714655523662071e-01, 1.517063697513145425e-02],
#    [1.489408241547009355e-02, 2.489408241547009029e-02, 1.187289889856411040e-01, 7.197881648309402136e-01, 1.216946813524785453e-01],
#    [0.000000000000000000e+00, 0.000000000000000000e+00, 1.490233421697592653e-02, 9.276910281735840924e-02, 8.923285629656657614e-01]])
prob_eps = mc_eps.P

prob = np.kron(prob_z, prob_eps) 

# prob = np.load('./DeBacker/prob_epsz.npy') # transition matrix from DeBacker et al.

zgrid_old = np.exp(mc_z.state_values) ** 2.0
num_z = 6
zgrid=np.ones(num_z)
zgrid[:5]=zgrid_old
zgrid[5]=zgrid_old[4]*1.25

epsgrid = np.exp(mc_eps.state_values) 


is_to_iz = np.array([i for i in range(num_z) for j in range(num_eps)])
is_to_ieps = np.array([j for i in range(num_z) for j in range(num_eps)])    
# is_to_iz = np.load('./input_data/is_to_iz.npy') #convert s to eps
# is_to_ieps = np.load('./input_data/is_to_ieps.npy') #convert s to z

# lifecycle-specific parameters
prob_yo = np.array([[44./45., 1./45.], [3./45., 42./45.]]) # transition matrix for young-old state
#[[y -> y, y -> o], [o -> y, o ->o]]
iota     = 1.0 # paternalistic discounting rate. 
la_tilde = 0.1 # 1 - la_tilde is sweat capital depreciation rate
tau_wo   = 0.5 # productivity eps is replaced by tau_wo*eps if the agent is old
tau_bo   = 0.5 # productivity z   is replaced by tau_bo*z   if the agent is old
trans_retire = 0.80 # receives this transfer if the agent is old.



# GDP guess and GDP_indexed parameters (except for nonlinear tax)

g_div_gdp = 0.133 # government expenditure relative to GDP
yn_div_gdp = 0.266 # non-business production relative to GDP
xnb_div_gdp = 0.110 # non-business consumption relative to GDP
GDP_guess = 4.12  #a guess for GDP value. This needs to be consistent with simulated GDP


g = g_div_gdp*GDP_guess # actual GDP 
yn = yn_div_gdp*GDP_guess # actual non-business production
xnb = xnb_div_gdp*GDP_guess # actual non-business consumption





###  nonlinear tax functions ###

# business tax (see Table A11 in the draft)
taub = np.array([.14, .183, .201, .235, .262, .269, 0.280])
bbracket_div_gdp = np.array([0.153, 0.304, 0.912, 2.667, 5.727,9.104]) # brackets relative to GDP

bbracket = bbracket_div_gdp * GDP_guess

# one intercept should be fixed
psib_fixed = 0.06 # value for the fixed intercept
bbracket_fixed = 2 # index for the fixed intercept
psib = get_consistent_phi(bbracket, taub, psib_fixed, bbracket_fixed) # obtain consistent intercepts

# labor income tax (see Table A11 in the draft)
taun = np.array([.293, .324, .343, .390, .40, .408, .419])
nbracket_div_gdp = np.array([.173, .262, .404, .732, 1.409, 3.3138]) # brackets relative to GDP
nbracket = nbracket_div_gdp * GDP_guess

# one intercept should be fixed
psin_fixed = 0.06 # value for the fixed intercept
nbracket_fixed = 4 # index for the fixed intercept
psin = get_consistent_phi(nbracket, taun, psin_fixed, nbracket_fixed) # obtain consistent intercepts

# computational parameters
sim_time = 1000 # simulation length
num_total_pop = 100_000 # population in simulatin

num_suba_inner = 20 #the number of equi-spaced subgrid between agrid
num_subkap_inner = 30 #the number of equi-spaced subgrid between kapgrid




# computational parameters for exogenous shocks
path_to_data_i_s = './tmp/data_i_s' # temporary directory for shock
path_to_data_is_o = './tmp/data_is_o' # temporary directory for shock
buffer_time = 2_000 #

 
