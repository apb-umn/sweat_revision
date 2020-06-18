# this python file simply records parameters in a csv file for display purpose
# It is assumed that 

#module_name='parameters_newbench'
#save_path='save_data_newbench/'
#exec('from '  + module_name + ' import *')

import numpy as np
import pandas as pd
import stats
pd.options.display.float_format = '{:.4f}'.format


f = open(save_path + 'preferences.csv', 'w')

f.write('Preferences')
f.write('\n')
f.write('Discount factor,' + str(beta))
f.write('\n')
f.write('Paternalistic discount factor,' + str(iota))
f.write('\n')                
f.write('Consumption weight,' + str(eta))
f.write('\n')
f.write('Leisure weight,' + str(1.-eta))
f.write('\n')        
f.write('Intertemporal elasticity inverse,' + str(mu))
f.write('\n')                
f.write('C-corporate good share in consumption,' + str(ome))
f.write('\n')
f.write('Love of business parameter,' + str(0.)) #this code does not have this term.
f.write('\n')
f.write('Labor productivity decline for the old,' + str(tau_wo)) #this code does not have this term.
f.write('\n')
f.write('Business productivity decline for the old,' + str(tau_bo)) #this code does not have this term.
f.write('\n')

f.close()

f = open(save_path + 'technologies.csv', 'w')

f.write('Technologies')
f.write('\n')                
f.write('Technology growth,'+ str(grate))
f.write('\n')                
f.write('C-corporate fixed asset share,'+ str(theta))
f.write('\n')                
f.write('Private business fixed asset share,'+ str(alpha))
f.write('\n')                
f.write('Private business sweat capital share,'+ str(phi))
f.write('\n')                
f.write('Private business ces labor composite share,'+ str(1. - alpha - phi)) # or nu
f.write('\n')                
f.write('Private business employee hours share parameter in ces labor composite,'+ str(varpi))
f.write('\n')                
f.write('Private business hours substitution parameter in ces labor composite,'+ str(upsilon))
f.write('\n')                
f.write('Fixed asset depreciation,'+ str(delk))
f.write('\n')                
f.write('Sweat capital depreciation,'+ str(delkap))
f.write('\n')                
f.write('Sweat capital owner hour share,'+ str(veps))
f.write('\n')                
f.write('Sweat capital c-good share,'+ str(vthet))
f.write('\n')                
f.write('Sweat capital deterioration for workers,'+ str(la))
f.write('\n')                
f.write('Sweat capital deterioration for bequests,'+ str(la_tilde))
f.write('\n')                
f.write('Working-capital constraint,'+ str(chi))
f.write('\n')                
f.write('Non-business production relative to GDP,'+ str(yn_div_gdp))
f.write('\n')                
f.write('Non-business consumption relative to GDP,'+ str(xnb_div_gdp))
f.write('\n')                


f.close()

f = open(save_path + 'policies.csv', 'w')

f.write('Policy variables except for income tax')
f.write('\n')                
f.write('Consumption tax,'+ str(tauc))
f.write('\n')                
f.write('Dividends tax,'+ str(taud))
f.write('\n')                
f.write('Profits tax,'+ str(taup))
f.write('\n')
f.write('Government expenditure relative to GDP,'+ str(g_div_gdp))
f.write('\n')
f.write('Retirement Transfer,'+ str(trans_retire))
f.write('\n')


f.close()


f = open(save_path + 'labor_income_tax.csv', 'w')

nbracket_tmp = np.zeros(len(taun) + 1)
nbracket_tmp[1:-1] = nbracket_div_gdp[:]
nbracket_tmp[0]  = -np.inf
nbracket_tmp[-1] = np.inf

f.write('nbracket_left, nbracket_right, taun, psin\n')        
for i in range(len(taun)):
    f.write(str(nbracket_tmp[i]) + ', ' + str(nbracket_tmp[i+1]) + ', ' +  str(taun[i]) + ', ' +  str(psin[i]))
    f.write('\n')                    

f.close()

f = open(save_path + 'business_income_tax.csv', 'w')

bbracket_tmp = np.zeros(len(taub) + 1)
bbracket_tmp[1:-1] = bbracket_div_gdp[:]
bbracket_tmp[0]  = -np.inf
bbracket_tmp[-1] = np.inf


f.write('bbracket_left, bbracket_right, taub, psib\n')        
for i in range(len(taub)):
    f.write(str(bbracket_tmp[i]) + ', ' + str(bbracket_tmp[i+1]) + ', ' +  str(taub[i]) + ', ' +  str(psib[i]))
    f.write('\n')

f.close()


f = open(save_path + 'labor_productivity_transition_table.csv', 'w')
f.write('Transition of eps')
f.write('\n')
f.write('')
for iz in range(num_eps):
    f.write(',' + str( epsgrid[iz]))
f.write('\n')

for ieps in range(num_eps):
    f.write(str(ieps))
    # f.write(str(epsgrid[ieps]))
            
    for iepsp in range(num_eps):
        f.write(',' + str(prob_eps[ieps, iepsp]))

    f.write('\n')            
            
f.close()


f = open(save_path + 'business_productivity_transition_table.csv', 'w')
f.write('Transition of z')
f.write('\n')
f.write('')
for iz in range(num_z):
    f.write(',' + str(zgrid[iz]))
f.write('\n')

for iz in range(num_z):
    f.write(str(iz))
    #f.write(str(zgrid[iz]))
    
    for izp in range(num_z):
        f.write(',' + str(prob_z[iz, izp]))
        
    f.write('\n')

f.close()


f = open(save_path + 'age_transition_table.csv', 'w')
f.write('Transition of age')
f.write('\n')
f.write(', y, o')
f.write('\n')
f.write('y,' + str(prob_yo[0,0]) + ',' + str(prob_yo[0,1]))
f.write('\n')
f.write('o,' + str(prob_yo[1,0]) + ',' + str(prob_yo[1,1]))
f.write('\n')


f.close()
