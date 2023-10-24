## Making one .npz file for 1000*10 flask maps
## In order to train comfortable
## Output of Flask is RING ordering, but INPUT of DeepSphere must be NEST ordering
import numpy as np
import healpy as hp
import time


print('start')
startTime = time.time()
nside=128
npix = hp.nside2npix(nside)
n_samples=10000
#n_samples=25000
temporary_file = np.zeros((n_samples,npix))
#path_to_test="/project/ls-gruen/users/r.kanaki/Masterarbeit/Takahashi/Flask34_Final/Validation/NSIDE128/lognormal"
#path_to_test="/project/ls-gruen/users/r.kanaki/Masterarbeit/Takahashi/Flask34_Final/Validation/NSIDE128/Gauss"
path_to_test="/project/ls-gruen/users/r.kanaki/Masterarbeit/Takahashi/Flask34_Final/Training/NSIDE128/Gauss"
#path_to_test="/project/ls-gruen/users/r.kanaki/Masterarbeit/Takahashi/Flask34_Final/Training/NSIDE128/lognormal"
LHS_data = np.load('/project/ls-gruen/users/r.kanaki/Masterarbeit/Hirosaki_universe/Flask_Simulation_New/Hirosaki_LHS_parameter_file_New.npz')
#LHS_data = np.load('/project/ls-gruen/users/r.kanaki/Masterarbeit/Hirosaki_universe/Flask_Validation_New/Hirosaki_LHS_parameter_file_New.npz')
Omega_M = np.zeros((n_samples))
Sigma_8 = np.zeros((n_samples))
counter=0
#for i in range(250):
for i in range(1000):
    print(i)
    for j in range(10):
        #counter  = j*2500 + i
        counter = j*1000 +i
        path     = path_to_test+'/map_realization_'+str(j)+'_cosmos_'+str(i)+'-f1z1.fits'
        map_temp = hp.read_map(path)
        temporary_file[counter]   =  map_temp
        Omega_M[counter]          =  LHS_data['omega_matter'][i]
        Sigma_8[counter]          =  LHS_data['sigma_8'][i]
params = {'lognormal_map': temporary_file,
         'Omega_M': Omega_M,
         'sigma_8': Sigma_8}
#np.savez('/project/ls-gruen/users/r.kanaki/Masterarbeit/Takahashi/Flask34_Final/Validation/NSIDE128/train_lognormal.npz', **params)
#np.savez('/project/ls-gruen/users/r.kanaki/Masterarbeit/Takahashi/Flask34_Final/Validation/NSIDE128/train_Gauss.npz', **params)
np.savez('/project/ls-gruen/users/r.kanaki/Masterarbeit/Takahashi/Flask34_Final/Training/NSIDE128/train_Gauss.npz', **params)
#np.savez('/project/ls-gruen/users/r.kanaki/Masterarbeit/Takahashi/Flask34_Final/Training/NSIDE128/train_lognormal.npz', **params)
print('test datasets are finished')

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
print('finish')