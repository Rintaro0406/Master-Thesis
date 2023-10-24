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
n_samples=300
temporary_file = np.zeros((n_samples,npix))
#path_to_test="/project/ls-gruen/users/r.kanaki/Masterarbeit/Takahashi/Flask34/Test/NSIDE2048/Gauss"
#path_to_test="/project/ls-gruen/users/r.kanaki/Masterarbeit/Takahashi/Flask34/Test/NSIDE2048/lognormal"
path_to_test="/project/ls-gruen/users/r.kanaki/Masterarbeit/Takahashi/Flask34_Final/Test/NSIDE128/Gauss"
Omega_M = np.zeros((n_samples))
Sigma_8 = np.zeros((n_samples))
counter=0
for i in range(300):
    counter = i
    path     = path_to_test+'/map_realization_'+str(i)+'-f1z1.fits'
    map_temp = hp.read_map(path)
    temporary_file[counter]   =  map_temp
    Omega_M[counter]          =  0.279
    Sigma_8[counter]          =  0.82
params = {'lognormal_map': temporary_file,
         'Omega_M': Omega_M,
         'sigma_8': Sigma_8}
#np.savez('/project/ls-gruen/users/r.kanaki/Masterarbeit/Takahashi/Flask34/Test/NSIDE2048/Test_Gauss.npz', **params)
np.savez('/project/ls-gruen/users/r.kanaki/Masterarbeit/Takahashi/Flask34_Final/Test/NSIDE128/Test_Gauss.npz', **params)
print('test datasets are finished')

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
print('finish')