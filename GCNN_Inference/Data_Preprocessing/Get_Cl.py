from concurrent.futures import ProcessPoolExecutor
import time
import healpy as hp
import numpy as np

#file_path ='/project/ls-gruen/users/r.kanaki/Masterarbeit/Takahashi/Flask34/Test/NSIDE128/Test_lognormal.npz'
file_path ='/project/ls-gruen/users/r.kanaki/Masterarbeit/Takahashi/Flask34/Test/NSIDE128/Test_Gauss.npz'
maps_data = np.load(file_path)
maps      = maps_data['lognormal_map']
ell       = np.linspace(1,400, 400,dtype=np.int32)
LMAX      = np.max(ell)-1
def get_ps(map_name):
    kappa_map      = maps[map_name]
    kappa_map_ring = hp.reorder(kappa_map, n2r=True)
    kappa_ps    = hp.anafast(kappa_map_ring, lmax=LMAX,use_pixel_weights=True)
    #np.savetxt("/project/ls-gruen/users/r.kanaki/Masterarbeit/Takahashi/Flask34/Test/NSIDE128/Cl_lognormal/Cl-"+str(map_name)+".dat",np.c_[kappa_ps])
    np.savetxt("/project/ls-gruen/users/r.kanaki/Masterarbeit/Takahashi/Flask34/Test/NSIDE128/Cl_Gauss/Cl-"+str(map_name)+".dat",np.c_[kappa_ps])
    return kappa_ps

def main():
    startTime = time.time()
    arg=np.linspace(0,299,300,dtype = int)
    with ProcessPoolExecutor(max_workers=50) as executor:
        rets=executor.map(get_ps, arg)
    counter=0
    for result in rets:
        counter+=1
    temporary_file = np.zeros((300,400))
    for i in range(300):
        path="/project/ls-gruen/users/r.kanaki/Masterarbeit/Takahashi/Flask34/Test/NSIDE128/Cl_lognormal/Cl-"+str(i)+".dat"
        #path="/project/ls-gruen/users/r.kanaki/Masterarbeit/Takahashi/Flask34/Test/NSIDE128/Cl_Gauss/Cl-"+str(i)+".dat"
        temporary_file[i]   = np.loadtxt(path)
        params = {'convergence powerspectrum': temporary_file}
    #np.savez('/project/ls-gruen/users/r.kanaki/Masterarbeit/Takahashi/Flask34/Test/NSIDE128/Test_lognormal_Cl.npz', **params)
    np.savez('/project/ls-gruen/users/r.kanaki/Masterarbeit/Takahashi/Flask34/Test/NSIDE128/Test_Gauss_Cl.npz', **params)
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
    
if __name__=='__main__':
    main()