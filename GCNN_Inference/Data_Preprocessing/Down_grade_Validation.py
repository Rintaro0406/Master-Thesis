from concurrent.futures import ProcessPoolExecutor
import time
import healpy as hp
import numpy as np

def downgrade_map_train(map_name):
    for i in range(10):
        #path="/project/ls-gruen/users/r.kanaki/Masterarbeit/Hirosaki_universe/Flask_Simulation_Validation/Kappa_Field/lognormal/map_realization_"+str(i)+"_cosmos_"+str(map_name)+"-f1z1.fits"
        path="/project/ls-gruen/users/r.kanaki/Masterarbeit/Hirosaki_universe/Flask_Validation_New/Kappa_Field/Gauss/map_realization_"+str(i)+"_cosmos_"+str(map_name)+"-f1z1.fits"
        MassMap = hp.read_map(path)
        MassMap  = hp.ud_grade(MassMap,nside_out=128, order_in="RING", order_out="NEST")
        hp.write_map("/project/ls-gruen/users/r.kanaki/Masterarbeit/Takahashi/Flask34_New/Validation/NSIDE128/Gauss/map_realization_"+str(i)+"_cosmos_"+str(map_name)+"-f1z1.fits", MassMap, overwrite=True)
        
    
    
def main():
    print('start')
    startTime = time.time()
    arg=np.linspace(0,249,250,dtype = int)
    with ProcessPoolExecutor(max_workers=50) as executor:
        rets=executor.map(downgrade_map_train, arg)
    counter=0
    for result in rets:
        print(counter)
        counter+=1
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
    print('finish')

if __name__=='__main__':
    main()