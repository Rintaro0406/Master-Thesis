import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
from astropy.io import fits
from classy import Class
from scipy import integrate
import ctypes
import sys
import numpy as np
import os
from classy import Class
from scipy import integrate
from scipy import interpolate
import multiprocessing
import concurrent.futures
import time

class Cosmology_T17:
    """
    Cosmological Constants for this analysis
    1. It is motivating for the analysis using T17 N-body simulation (arXiv:2304.01187)
    2. Marginalizing \Lambda CDM parameters except \sigma_8 and \Omega_m
    LMAX is from LMAX=3*NSIDE
    """
    NSIDE=2048
    #### Because the running time of flask scales lineary with l and maybe I will never use the maps NSIDE=2048
    l  = np.linspace(1,3000, 3000,dtype=np.int32)
    LMAX = np.max(l)
    """
    Defines a cosmology in היראסאקי. 
    היראסאקי מוס שײנ
    """
    Obfid = 0.279-0.233
    hfid = 0.7
    nsfid = 0.97

    """
    Defines source redshift bins
    here I am using the redshift source bin34 of arXiv:2304.01187 which is consistent with map prjecting procedure of T17 arXiv:2304.01187
    """
    # בײֵטםן redshit distribution!
    gz=np.loadtxt("/project/ls-gruen/Takahashi/Rintaro_weak_lesning_maps/source_planes_Takahashi_DESY3_source_BINS_weights.dat",unpack=True)
    zs=gz[0][0:-8]
    g=gz[6][0:-8]
    zmax=np.max(zs)
    zmin=np.min(zs)
    
class Kappa_Powerspecrum_T17(Cosmology_T17):
    """
    1. Generating the powerpectrum using the Boltzmann solver and getting the matter powerspectrum
    2. Projecting matter powerspectrum in the line of sight using BIN 34 of source redshift disrtibution from arXiv:2304.01187 
    3. It is consisting with the map projectition procedure of T17 simulation (arXiv:2304.01187)
    4. Then I get the weak lensing convergence power spectrum
    """
    def __init__(self):
        """
        Initialise arrays and variables.
        """  
        ### Arrays which is used for Limber Integral
        self.z_range=np.linspace(0,self.zmax,100)
        self.q=np.zeros(len(self.z_range))
        self.w=np.zeros(len(self.z_range))
        self.H=np.zeros(len(self.z_range))
        self.OmM=np.zeros(len(self.z_range))
        self.P=np.zeros(len(self.z_range))
        
        ### Constants which is used for shell-thickness correction
        self.c1  = 9.5171*(0.0001)
        self.c2  = 5.1543*(0.001)
        self.al1 = 1.3063
        self.al2 = 1.1475
        self.al3 = 0.62793
        
        
    def LambdaCDM(self, Omega_m, sigma_8):
        """
        Define the Cosmology using the Boltzmann solver class
        Parameters:
        ----------
        Omega_m : Matter density
        sigma_8 : Present matter fluctuation on 8h^-1 MPC
        
        Notes:
        ----------
        I use the Halofit to calculate the non-linear power spectrum
        
        Returns:
        ----------
        LambdaCDM: LambdaCDM Cosmology
        """
        LambdaCDM = Class()
        h2=self.hfid*self.hfid
        LambdaCDM.set({'omega_b':self.Obfid*h2,'omega_cdm':(self.Omega_m-self.Obfid)*h2,'h':self.hfid,'n_s':self.nsfid,
                        'tau_reio':0.05430842,'N_ur':3.046,'z_max_pk':10.0,'sigma8':self.sigma_8,'non linear':'HALOFIT'})
        LambdaCDM.set({'output':'mPk','P_k_max_1/Mpc':10.})
        # run class
        LambdaCDM.compute()
        return LambdaCDM
    
    def comoving_distance(self,Cosmo,z):
        """
        Calculate the comoving distance using class
        Parameters:
        ----------
        Cosmo: Cosmology from class
        z: redshift
        
        Returns:
        comoving_distance: comoving distance in MPC/h^-1
        """
        return Cosmo.angular_distance(z)*(1.+z)
    
    def Omega_M(self,Cosmo):
        """
        Calculate the matter density using class
        Parameters:
        ----------
        Cosmo: Cosmology from class
        z: redshift
        
        Returns:
        Omega_m: matter density at given redshift z
        """
        return Cosmo.Omega_m()
    
    def Hubble(self, Cosmo, z):
        """
        Calculate the Hubble paramter using class
        Parameters:
        ----------
        Cosmo: Cosmology from class
        z: redshift
        
        Returns:
        Hubble: Hubble parameter at given redshift z
        """
        return Cosmo.Hubble(z)
    
    def P_NL(self,Cosmo, k, z):
        """
        Calculate the non linear 3D matter powerspectrum and make sure it's not going above maximum of k
        Also multiplying the shell thickness effect which is reported in arXiv:2304.01187

        Parameters:
        ----------
        Cosmo: Cosmology from class
        z: redshift
        k: frequency
        
        Returns:
        ----------
        P_NL: non linear powerspectrum in MPc^3/h^-3 which is corrected by the shell thickness effect
        """
        kmax = 10. #from test in last year April
        # we do not compute matter power spectrum at scales smaller than the limit set by kmax
        if k > kmax: 
            return 0
        else:
        # here correcing the shell-thickness effect
            return Cosmo.pk(k, z) * ((1.+self.c1*(k**(-self.al1)))**self.al1)/((1.+self.c2*(k**(-self.al2)))**self.al3)
        
    def lensing_kernel(self, Cosmo, z, H_0, Omega0_m):
        """
        Parameters:
        ----------
        z: redshift

        Returns:
        W: lensing kernel mutliplied by comoving distance at the redshift of z
        """
        lensing_kernel=0
        for i in range(len(self.zs)):
            if z<self.zs[i]:
                chi_z = self.comoving_distance(Cosmo,z)
                chi_zs = self.comoving_distance(Cosmo, self.zs[i])
                lensing_kernel+= 3./2. * H_0 * H_0 * Omega0_m * (1.+z) * (chi_zs - chi_z) / chi_zs*self.g[i]
        # here the output is lensing kernel mutliplied by comoving distance at the redshift of z
        ## in order to avoid the division of extremely large number  
        return lensing_kernel
    
        
    def convergence_powerspectrum(self, Cosmo, l):
        """
        Calculate the Limber integral using DES Y3 all source bins
        Parameters:
        ----------
        Omega_m : Matter density
        sigma_8 : Present matter fluctuation on 8h^-1 MPC
        
        Returns:
        ----------
        Pk_2D: convergence powerspectrum
        """
        for i in range(len(self.z_range)):
            self.H[i]=self.Hubble(Cosmo,self.z_range[i])
            self.w[i]=self.comoving_distance(Cosmo,self.z_range[i])
            self.P[i]=self.P_NL(Cosmo,l/self.w[i], self.z_range[i])
            self.q[i]=self.lensing_kernel(Cosmo, self.z_range[i], self.Hubble(Cosmo,0),self.Omega_M(Cosmo))
        Pk_2D = np.trapz((1/self.H)* self.P*self.q**2 ,self.z_range)
        return Pk_2D

    def Cl(self, Omega_m, sigma_8):
        """
        Parameters:
        ----------
        Omega_m : Matter density
        sigma_8 : Present matter fluctuation on 8h^-1 MPC
        
        Note:
        ----------
        output is already corrected by nside resolution effect which is reported in arXiv:2304.01187
        
        Returns:
        ----------
        Cl_2D: convergence powerspectrum which is the array of ([l, Cl])
        l : multipole
        Cl: Convergence power spectrum 
        """
        self.Omega_m = Omega_m
        self.sigma_8 = sigma_8
        Cosmo      = self.LambdaCDM(self.Omega_m, self.sigma_8)
        Cl_2D      = np.zeros((len(self.l), 2))
        Cl_2D[:,0] = self.l
        ## here correcting the effect of nside resolution which is reported in arXiv:2304.01187
        NSIDE     = 2048
        l_lres    = Cl_2D[:,0]/(1.6*self.NSIDE)
        FRL_fac   = 1/(1.+l_lres**2)
        for i in range(len(self.l)):
            Cl_2D[:,1][i]=self.convergence_powerspectrum(Cosmo, self.l[i])
        Cl_2D[:,1]=Cl_2D[:,1]*FRL_fac
        return Cl_2D
    
class Generate_Flask_maps(Kappa_Powerspecrum_T17):
    """
    1. Loading the sample of cosmological parameters (\sigma_8 and \Omega_m)
    2. Getting the log normal shift parameter using CosMomentum (arXiv:1912.06621)
    3. Generating Flask log normal maps (arXiv:1602.08503) for 10 realization for given cosmology
    """
    def generate_log_normal_maps(self, data):
        """
        1. Get the log-normal convergence maps from given prior range of Cosmology using flasks.
        2. Excuting the function self.generate_log_normal_map for this prior range
        Input:
        ----------
        data :path to the list of array which includes two cosmological parameter which is Omega_m and sigma_8.
        """
        startTime_total = time.time()
        parameters = np.load(data)
        Omega_m = parameters['omega_matter']
        sigma_8 = parameters['sigma_8']
        ### loop for cosmology 
        for i in range(870, len(Omega_m)):
            self.generate_log_normal_map(Omega_m[i], sigma_8[i], i)
        executionTime_total = (time.time() - startTime_total)
        print('Execution time in seconds for total: ' + str(executionTime_total))
        
    def generate_log_normal_map(self, Omega_m, sigma_8, cosmo_indx):
        """
        1. getting the powerspectrum using class (python class here I mean) Kappa_Powerspecrum_T17
        1. excecuting CosMomentum via self.log_normal_shift_parameter
        2. excecuting flask via self.use_flask for 10 realization each cosmologies
        Parameters:
        ----------
        Omega_m : Matter density
        sigma_8 : Present matter fluctuation on 8h^-1 MPC
        cosmo_indx : index of the cosmology
        
        Notes:
        ----------
        I use the Halofit to calculate the non-linear power spectrum
        
        Returns:
        ----------
        Good thing
        """
        ### getting convergence power spectrum using Cl
        cl=self.Cl(Omega_m, sigma_8)
        startTime_cosmomentum = time.time()
        shift_for_sources=self.log_normal_shift_parameter(Omega_m, sigma_8)
        executionTime_cosmomentum = (time.time() - startTime_cosmomentum)
        print('די צייט פאר קאסמאמענטומ: ' + str(executionTime_cosmomentum))
        ###  here saving the convergence power spectraand log normal shift parameters which is input of flask
        params = {'convergence powerspectrum': cl,
                  'Omega_m': Omega_m,
                  'sigma_8': sigma_8,
                  'log_nomral_shift_parameter': shift_for_sources}
        #### Define your path which you want to save Inputs
        np.savez("/project/ls-gruen/users/r.kanaki/Masterarbeit/Hirosaki_universe/Flask_Simulation_Final/Inputs_Flask/Inputs-"+str(cosmo_indx)+"_Flask.npz", **params)
        #### which is input of flask (see prior.config)
        np.savetxt("/project/ls-gruen/users/r.kanaki/Masterarbeit/Hirosaki_universe/Flask_Simulation_Final/Cl/Cl-"+str(cosmo_indx)+"-f1z1f1z1.dat",np.c_[cl[:,0],cl[:,1]])
        startTime_flask = time.time()
        self.use_flask(shift_for_sources, cosmo_indx)
        executionTime_flask = (time.time() - startTime_flask)
        print('די צייט פאר פלאסק: ' + str(executionTime_flask))
        #### Back in the directory which you saved this code
        os.chdir("/home/r/R.Kanaki/Masterarbeit/Juni_2023/New_Flask_Simulation")        
        
    def log_normal_shift_parameter(self, Omega_m, sigma_8):
        """
        Get the log-normal shift parameter from given Cosmology using CosMomentum (arXiv:1912.06621)
        Parameters:
        ----------
        Omega_m : Matter density
        sigma_8 : Present matter fluctuation on 8h^-1 MPC
        
        Notes:
        ----------
        I am using the source redshift distribution from BIN 34 of source redshift disrtibution from arXiv:2304.01187 
        
        Returns:
        ----------
        shift_for_sources : LambdaCDM Cosmology itself
        """
        ### the directory which you saved CosMomentum
        os.chdir("/home/r/R.Kanaki/Masterarbeit/April_2023/CosMomentum-Kaputtmachen_ok")
        os.system("cd cpp_code; make DSS")
        lib=ctypes.CDLL("./cpp_code/DSS.so")
        
        ### Using the cosmological paramters which is used by T17 simulations (arXiv:2304.01187)
        a_initial = 0.000025
        a_final = 1.0
        
        Omfid   = Omega_m
        Obfid   = self.Obfid
        hfid    = self.hfid
        nsfid   = self.hfid
        sig8fid = sigma_8
        
        density_sample_1 = 69.47036304452095/(np.pi*30.0**2)
        b1_sample_1 = 1.8
        b2_sample_1 = 0.0
        a0 = 1.26
        a1 = 0.28

        z = 0.0 # everywhere in this codes
        
        
        # initialising a new universe and its matter content
        initialise_new_Universe = lib.initialise_new_Universe
        initialise_new_Universe.argtypes = [ ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        initialise_new_Universe.restype = None
        initialise_new_Universe(a_initial, a_final, Omfid, Obfid, 0.0, 1.0-Omfid, sig8fid, nsfid, hfid, -1.0, 0.0)
        
        add_projected_galaxy_sample = lib.add_projected_galaxy_sample
        add_projected_galaxy_sample.argtypes = [         ctypes.c_int,         ctypes.c_char_p,                    ctypes.c_double,  ctypes.c_double,  ctypes.c_double, ctypes.c_double, ctypes.c_double]
        add_projected_galaxy_sample.restype = None


        return_N_max_projected = lib.return_N_max_projected
        return_N_max_projected.argtypes = [              ctypes.c_int,        ctypes.c_double,       ctypes.c_double]
        return_N_max_projected.restype = ctypes.c_int


        change_parameters_of_projected_galaxy_sample = lib.change_parameters_of_projected_galaxy_sample
        change_parameters_of_projected_galaxy_sample.argtypes = [              ctypes.c_int, ctypes.c_double,                    ctypes.c_double,  ctypes.c_double,  ctypes.c_double, ctypes.c_double, ctypes.c_double]
        change_parameters_of_projected_galaxy_sample.restype = None

        change_b2_to_minimise_negative_densities_projected = lib.change_b2_to_minimise_negative_densities_projected
        change_b2_to_minimise_negative_densities_projected.argtypes = [              ctypes.c_int,        ctypes.c_double,       ctypes.c_double]
        change_b2_to_minimise_negative_densities_projected.restype = ctypes.c_double
        
        
        # Creating first galaxy sample (lenses)
        n_of_z_file_str = 'Data/redshift_distributions/pofz_Y1_redMaGiC_bin4.dat'
        n_of_z_file = ctypes.c_char_p(n_of_z_file_str.encode('utf-8'))
        add_projected_galaxy_sample(0, n_of_z_file, density_sample_1, b1_sample_1, b2_sample_1, a0, a1)
        
        # Creating second galaxy sample (sources)
        # ALso here modified to BIN 34 of source redshift disrtibution from arXiv:2304.01187 
        n_of_z_file_str = 'Data/redshift_distributions/Laurence_bin34.tab'
        n_of_z_file = ctypes.c_char_p(n_of_z_file_str.encode('utf-8'))
        add_projected_galaxy_sample(0, n_of_z_file, density_sample_1, b1_sample_1, b2_sample_1, a0, a1)
        
        return_R_in_Mpc_over_h_from_angular_scale = lib.return_R_in_Mpc_over_h_from_angular_scale
        return_R_in_Mpc_over_h_from_angular_scale.argtypes = [              ctypes.c_int,        ctypes.c_double]
        return_R_in_Mpc_over_h_from_angular_scale.restype = ctypes.c_double
        
        configure_FLASK_for_delta_g_and_kappa = lib.configure_FLASK_for_delta_g_and_kappa
        configure_FLASK_for_delta_g_and_kappa.argtypes = [ctypes.c_int,        ctypes.c_double, ctypes.c_double, ctypes.c_double,             ctypes.c_int,               ctypes.c_int,         ctypes.c_char_p]
        configure_FLASK_for_delta_g_and_kappa.restype = None
        
        return_lognormal_shift_for_individual_FLASK_bin = lib.return_lognormal_shift_for_individual_FLASK_bin
        return_lognormal_shift_for_individual_FLASK_bin.argtypes = [       ctypes.c_double,               ctypes.c_int,            ctypes.c_int]
        return_lognormal_shift_for_individual_FLASK_bin.restype = ctypes.c_double
        
        theta_in_arcmin = 10
        shift_for_sources = return_lognormal_shift_for_individual_FLASK_bin(theta_in_arcmin, 1, 0)
        
        return shift_for_sources
    
    def use_flask(self, shift_for_sources, cosmo_indx):
        """
        Running the Flask for giving convergence power spectra and shift parameter in order to get the log normal maps
        
        Parameters:
        ----------
        Omega_m : Matter density
        sigma_8 : Present matter fluctuation on 8h^-1 MPC
        
        Notes:
        ----------
        Getting the weak lensing convergence maps for 10 realization for one cosmology for Gaußian and log normal realization (20 maps total)
        """
        
        ### the directly which you saved _info.dat
        ### here giving the information of shift parameters
        foj=open("/home/r/R.Kanaki/Masterarbeit/Juni_2023/New_Flask_Simulation/Flask_info.dat","w")
        foj.writelines("# Field number, z bin number, mean, shift, field type, zmin, zmax""\n")
        foj.writelines("# Types: 1-galaxies 2-shear""\n")
        foj.writelines("     {FieldNumber:01d}      {zBinNumber:01d}   {mean:03f}   {shift:06f}      {FieldType:01d}   {zmin:06f}   {zmax:06f}".format(FieldNumber=1, zBinNumber=1, mean=0., shift=shift_for_sources, FieldType=2,zmin=self.zmin,zmax=self.zmax))
        foj.close()
        ### pyFlask works only the directly which you saved flask 
        os.chdir("/home/r/R.Kanaki/flask")
        import pyFlask
        i=1
        ### for 10 random realization
        for i in range(10):
            #Because the plan is making 1000 cosmos which have different cosmological parameters, and also varying every realization...
            seed=cosmo_indx+i*1000+777
            #### here you choose the directory where you saved .info and .config
            #### here you input the powerspectrum
            ### generate log normal field
            pyFlask.flask(["flask","../Masterarbeit/Juni_2023/New_Flask_Simulation/Flask.config","RNDSEED:",str(seed),"DIST:", "LOGNORMAL",
                          "CL_PREFIX:","/project/ls-gruen/users/r.kanaki/Masterarbeit/Hirosaki_universe/Flask_Simulation_Final/Cl/Cl-"+str(cosmo_indx)+"-"])
            os.chdir("/project/ls-gruen/users/r.kanaki/Masterarbeit/Hirosaki_universe")
            os.rename("map-f1z1.fits", "Flask_Simulation_Final/Kappa_Field/lognormal/map_realization_"+str(i)+"_cosmos_"+str(cosmo_indx)+"-f1z1.fits")
            os.chdir("/home/r/R.Kanaki/flask")
            ### generate Gaussian field
            pyFlask.flask(["flask","../Masterarbeit/Juni_2023/New_Flask_Simulation/Flask.config","RNDSEED:",str(seed),"DIST:", "GAUSSIAN",
                          "CL_PREFIX:","/project/ls-gruen/users/r.kanaki/Masterarbeit/Hirosaki_universe/Flask_Simulation_Final/Cl/Cl-"+str(cosmo_indx)+"-"])
            os.chdir("/project/ls-gruen/users/r.kanaki/Masterarbeit/Hirosaki_universe")
            os.rename("map-f1z1.fits", "Flask_Simulation_Final/Kappa_Field/Gauss/map_realization_"+str(i)+"_cosmos_"+str(cosmo_indx)+"-f1z1.fits")
            os.chdir("/home/r/R.Kanaki/flask")