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
    l  = np.linspace(1,6000, 6000,dtype=np.int32)
    LMAX = np.max(l)
    """
    Defines a cosmology in היראסאקי. 
    היראסאקי מוס שײנ
    """
    Obfid = 0.279-0.233
    hfid = 0.7
    nsfid = 0.97
    Omfid = 0.279
    sig8fid = 0.82

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
            return Cosmo.pk(k, z) 
        
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
        Cosmo      = self.LambdaCDM(Omega_m, sigma_8)
        Cl_2D      = np.zeros((len(self.l), 2))
        Cl_2D[:,0] = self.l
        ## here correcting the effect of nside resolution which is reported in arXiv:2304.01187
        NSIDE     = 2048
        l_lres    = Cl_2D[:,0]/(1.6*self.NSIDE)
        FRL_fac   = 1/(1.+l_lres**2) ## Finite resolution effect
        for i in range(len(self.l)):
            Cl_2D[:,1][i]=self.convergence_powerspectrum(Cosmo, self.l[i])
        Cl_2D[:,1]=Cl_2D[:,1]*FRL_fac
        return Cl_2D