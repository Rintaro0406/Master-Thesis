## Section 2: GCNN Inference
The complementary aspect of simulation is inference, where we aim to estimate the cosmological parameters from the density field. In my inference pipeline, I utilize the DeepSphere framework, which incorporates graph convolutional neural networks (GCNN) instead of summarizing data to a two-point function. This directory contains my code for the training steps of GCNN models and data preprocessing. The content in this subdirectory primarily aligns with Chapter 7 of my Master's thesis.

## Structure of Inference Pipeline

[<img src="GCNN_Model.jpg" width="700"/>](GCNN_Model.jpg)

The primary focus of this thesis is to investigate the validity of the Lognormal model approximation. As such, I trained GCNN networks using the Lognormal model, and additionally generated the Gaussian model for comparison. We can also assess the validity of our neural network model by comparing it with the Fisher contour.

(i) **Inputs** The input data consists of maps that have been downgraded to NSIDE=128, along with cosmological parameters as labels. For the Gaussian model, I used Gaussian maps as training data, and for the Lognormal model, I used lognormal maps as training data. Apart from the input data, the model architecture and hyperparameters are the same for both models. 

(ii) **DeepSphere** In contrast to CNN, GCNN can handle input data that is not defined in Euclidean space. DeepSphere is a GCNN package designed for spherical data, such as HEALPix maps. I employed layers from DeepSphere to process full-sky maps.

(iii) **Loss Function** The loss function used is the negative Gaussian log-likelihood loss. This loss function provides estimates of parameters and the covariance matrix under the Gaussian assumption.

(iV) **Outouts** The first two outputs are estimates of model parameters, specifically $\Omega_m$ and $\sigma_8$. The remaining three outputs are estimates of the covariance matrix.


## Contents

In this directory, you'll find the following code and Python notebooks:
1. **Validation_Simulaion_Pipeline.ipynb:** This notebook assesses the validity of our simulation pipeline by comparing the measured power spectrum with that from the T17 simulation [[2]](https://doi.org/10.3847%2F1538-4357%2Faa943d). It also discusses various correction factors applied to power spectrum modeling.
   
2. **Limber_Model_Without_Both.py:** Limber power spectrum without any correction factors.

3. **Limber_Model_Test_Without_Finite_Resolution.py:** Limber power spectrum without any correction factors.

4. **Limber_Model_Test_Without_Shellthickness.py:** Limber power spectrum with finite angular resolution of sky maps.
   
5. **Limber_Model_Test.py:** Limber power spectrum with both correction factors.

In the subdirectory, you can find the source code for the simulation pipeline used in my thesis:

6. **Prior_Training.ipynb:** This notebook samples a set of cosmological parameters from a range of priors using Latin Hypercube sampling [[8]](https://arxiv.org/abs/2106.03846). A total of 1,000 sets of cosmological parameters are sampled for training data sets.
   
7. **Prior_Validation.ipynb:** It is the same as Prior_Training.ipynb, but a total of 250 sets of cosmological parameters are sampled for validation data sets.
   
8. **Flask_sim_train.py:** This code covers steps (i) to (iv) in the structure of the pipeline. First, *class Cosmology_T17* loads constants, the redshift distribution, and cosmological parameters except $\sigma_8$ and $\Omega_m*. *class Kappa_Power_Spectrum_T17* calculates the convergence power spectrum. CLASS [[3]](https://arxiv.org/abs/1104.2932) is run in function LambdaCDM, then *function Cl* provides the convergence power spectrum from the given matter power spectrum. *class Generate_Flask_Maps* generates lognormal maps and Gaussian maps. *function log_normal_shift_parameter* is based on CosMomentum [[3]](https://doi.org/10.1093%2Fmnras%2Fstaa216) and calculates the shift parameter for given cosmological parameters. *function generate_log_normal_map* runs Flask [[6]](https://doi.org/10.1093%2Fmnras%2Fstw874) for the given convergence power spectrum from *class Kappa_Power_Spectrum_T17* and shift parameters from function log_normal_shift_parameter.
 
10. **Flask_sim_test.py:** This code is used for test data sets, similar to **Flask_sim_train.py**.

11. **Flask_sim_val.py:** This code is used for validation data sets, similar to **Flask_sim_train.py**.

12. **Flask_info.dat:** This data is updated throughout the simulation pipeline.

13. **Flask.config:** It contains various variables used for Flask in **Flask_sim_train.py**.

14. **Flask_val.config:** This is the configuration file for Flask used in validation data sets.

## External Links and Installation
(1) **Install HealPix**, You need HealPix (not healpy) to run Flask. You can install it from from [https://healpix.jpl.nasa.gov/html/install.htm](https://healpix.jpl.nasa.gov/html/install.htm)

(2) **Install Flask**, You can install Flask using this guide at [https://github.com/ucl-cosmoparticles/flask](https://github.com/ucl-cosmoparticles/flask). If you encounter any errors, you may need to modify the source code.

## References
[1] Hilbert S., Hartlap J., Scheider P., Cosmic shear covariance: the log-normal approximation, Astronomy and Astrophysics, 2011, [https://arxiv.org/abs/1105.3980](https://arxiv.org/abs/1105.3980) 

[2] Takahashi R., Hamana T., Shirasaki M., Namikawa T., Nishimichi T., Osato K., Shiroyama K., Full-sky Gravitational Lensing Simulation for Large-area Galaxy Surveys and Cosmic Microwave Background Experiments, The Astrophysical Journal, 2017, [https://doi.org/10.3847%2F1538-4357%2Faa943d](https://doi.org/10.3847%2F1538-4357%2Faa943d)

[3] Lesgourgues J., The Cosmic Linear Anisotropy Solving System (CLASS) I: Overview, arXiv e-prints, 2011, [https://arxiv.org/abs/1104.2932](https://arxiv.org/abs/1104.2932)

[4] Gong Z., Halder A., Barreira A., Seitz S., Friedrich O., Cosmology from the integrated shear 3-point correlation function:  simulated likelihood analyses with machine-learning emulators, Journal of Cosmology and Astroparticle Physics, 2023, [https://iopscience.iop.org/article/10.1088/1475-7516/2023/07/040](https://iopscience.iop.org/article/10.1088/1475-7516/2023/07/040)

[5] Friedrich O., Uhlemann C., Villaescusa-Navarro F., Baldauf T., Manera M., Nishimichi T., Primordial non-Gaussianity without tails - how to measure fNL with the bulk of the density PDF, Monthly Notices of the Royal Astronomical Society, 2020, [https://doi.org/10.1093%2Fmnras%2Fstaa216](https://doi.org/10.1093%2Fmnras%2Fstaa216)

[6]  Xavier H., Abdalla F., Joachimi B., Improving lognormal models for cosmological fields, Monthly Notices of the Royal Astronomical Society, 2016, [https://doi.org/10.1093%2Fmnras%2Fstw874](https://doi.org/10.1093%2Fmnras%2Fstw874)

[7] Halder A., Friedrich O., Seitz S., Varga T., The integrated three-point correlation function of cosmic shear, Monthly Notices of the Royal Astronomical Society, 2021, [https://doi.org/10.1093%2Fmnras%2Fstab1801](https://doi.org/10.1093%2Fmnras%2Fstab1801)

[8] Spurio Mancini A., Piras D., Alsing J., Joachimi B., Hobson M., CosmoPower: emulating cosmological power spectra for accelerated Bayesian inference from next-generation surveys, Monthly Notices of the Royal Astronomical Society, 2022, [https://arxiv.org/abs/2106.03846](https://arxiv.org/abs/2106.03846) 

