## Section 2: GCNN Inference
The complementary aspect of simulation is inference, where we aim to estimate the cosmological parameters from the density field. In my inference pipeline, I utilize the DeepSphere framework, which incorporates graph convolutional neural networks (GCNN) instead of summarizing data to a two-point function. This directory contains my code for the training steps of GCNN models and data preprocessing. The content in this subdirectory primarily aligns with Chapter 7 of my Master's thesis.

## Structure of Inference Pipeline

[<img src="GCNN_Model.jpg" width="700"/>](GCNN_Model.jpg)

The primary focus of this thesis is to investigate the validity of the Lognormal model approximation. As such, I trained GCNN networks using the Lognormal model, and additionally generated the Gaussian model for comparison. We can also assess the validity of our neural network model by comparing it with the Fisher contour.

(i) **Inputs:** The input data consists of maps that have been downgraded to NSIDE=128, along with cosmological parameters as labels. For the Gaussian model, I used Gaussian maps as training data, and for the Lognormal model, I used lognormal maps as training data. Apart from the input data, the model architecture and hyperparameters are the same for both models. 

(ii) **DeepSphere:** In contrast to CNN, GCNN can handle input data that is not defined in Euclidean space. DeepSphere is a GCNN package designed for spherical data, such as HEALPix maps. I employed layers from DeepSphere to process full-sky maps.

(iii) **Loss Function:** The loss function used is the negative Gaussian log-likelihood loss. This loss function provides estimates of parameters and the covariance matrix under the Gaussian assumption.

(iV) **Outouts:** The first two outputs are estimates of model parameters, specifically $\Omega_m$ and $\sigma_8$. The remaining three outputs are estimates of the covariance matrix.


## Contents

In this directory, you'll find the following code and Python notebooks:

1. **Downgrading_Validation.ipynb:** This notebook examines the effect of downgrading using KL divergence.
   
2. **Loss_comparison.ipynb:** This notebook visualizes the loss and metric curves during training steps. It includes the content of Appendix B.

In the subdirectory **Data_Preprocessing**, you can find the source code for the downgrading used in my thesis:

3. **Down_grade_Training.py:** Multiprocessing of *healpy.udgrad* for speeding up.
   
4. **Down_grade_Test.py:** Same as **Down_grade_Training.py**.
   
5. **Down_grade_Validation.py:** Same as **Down_grade_Training.py**.
 
6. **Get_Cl.py:** Multiprocessing of *healpy.anafast* for speeding up.

7. **Making_npz_file.py:** Creating a *.npz* file.

8. **Making_Test_npz_file.py:** Same as **Making_npz_file.py**.
   
In the subdirectory **GCNN_Models**, you can find the source code for the trained models used in my thesis:

9. **Gauss_model.ipynb:** Constructs Gaussian model 1 in my thesis.

10. **Gauss_model_2.ipynb:** Constructs Gaussian model 2 in my thesis.

11. **lognormal_model.ipynb.ipynb:** Constructs lognormal model 1 in my thesis.

12. **lognormal_model_2.ipynb.ipynb:** Constructs lognormal model 2 in my thesis.

13. **cov_by_hand.py:**  Calculates sample covariance and $\chi^2$. I don't remember why I didn't just use *numpy*.

14. **Loss_mean.py:** Calculates the loss function and metrics.

15. **standard_2D.py:** Stadidizing method for data preprocessing. I did not end up using it.

16. **__init__.py, build_model.py, gnn_layers_custom.py, gnn_transformers_custom.py, utils_custom.py:** Customized modules of *DeepSphere* used somewhere in the project.


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

