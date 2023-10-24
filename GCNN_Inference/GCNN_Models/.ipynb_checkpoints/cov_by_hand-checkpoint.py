import numpy as np

def cov(theta, mu):
    diff = theta-mu
    cov = 1./(len(diff)-1.) * np.matmul(diff.T, diff)
    return cov

def chisq(cov, mu_theta, theta):
    cov_inv = np.linalg.inv(cov)
    diff = np.abs(mu_theta- theta)
    mat = np.matmul(diff, cov_inv)
    chisq= np.matmul(mat, diff.T)
    return chisq