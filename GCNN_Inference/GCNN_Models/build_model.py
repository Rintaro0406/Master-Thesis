import healpy as heal
import numpy as np
import tensorflow as tf
from healpy_networks_custom import HealpyGCNN
import healpy_layers_custom as hp_layer
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp


def build_model(hp):
    nside=128
    npix = heal.nside2npix(nside)
    indices = np.arange(npix)
    layers =[hp_layer.HealpyChebyshev(K=5, Fout=5, use_bias=False, use_bn=True,          #0
                                               activation="relu", kernel_regularizer=None),
                      hp_layer.HealpyPool(p=1),                                          #1
                      hp_layer.HealpyChebyshev(K=5, Fout=5, use_bias=False, use_bn=True, #2
                                           activation="relu", kernel_regularizer=None),
                      hp_layer.HealpyPool(p=1),                                          #3
                      hp_layer.HealpyChebyshev(K=5, Fout=5, use_bias=False, use_bn=True, #4
                                           activation="relu", kernel_regularizer=None),
                      hp_layer.HealpyPool(p=1),                                          #5
                      hp_layer.HealpyChebyshev(K=5, Fout=5, use_bias=False, use_bn=True, #6
                                           activation="relu", kernel_regularizer=None),
                      hp_layer.HealpyPool(p=1),                                          #7
                      hp_layer.HealpyChebyshev(K=5, Fout=5, use_bias=False, use_bn=True, #8
                                           activation="relu", kernel_regularizer=None),
                      hp_layer.HealpyPool(p=1),                                          #9   
                      hp_layer.HealpyChebyshev(K=5, Fout=5, use_bias=False, use_bn=True, #10
                                           activation="relu", kernel_regularizer=None),
                      hp_layer.HealpyPool(p=1),                                          #11   
                      hp_layer.HealpyChebyshev(K=5, Fout=5, use_bias=False, use_bn=True, #12
                                           activation="relu", kernel_regularizer=None),  #13
                      tf.keras.layers.Flatten(),                                         #14
                      tf.keras.layers.Dense(5,use_bias=False)]                           #15
    model = HealpyGCNN(nside=nside, indices=indices, layers=layers, n_neighbors=20)      
    model.build(input_shape=(None, len(indices), 1)) 
    tf.keras.backend.clear_session()
    return model