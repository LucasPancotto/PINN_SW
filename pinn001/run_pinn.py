# State preparation

import tensorflow as tf
from   tensorflow import keras
tf.keras.backend.set_floatx('float32')

from pinn      import PhysicsInformedNN

from   dom import *
from   mod import *
import numpy as np
import time
import random



# Get parameters
params = Run()

# NN params
layers = [3]+[params.hu]*params.layers+[4] # coords : t,x,y , and output u , v , h , hb

# Load data
X_data, Y_data, flags = generate_data(params, '../gauss_topo/gauss_topo011/outs')

# Normalization layer
inorm = [X_data.min(0), X_data.max(0)]
means     = Y_data.mean(0)
means[-1] = params.P # guess for the mean of distribution of output layer transformation
stds      = Y_data.std(0)
stds[-1]  = params.sig_p # guess for the std of distribution of output layer transformation
onorm = [means, stds]

# Optimizer scheduler
if params.depochs:
    dsteps = params.depochs*len(X_data)/params.mbsize
    params.lr = keras.optimizers.schedules.ExponentialDecay(params.lr,
                                                            dsteps,
                                                            params.drate)
# Initialize model
from equations import SWHD as Eqs
eq_params = ([np.float32(params.g)])
eq_params = [np.float32(p) for p in eq_params]
PINN = PhysicsInformedNN(layers,
                         dest=params.paths.dest,
                         norm_in=inorm,
                         norm_out=onorm,
                         optimizer=keras.optimizers.Adam(learning_rate=params.lr),
                         eq_params=eq_params )
PINN.optimizer.learning_rate.assign(params.lr)

# Validation function
PINN.validation = cte_validation(PINN, params, '../gauss_topo/gauss_topo011/outs', 5)

# Train
# Ir cambiando el train data
PINN.train(X_data, Y_data,
           Eqs,
           epochs=params.epochs,
           batch_size=params.mbsize,
           #alpha=0.1,
           flags=flags,
           lambda_data=1.0,
           lambda_phys=0.0,
           print_freq=10,
           valid_freq=10,
           save_freq=10,data_mask=(True,True,True,False))
