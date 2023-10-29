import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import numpy as np



@tf.function
def SWHD(model, coords, params, separate_terms=False):
    """ SWHD equations """

    g    = params[0]

    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch(coords) # coords t , x , y
        Yp = model(coords)[0]
        u  = Yp[:,0] # ux
        v  = Yp[:,1] # uy
        h  = Yp[:,2] # h
        hb = Yp[:,3] #

    # First derivatives
    grad_u = tape1.gradient(u, coords)
    u_t = grad_u[:,0]
    u_x = grad_u[:,1]
    u_y = grad_u[:,2]

    grad_v = tape1.gradient(v, coords)
    v_t = grad_v[:,0]
    v_x = grad_v[:,1]
    v_y = grad_v[:,2]

    grad_h = tape1.gradient(h, coords)
    h_t = grad_h[:,0]
    h_x = grad_h[:,1]
    h_y = grad_h[:,2]

    grad_hb = tape1.gradient(hb, coords)
    hb_t = grad_hb[:,0]
    hb_x = grad_hb[:,1]
    hb_y = grad_hb[:,2]

    del tape1

    # Equations to be enforced
    if not separate_terms:
        #f0 = u_x + v_y + h_z # no es incompresible
        f0 = (u_t + u*u_x + v*u_y  + g*h_x)
        f1 = (v_t + u*v_x + v*v_y  + g*h_y)
        f2 = (h_t + u*(h_x-hb_x) + u_x*(h-hb) + v*(h_y-hb_y) + v_y*(h-hb)   )

        return [f0, f1, f2]
