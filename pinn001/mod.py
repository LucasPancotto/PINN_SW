
from dom import *
import numpy as np

def generate_data(params, path):
    try:
        coords = np.load('data/coords.npy')
        fields = np.load('data/fields.npy')
        flags  = np.load('data/flags.npy')
    except:
        xi = np.linspace(0,1,num=params.N,endpoint=False) # N = 1024
        xx, yy = np.meshgrid(xi, xi, indexing='ij')
        xx = xx.flatten()
        yy = yy.flatten()
        # zz = zz.flatten() es 2D
        y_flags = (yy*params.N).astype(int)

        coords = []
        fields = []
        flags  = []

        # Load files
        for ii in range(params.tsteps):
            vv = np.array([abrirbin(f'{path}/{comp}.{ii+1:03}.out', params.N).flatten() # abrirbin(f,shape,dim=3,apad=(None,None),order='F',dtype=np.float32,tocomplex=False) in dom.py
                           for comp in ['vx', 'vy', 'th']])
            tt = np.ones(len(xx))*ii*params.dt

            for jj in range(len(xx)):
                if np.random.rand()>params.sample_prob:
                    coords.append([tt[jj], xx[jj], yy[jj]])
                    fields.append([vv[0][jj], vv[1][jj], vv[2][jj], vv[2][jj]]) # repeat vv[2][jj] so as to pad... we do not have hb filed
                    flags.append(y_flags[jj])

        coords = np.array(coords).astype(np.float32)
        fields = np.array(fields).astype(np.float32)
        flags  = np.array(flags)

        np.save('data/coords', coords)
        np.save('data/fields', fields)
        np.save('data/flags', flags)

    return coords, fields, flags

def plot_points(params, tidx):
    '''Creates a grid that matches the one in simulations'''
    N = params.N

    T = tidx*params.dt # a particular time in the simulation
    X = np.linspace(0,1,endpoint=False,num=params.N)
    Y = np.linspace(0,1,endpoint=False,num=params.N)

    T, X, Y= np.meshgrid(T, X, Y, indexing='ij')

    T = T.reshape(-1,1)
    X = X.reshape(-1,1)
    Y = Y.reshape(-1,1)

    X = np.concatenate((T, X, Y), 1)
    return X.astype(np.float32)

def cte_validation(self, params, path, tidx):
    def validation(ep):
        N = params.N

        # Get predicted
        X_plot = plot_points(params, tidx=tidx)
        Y  = self.model(X_plot)[0].numpy() # model evaluated in all X,Y, T coordenates of data

        u_p = Y[:,0].reshape((N,N))
        v_p = Y[:,1].reshape((N,N))
        th_p = Y[:,2].reshape((N,N))
        hb_p = Y[:,3].reshape((N,N))

        np.save("predicted.npy",[u_p, v_p, th_p, hb_p]) # Predicted fields in predicted.npy
        pinn = np.array([u_p, v_p, th_p]) # do not include hb_p here, because it is used for validating against simulation
        ref = np.array([abrirbin(f'{path}/{comp}.{tidx+1:03}.out', params.N)
                       for comp in ['vx', 'vy', 'th']])
        err  = [np.sqrt(np.mean((ref[ff]-pinn[ff])**2))/np.std(ref[ff]) for ff in range(3)]

        # Loss functions
        output_file = open(self.dest + 'validation.dat', 'a')
        print(ep, *err,
              file=output_file)
        output_file.close() # print validation against simulation in validation.dat

    return validation # returns the function validation(ep)

def get_predictions(params, tidxs]):
    N = params.N
    for tidx in tidxs:

        # Get predicted
        X_plot = plot_points(params, tidx=tidx)
        Y  = self.model(X_plot)[0].numpy() # model evaluated in all X,Y, at particular T=tidx*dt coordenates of data

        u_p = Y[:,0].reshape((N,N))
        v_p = Y[:,1].reshape((N,N))
        th_p = Y[:,2].reshape((N,N))
        hb_p = Y[:,3].reshape((N,N))

        np.save(f"predicted_T{tidx*params.dt}.npy",[u_p, v_p, th_p, hb_p]) # Predicted fields in predicted.npy
