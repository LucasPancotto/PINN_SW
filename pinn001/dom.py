#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import copy
import itertools
import shutil            as sh
import numpy             as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

class Run:
    def __init__(self,odir='.',legacy=False,dest=None,lab=False,zpad=4,oall=False):
        if type(odir) is int:
            odir = str(odir).zfill(zpad)
            if lab:
                odir = 'L'+odir

        self.paths  = Run.Paths(odir,dest)
        paramfile = 'param.py'
        if legacy: paramfile = 'params.py'
        try:
            with open(self.paths.odir+paramfile) as f:
                for line in f:
                    if not line.startswith('#'):
                        exec('self.'+line)
            try:
                sh.copy(self.paths.odir+paramfile,self.paths.dest)
            except sh.SameFileError:
                pass
        except FileNotFoundError:
            print("param.py wasn't found")

        try:
            self.dt    /= self.mult
            self.step  *= self.mult
            self.tstep *= self.mult
            self.sstep *= self.mult
            self.cstep *= self.mult
        except AttributeError:
            pass

        try:
            self.k = np.arange(self.N/2+1)+1
            self.extent = (0,self.N/3+1,0,np.pi/(self.dt*self.sstep))
            self.dims = (0,2*np.pi,0,2*np.pi)
        except AttributeError:
            pass

        if oall:
            Run.openall(self)

    def add_params(self, paramfile):
        with open(paramfile) as f:
            for line in f:
                if not line.startswith('#'):
                    exec('self.'+line)

    class Paths:
        def __init__(self,odir,dest):
            if not odir.endswith('/'):
                odir += '/'
            self.odir = odir
            if dest is None:
                dest = odir
            if not dest.endswith('/'):
                dest += '/'
            self.dest = dest
            if (not os.access(self.odir,os.F_OK) or
                not os.access(self.dest,os.F_OK) ):
                print('No existe alguno de estos directorios: {} {}'.format(self.odir,self.dest))
                sys.exit()

    class Params:
        def __init__(self,paths):
            try:
                with open(paths.odir+'param.py') as f:
                    for line in f:
                        if not line.startswith('#'):
                            exec('self.'+line)
            except FileNotFoundError:
                print("param.py wasn't found")

    def open(self,File,*args,shape=None,**kwargs):
        '''Opens stuff
        For npy files, simply give the name of the file. You may choose a name

        For txt files, if they only have one column, the name of the file will
        suffice, otherwise you should pass the names as positional parameters
        (including Nones for undesired columns)'''
        if   os.path.splitext(File)[1] == '.npy':
            if not args:
                name = os.path.basename(os.path.splitext(File)[0])
            else:
                name = args[0]
            setattr(self,name,np.load(File)[:])
        elif os.path.splitext(File)[1] == '.txt':
            cols = np.loadtxt(File,unpack=True,**kwargs)
            if len(args) == 1:
                name = args[0]
                setattr(self,name,cols)
            else:
                names = tuple(args)
                for i,name in enumerate(names):
                    if name is not None:
                        setattr(self,name,cols[i])
        elif os.path.splitext(File)[1] == '.out':
            if shape is None:
                shape = (self.N,self.N)
            a = np.fromfile(File,dtype=self.dtype)
            a = a.reshape(shape,order='F')
            name = args[0]
            setattr(self,name,a)
        else:
            print('Unknown file extension')

    def openall(self):
        for f in glob.glob(self.paths.odir+'*.npy'):
            self.open(f)

    def setter(self,which,name,func,*args,**kwargs):
        attrs = copy.copy(self.__dict__)
        for key in attrs:
            if key.startswith(which):
                setting = func(getattr(self,key),*args,**kwargs)
                setattr(self,name+key[len(which):],setting)

def histog(*args,density=True,bins=100,**kwargs):
    '''Re-wrap de np.histogram
    Tiene como output a los puntos para graficar, en vez de los edges
    Density=True
    bins=100'''
    histo, edges = np.histogram(*args,density=density,bins=bins,**kwargs)
    points = [(edges[i] + edges[i+1])/2 for i in range(len(edges)-1)]
    points = np.array(points)
    return histo, points

def implot(arr,extent=None,origin='lower',**kwargs):
    '''Re-wrap para plt.imshow
    Grafica el array transpuesto y con el origin en lower, extent se puede
    pasar como positional'''
    im = plt.imshow(arr.T,extent=extent,origin=origin,**kwargs)
    return im

def contplot(arr,extent=None,origin='lower',**kwargs):
    '''Re-wrap para plt.contourf
    Grafica el array transpuesto y extent se puede pasar como positional'''
    plt.contourf(arr.T,extent=extent,origin=origin,**kwargs)

def txtload(File,**kwargs):
    '''Re-wrap para np.loadtxt
    Tiene unpack=True'''
    return np.loadtxt(File,unpack=True,**kwargs)

def fplot(File,*args,log=False,**kwargs):
    '''Re-wrap para plt.plot y np.loadtxt
    Las columnas se pasan como parámetros posicionales'''
    if len(args) == 0:
        cols = None
    else:
        cols = tuple(args)
    cols = np.loadtxt(File,usecols=cols,**kwargs)
    if cols.ndim > 1:
        for i in range(cols[:,1:].shape[1]):
            plt.plot(cols[:,0],cols[:,i+1])
    else:
        plt.plot(cols)
    if log:
        plt.xscale('log')
        plt.yscale('log')

def rmsvalue(a):
    '''Valor rms del array (suma sobre todas las dimensiones)'''
    return np.sqrt(np.sum(a**2)/a.size)

def abrirbin(f,shape,dim=2,apad=(None,None),order='F',dtype=np.float32,tocomplex=False):
    '''Abre binarios, ordenanadolos correctamente'''
    a = np.fromfile(f,dtype=dtype)[apad[0]:apad[1]]
    if type(shape) is not tuple:
        shape = tuple([shape]*dim)
    if tocomplex:
        a = a[::2]+1.0j*a[1::2]
    return a.reshape(shape,order=order)

def runs(*args):
    '''Enlista runs'''
    return [str(i).zfill(4) for i in args]

def idx_nearest(a,val):
    '''Devuelve idx tal que a[idx] es el valor de a mas cercano a val'''
    return np.abs(a-val).argmin()

def deriv1d(a,direc='y'):
    '''Deriva un array en una direccion espectralmente.
    Los resultados siempre quedan en la dirección 'y' del output
    No esta probado para derivadas en direccion z'''
    if   direc == 0:
        direc = 'x'
    elif direc == 1:
        direc = 'y'

    b = np.zeros(np.shape(a))
    if direc == 'x':
        a = a.transpose()

    c = np.fft.rfft(a)
    k = complex(0,1)*np.arange(np.shape(c)[-1])
    b = np.fft.irfft(k*c)
    if direc == 'x': b = b.T
    return b

def clearfigs():
    for fig in plt.get_fignums():
        plt.figure(fig)
        plt.clf()

def drawandshow():
    for fig in plt.get_fignums():
        plt.figure(fig)
        plt.draw()
    plt.show()

def read_spettro(run,which=1,nudging=False,perp=False,para=False):
    # Reads spectra from Complete.
    # By default reads total kinetic energy.
    cols = 6
    File = 'spettro'
    if nudging:
        cols  = 3
        File  = 'spettro_ndg'
    elif perp:
        cols  = 4
        File  = 'spettro_perp'
    elif para:
        cols  = 3
        File  = 'spettro_par'
    spec = np.loadtxt(run.paths.dest+'/{}.dat'.format(File),unpack=True)
    ltot = spec.shape[-1]
    spec = spec.reshape((cols,int(2*ltot/run.N),int(run.N/2)))
    kk   = spec[0,0,:]
    ts   = spec[-1,:,0]
    sp   = spec[which,:,:]
    return kk, ts, sp

def read_flusso(run,nudging=False,ourot=False,flux_file='/flusso.dat'):
    # Reads fluxes from Complete.
    spec = np.loadtxt(run.paths.dest+flux_file,unpack=True)
    ltot = spec.shape[-1]
    spec = spec.reshape((5,int(2*ltot/run.N),int(run.N/2)))
    kk   = spec[0,0,:]
    tk   = spec[1,:,:]
    hk   = spec[2,:,:]
    ts   = spec[3,:,0]
    zk   = spec[4,:,:]
    if nudging:
        spec = np.loadtxt(run.paths.dest+'/flusso_ndg.dat',unpack=True)
        ltot = spec.shape[-1]
        spec = spec.reshape((5,int(2*ltot/run.N),int(run.N/2)))
        hk   = spec[1,:,:] - tk
    if ourot:
        spec = np.loadtxt(run.paths.dest+'/flusso_ou.dat',unpack=True)
        ltot = spec.shape[-1]
        spec = spec.reshape((5,int(2*ltot/run.N),int(run.N/2)))
        hk   = spec[1,:,:] - tk
    return kk, tk, hk, ts, zk

dash_lines = [[5,5],[5,3,1,3],[1,3],(None,None),[5,2,5,2,5,10]]
