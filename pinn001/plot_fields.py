# python3

from dom import *
#from pylab import *
import matplotlib.pyplot as plt
import numpy as np

#pred = np.txtload('predicted.npy')
#vx = abrirbin('../hit_base/odir/vx.0006.out', 32)

ep, lu, lf = txtload('output.dat') # it is np.loadtxt in dom.py
plt.figure()
plt.semilogy(ep, lu)
plt.semilogy(ep, lf)

# ep, vax, vay, vaz = txtload('validation.dat')
#plt.semilogy(ep, vax)

# figure()
# implot(pred[0][0,:,:])
# title('Prediction')
# colorbar()
#
# figure()
# implot(vx[0,:,:])
# title('Truth')
# colorbar()
#
# figure()
# implot(abs(pred[0][0,:,:]-vx[0,:,:]))
# title('dif')
# colorbar()

#figure()
#plot(vx[16,16,:])
#plot(pred[0][16,16,:], '--')

#drawandshow()
