import numpy as np
from matplotlib import pyplot as plt

a=4
V0=3
y=np.linspace(0,a,1001)
x=2*y

pot=0*y
nmax=320
potmat=0
for n in range(1,nmax,2):
    cn=4/n/np.pi*V0 #the c_n we calculated just now
    yvec=np.sin(n*np.pi*y/a)
    pot=pot+cn*yvec #we know potential is the cn*sin()
    xvec=np.exp(-n*x/a)
    potmat=potmat+cn*np.outer(yvec,xvec)
plt.clf()
plt.imshow(potmat);
plt.colorbar()
