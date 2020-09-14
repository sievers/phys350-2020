import numpy as np
from matplotlib import pyplot as plt
from numba import njit
import numba as nb


@njit
def calc_field_nb_inner(r,x,y,z,q,E):
    for i in range(len(x)):
        dx=r[0]-x[i]
        dy=r[1]-y[i]
        dz=r[2]-z[i]
        dsqr=dx**2+dy**2+dz**2
        d=np.sqrt(dsqr)
        qd3=q[i]/(dsqr*d)
        E[0]=E[0]+dx*qd3
        E[1]=E[1]+dy*qd3
        E[2]=E[2]+dz*qd3

@njit
def calc_field_nb_loop_inner(r,x,y,z,q,E):
    for j in nb.prange(r.shape[0]):
        for i in range(len(x)):
            dx=r[j,0]-x[i]
            dy=r[j,1]-y[i]
            dz=r[j,2]-z[i]
            dsqr=dx**2+dy**2+dz**2
            d=np.sqrt(dsqr)
            qd3=q[i]/(dsqr*d)
            E[j,0]=E[j,0]+dx*qd3
            E[j,1]=E[j,1]+dy*qd3
            E[j,2]=E[j,2]+dz*qd3

def calc_field_nb_loop(r,x,y,z,q):
    E=np.zeros(r.shape)
    calc_field_nb_loop_inner(r,x,y,z,q,E)
    return E
    

def calc_field_nb(r,x,y,z,q):
    E=np.zeros(3)
    calc_field_nb_inner(r,x,y,z,q,E)
    return E

def calc_field(r,x,y,z,q):
    #calculate the electric field at position r
    #for simplicity ignore 4 pi eps_0
    dx=r[0]-x
    dy=r[1]-y
    dz=r[2]-z
    d=np.sqrt(dx**2+dy**2+dz**2)
    #field is r^ / |r^2| or r /|r^3|
    Ex=np.sum(dx/d**3*q)
    Ey=np.sum(dy/d**3*q)
    Ez=np.sum(dz/d**3*q)
    return np.asarray([Ex,Ey,Ez])




vec=np.linspace(-1,1,101) #gives 101 points spaced between -1 and 1
x_grid,y_grid=np.meshgrid(vec,vec)
x=np.ravel(x_grid)
y=np.ravel(y_grid)
z=0*x


Q=4
q=np.ones(len(x))*Q/len(x)

x=np.hstack([x,x])
y=np.hstack([y,y])
z=np.hstack([z,0.5+z])
q=np.hstack([q,-q])

print('total charge is ',np.sum(q))



r=np.asarray([0,0,0.2])
E=calc_field(r,x,y,z,q)
print('electric field is ',E)

vec=np.linspace(-2,2,20)
xvec,zvec=np.meshgrid(vec,vec)
rsqr=xvec**2+zvec**2
xx=np.ravel(xvec)
zz=np.ravel(zvec)
yy=np.zeros(len(xx))

rr=np.vstack([xx,yy,zz]).T
E=calc_field_nb_loop(rr,x,y,z,q)
Ex=np.reshape(E[:,0],xvec.shape)
Ey=np.reshape(E[:,1],xvec.shape)
Ez=np.reshape(E[:,2],xvec.shape)


