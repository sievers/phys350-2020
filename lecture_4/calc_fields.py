import numpy as np
from matplotlib import pyplot as plt

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
print('total charge is ',np.sum(q))

r=[0,0,1000]
E=calc_field(r,x,y,z,q)
print('electric field is ',E)

