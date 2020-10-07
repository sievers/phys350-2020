import numpy as np
from scipy.special import sph_harm

if True:
    q=np.asarray([1,-1])
    #z=np.asarray([1,-1])
    #y=np.asarray([0,0])
    #x=np.asarray([0,0])
    #y=x.copy()
    x=np.random.randn(2)
    y=np.random.randn(2)
    z=np.random.randn(2)
    l_use=1

else:
    q=np.asarray([1.0,-2,1.0])
    y=np.asarray([1,0,-1])
    x=np.asarray([0.5,0,-0.5])*0
    z=np.asarray([0,0,0])
    l_use=2

if True:
    nn=10
    q=np.random.randn(nn)
    q=q-q.mean()
    x=2*(np.random.rand(nn))-1
    y=2*(np.random.rand(nn))-1
    z=2*np.random.rand(nn)-1


r=np.sqrt(x**2+y**2+z**2)
th=np.arccos(z/r)
th[r==0]=0
phi=np.arctan2(-y,x)
phi[r==0]=0


xx=r*np.sin(th)*np.cos(phi)
yy=r*np.sin(th)*np.sin(phi)
zz=r*np.cos(th)

lmax=5
coeffs=np.zeros([lmax+1,3*(lmax+2)],dtype='complex')
for l in range(lmax+1):
    if l==0:
        vec=1
    else:
        vec=r**l
    for m in range(0,l+1):
        y_lm=sph_harm(m,l,phi,th)#*(-1**m)
        coeffs[l,m]=np.sum(q*y_lm*vec)
        if (l==1):
            print(l,m,coeffs[l,m])

x_targ=15.0*4
y_targ=12.0*3
z_targ=-33.5*2
r_targ=np.sqrt(x_targ**2+y_targ**2+z_targ**2)
th_targ=np.arccos(z_targ/r_targ)
phi_targ=np.arctan2(y_targ,x_targ)
V_targ=0
for l in range(lmax+1):
    m=0
    #print(sph_harm(m,l,phi_targ,th_targ))

    #tmp=np.real(coeffs[l,0]*sph_harm(0,l,phi_targ,th_targ))
    tmp=0
    for m in range(0,l+1):
        tmp=tmp+(2.0-(m==0))*np.real(coeffs[l,m]*sph_harm(m,l,phi_targ,th_targ))#*(-1**m))
    V_targ=V_targ+tmp/r_targ**(l+1)*(4*np.pi/(2*l+1))

    #
    #V_targ=V_targ+coeffs[l,0]*r_targ**(-(l+1))#
    #
    #    V_targ=V_targ+2*np.real(coeffs[l,m])*(r_targ**(-(l+1)))
    #fac=np.sqrt(4*np.pi/

dx=x_targ-x
dy=y_targ-y
dz=z_targ-z
dist=np.sqrt(dx**2+dy**2+dz**2)
V2=np.sum(q/dist)
print("potentials are ",V2,V_targ)


#print(np.std(yy-y))

