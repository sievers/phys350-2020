import numpy as np
from matplotlib import pyplot as plt
#example code to show spherical solution to separation of variables.
#we'll work out the potential induced by a sphere next to a point
#charge, and compare it to the method of images to show they agree



def legendre_mat(x,n):
    #you can also get this from np.polynomial.legendre.legvander
    #but this shows how easy the recurrence relation is to code
    mat=np.zeros([len(x),n+1])
    mat[:,0]=1.0
    if n==0:
        return mat
    mat[:,1]=x
    if n==1:
        return mat
    for i in range(n):
        mat[:,i+1]=((2*i+1)*x*mat[:,i]-i*mat[:,i-1])/(i+1)
    return mat


R=1.0
a=5.0
q=1.0
nterm=20

costh=np.linspace(-1,1,1001)
mat=legendre_mat(costh,nterm)


#we need the potential on the surface of the sphere
#by the law of cosines, d^2=a^2+R^2-2aRcos(theta)
#we'll let epsilon_0=1 for now
V=q/4/np.pi/np.sqrt(a**2+R**2-2*a*R*costh)
coeffs=np.linalg.inv(mat.T@mat)@(mat.T@V)
V_out=mat@coeffs

vec=np.linspace(-2*R,2*R,1000)
x,y=np.meshgrid(vec,vec)
r=np.sqrt(x**2+y**2)
th=np.arctan2(y,x)
rvec=np.ravel(r)
costhvec=np.ravel(np.cos(th))
mat2=legendre_mat(costhvec,len(coeffs-1))
pot=0
for l in range(len(coeffs)):
    pot=pot+mat2[:,l]*rvec**(-l-1)*coeffs[l]
pot[rvec<R]=0
pot=np.reshape(pot,x.shape)

#and let's compare to method of images solution
q_image=q*R/a
x_image=R**2/a
r2=np.sqrt((x-x_image)**2+y**2)
pot2=q_image/r2/4/np.pi
pot2[r<R]=0
print('mean error is ',np.mean(np.abs(pot2-pot)))
plt.ion()
