import numpy as np
from matplotlib import pyplot as plt
plt.ion()
#let's have a setup where we have a conductor held at
#fixed potential between two grounded plates, but we cut
#cut a hole in the conductor.
#for simplicity, we'll assume the plate goes from
#-a/2 to a/2, and the hole goes from -b/2 to b/2

#for symmetry reasons, we can write the potential as
#sum cos(n pi x/a)exp(+/- n pi z/a) where
#the exponential must be negative for z>0
#and positive for z<0, so V goes to zero at +/- infinity
#further, we know that E must be continuous across
#the boundary in the gap region, because sigma is zero.
#in this particular case, symmetry means that Ez must be zero
#though in general, one would need to match derivatives.

#This gives:  sum(c_n cos(n pi x/a)) = V0 for b/2<x<a/2
#             Ez=dV/dz=+/-cos(n pi x/a)n pi/a = 0 for x<b/2
#We can write this down as a matrix equation and do
#the least-squares solution for the coefficients


a=4
b=1
V0=2
nx=10001
x=np.linspace(-a/2,a/2,nx)
x1=x[np.abs(x)>=b/2]
x2=x[np.abs(x)<b/2]
nterm=1000
mat=np.zeros([nx,nterm])
#first, do the standard, hold-free setup
for i in range(nterm):
    n=2*i+1 #we want to start with the first term, as the 
            #zero term doesn't go to zero on the edges, plus by symmetry
            #we only have odd terms
    mat[:,i]=np.cos(x*n*np.pi/a)
rhs=np.zeros(nx)+V0
fitp=np.linalg.inv(mat.T@mat)@(mat.T@rhs)
pred=mat@fitp

#now implement the holes.  We know 
#we have sum  c_n n pi /a cos(n pi a x) is zero, so 
#we need to rescale the coefficients in the gap by
#n pi /a, and set the rhs to be zero for those
mat2=mat.copy()
ind=np.abs(x)<b/2
for i in range(nterm):
    n=2*i+1
    mat2[ind,i]=mat2[ind,i]*n*np.pi/a 
rhs2=rhs.copy()
rhs2[ind]=0
fitp2=np.linalg.inv(mat2.T@mat2)@(mat2.T@rhs2)
pred2=mat@fitp2 #this gives us the potential along the y axis

#now let's look at the electric field along the plate.
#we can do this by evaluating dV/dz at z=0.  We can do this by 
#applying the thing we've already done to the full matrix

mat3=mat.copy()
for i in range(nterm):
    n=2*i+1
    mat3[:,i]=mat3[:,i]*n*np.pi/a

Ez=mat3@fitp2

