import numpy as np
from scipy.special import sph_harm

def cart2sph(x,y,z):
    r=np.sqrt(x**2+y**2+z**2)
    try:
        th=np.arccos(z/r)
        th[r==0]=0 #just make sure we don't get a nan if z=r=0
    except:
        pass
    try:
        phi=np.arctan2(y,x)
        phi[r==0]=0
    except:
        pass
    return r,th,phi

def get_alm(x,y,z,q,lmax):
    #find the coefficients for a multipole expansion
    r,th,phi=cart2sph(x,y,z)
    coeffs=np.zeros([lmax+1,lmax+1],dtype='complex')
    for l in range(lmax+1):
        for m in range(0,l+1):
            y_lm=sph_harm(m,l,phi,th)
            #find the coefficient.  Putting in a square root here
            #keeps dipole moment equal to sum of q times r
            coeffs[l,m]=np.sum(q*y_lm*r**l)*np.sqrt(4*np.pi/(2*l+1))
    return coeffs
def evaluate_pot(a_lm,x,y,z,lmax=-1):
    #evaluate the potential given a multipole expansion
    if lmax<0:
        lmax=np.max(a_lm.shape)-1
    r,th,phi=cart2sph(x,y,z)
    V=0
    for l in range(lmax+1):
        tot=0
        #loop over m first. 2.0-(m==0) gives you 1 if m=0 and 2 otherwise,
        #which comes from only using the non-negative m, so we have to count
        #the contribution from the positive m twice to make up for the 
        #missing negative m's.
        #the minus sign here is how the sign flip comes in
        #given the (somewhat non-standar) scipy convention
        for m in range(l+1):
            tot=tot+np.real(sph_harm(m,l,-phi,th)*coeffs[l,m])*(1.0+(m>0))
        #now add to potential with normalization and r scaling
        V=V+np.sqrt(4*np.pi/(2*l+1))*tot/r**(l+1)
    return V

#pick a few charge setups (possibly random) to demonstrate
if True:
    q=np.asarray([1,-1])
    z=np.asarray([1,-1])
    q=q*10
    z=z/10
    y=np.asarray([0,0])
    x=np.asarray([0,0])
else:
    q=np.asarray([1.0,-2,1.0])
    y=np.asarray([1,0,-1])
    x=np.asarray([0.5,0,-0.5])*0
    z=np.asarray([0,0,0])

if False:
    nn=10
    np.random.seed(1)
    q=np.random.randn(nn)
    q=q-q.mean() #if uncommented, this ensures the monopole is zeros
    x=2*np.random.rand(nn)-1
    y=2*np.random.rand(nn)-1
    z=2*np.random.rand(nn)-1
    if False:
        #we can manually zero the dipole here.  we'll do 
        #this by appending charges at (1,0,0), (0,1,0),(0,0,1), and then
        #a charge at (0,0,0) to cancel the added charge
        px=np.sum(x*q)
        py=np.sum(y*q)
        pz=np.sum(z*q)
        ptot=px+py+pz
        q=np.hstack([q,[-px,-py,-pz,ptot]])
        x=np.hstack([x,[1,0,0,0]])
        y=np.hstack([y,[0,1,0,0]])
        z=np.hstack([z,[0,0,1,0]])
    


#this gets our multipole expansion
lmax=5
coeffs=get_alm(x,y,z,q,lmax)
        
#pick a random point to evaluate the potential
x_targ=15.0
y_targ=12.0
z_targ=-33.5
#now get the potential at our point
#note - no matter how many charges we started with, 
#we only need to evaluate a set number of coefficients
#which can be much,much faster in the limit of lots of
#charges 
V_targ=evaluate_pot(coeffs,x_targ,y_targ,z_targ)

#check against the brute-force
dx=x_targ-x
dy=y_targ-y
dz=z_targ-z
dist=np.sqrt(dx**2+dy**2+dz**2)
V2=np.sum(q/dist)
print("Direct potential is ",V2," and potential via multipoles is ",V_targ)

