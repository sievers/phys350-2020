import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d


def facfac(n):
    #compute the double factorial, which is the product of all numbers of the same parity up to n
    if (n%2)==1:  #if n is odd this will be true
        vec=np.arange(1,n+1,2)
    else:
        vec=np.arange(2,n+1,2)
    return np.product(vec)

plt.ion()


def plm(x,l,m):
    #find general P_lm(x) by using the recurrence relation
    #(l-m+1)P_m,l+1 = (2l+1)x P_ml - (l+m) P_m,l-1
    #with P_mm=-1**l (2l-1)!! (1-x^2)**(l/2)
    #and P_m+1,m = x (2l+1)P_mm

    #p_cur=(-1)**m * facfac(2*m-1)*(1.0-x**2)**(0.5*m)
    p_cur=((-1)**m)*(1.0-x**2)**(0.5*m)
    if l==m:
        return p_cur

    p_old=p_cur
    p_cur=x*(2*m+1)*p_old
    if l==m+1:
        return p_cur
    for ll in range(m+1,l):
        p_next=((2*ll+1)*x*p_cur - (ll+m)*p_old)/(ll-m+1)
        p_old=p_cur
        p_cur=p_next
    return p_cur



th=np.linspace(0,np.pi,200)
phi=np.linspace(0,2*np.pi,400)

thmat,phimat=np.meshgrid(th,phi)
costh=np.cos(thmat)

l=15  #number of total wavelengths (=#of lobes/2)
m=15 #number of lobes in phi direction is 2m (one positive one negative)

rmat=plm(costh,l,m)
rmat=rmat*np.cos(m*phimat)  #make the executive decision to only use the cos term.
                            #in real life, for m>0, you also have sin

rmat=np.abs(rmat)  #the absolute value is for plotting purposes only.  Do not use for science!


z=rmat*costh
x=rmat*np.sin(thmat)*np.cos(phimat)
y=rmat*np.sin(thmat)*np.sin(phimat)

fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(1,1,1, projection='3d')
plot = ax.plot_surface(
    x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
    linewidth=0, antialiased=False, alpha=0.5)

