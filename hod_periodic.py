import numpy as np
import scipy as sp
import emcee
import time
from Corrfunc.theory.wp import wp
from multiprocessing import Pool

st=time.time()
boxsize = 1000.0
nhalos = 10446515
nsates = 380625
nthreads = 1
pimax = 100.0
nrpbins = 8
bins = np.logspace(-1.2, 3.4, nrpbins + 1, base=2.71828) #equally spaced in ln R between 0.3 Mpc/h and 30 Mpc/h.

halofile=open('/mnt/Data/Multidark/mdhalo_z038_1000.dat','r') #1000^3 subsample in the center of Multidark
halolines=np.loadtxt(halofile,usecols=(0,3,4,5)) 
satefile1=open('/mnt/Data/Multidark/sate/sate_ind_1000.dat','r') #pre build satellite catalog for the in phase purpose
sateind=np.loadtxt(satefile1,usecols=(1,3))
satefile2=open('/mnt/Data/Multidark/sate/sate_pos_1000.dat','r')
satepos=np.loadtxt(satefile2,usecols=(1,2,3))
wpdata=open('/mnt/Data/Boss/counts/wp/boss_wp_lnmw.dat','r') #wp measurement from boss
wp_obs=np.loadtxt(wpdata)
crosscov=open('/mnt/Data/Patchy/NGC/count/wp/cov/wp_cov_lnmw.dat','r') #cov-matrix
cov=np.loadtxt(crosscov)
invcov=np.linalg.inv(cov)
bk1=time.time()
print('data input completed, took {0:.1f} seconds'.format(bk1-st))

#print(wp_obs)
#print(cov)
#print(invcov)
np.random.seed(123)
random_u=np.random.rand(nhalos)


def poisson_inphase(mean,u):
    k=0
    s=np.exp(-mean)
    p=np.exp(-mean)
    while u>s:
        k=k+1
        p=p*mean/k
        s=s+p
    return k

def log_prior(theta):
    m_cut,m1,sigma,kappa,alpha = theta
    if 12.0 < m_cut < 14.0 and 13.0 < m1 < 15.0 and 0.0 < sigma < 2.0 and 0.0 < kappa < 3 and 0.0 < alpha < 2:
        return 0.0
    return -np.inf
     


def log_probability(theta): #main function to calculate loglike
    bk2=time.time()
    m_cut,m1,sigma,kappa,alpha=theta
    count_cent=0
    count_sate=0
    count_total=0
    x=[]
    y=[]
    z=[]
    for i in range(nhalos):
        np.random.seed(456)
        mass=halolines[i][0]
        hx=halolines[i][1]
        hy=halolines[i][2]
        hz=halolines[i][3]
        num_sat=int(sateind[i][0])
        ind_sat=int(sateind[i][1])
        mass_cut=10.0**m_cut
        mass1=10.0**m1
        N_cent = 1.0/2.0*sp.special.erfc(np.log(mass_cut/mass)/(np.sqrt(2)*sigma))
        if(np.random.rand() < N_cent):
            count_total+=1
            count_cent+=1
            x.append(hx)
            y.append(hy)
            z.append(hz)
            if(mass < kappa*mass_cut):
                tmp_nsate=0
            else:
                N_sat = ((mass-kappa*mass_cut)/mass1)**alpha
                tmp_nsate=poisson_inphase(N_sat,random_u[i])
            if(tmp_nsate > num_sat):
                tmp_nsate=int(num_sat) 
            for j in range(tmp_nsate): 
                count_sate+=1
                count_total+=1
                x.append(satepos[int(ind_sat)+j-1][0])
                y.append(satepos[int(ind_sat)+j-1][1])
                z.append(satepos[int(ind_sat)+j-1][2])
    bk3=time.time()
    print('hod script completed, took {0:.1f} secs'.format(bk3-bk2))
    wp_counts = wp(boxsize,pimax,nthreads,bins,x,y,z) #corrfunc lib to compute wp
    bk4=time.time() 
    print('wp calculation completed, took {0:.1f} secs'.format(bk4-bk3))
    print(wp_counts)
    log_like = 0.0    
    for i in range(8):
        for j in range(8):
            log_like = log_like+(wp_counts[i][3]-wp_obs[i])*invcov[i][j]*(wp_counts[j][3]-wp_obs[j])
    print(0.5*log_like)
    lp = log_prior(theta)
    bk5=time.time()
    print('likelihood calculation completed, took {0:.1f} secs'.format(bk5-bk4))
    if not np.isfinite(lp):
        return -np.inf
    return lp+-0.5*log_like
   
with Pool() as pool: #parallel mcmc
    
    initial = np.array([13.08,14.06,0.98,1.13,0.9])
    ndim = len(initial)
    nwalkers = 16
    p0 = [np.array(initial)+0.1*np.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers,ndim,log_probability,pool=pool)
    state=sampler.run_mcmc(p0,1000)

