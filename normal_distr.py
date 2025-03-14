# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize
import casadi as ca
import scienceplots

# %% Flag for CSD Calculation after Jiang et al. 2016
Jiang_CSD_metric=True
# %%
sigma1= 1
sigma2= 2

mu1= 0
mu2= 10

x_max=(sigma1**2*mu2-sigma2**2*mu1)/(sigma1**2-sigma2**2)
x= np.linspace(min([mu1,mu2])-10*max([sigma1,sigma2]),max([mu1,mu2])+10*max([sigma1,sigma2]),1000)

y1= stats.norm.pdf(x,mu1,sigma1)
y2= stats.norm.pdf(x,mu2,sigma2)
# %%
fig, ax= plt.subplots(3,1)
#FIrst plot is two densities
ax[0].plot(x,y1)
ax[0].plot(x,y2)
ax[0].axvline(x=x_max, color='r', linestyle='--')
ax[0].set_ylabel('Density')
#ax[0].legend(['$\mu=0, \sigma=1$','$\mu=2, \sigma=2$'])

# Second plot is the quotient of y1 / y2
ax[1].plot(x,y1/y2)
ax[1].axvline(x=x_max, color='r', linestyle='--')
ax[1].set_ylabel('y1/y2')
# Third plot is the quotient of log (y1/y2)
ax[2].plot(x,np.log(y1/y2))
ax[2].set_ylabel('log y1/y2')

# %% Now we want to calculate the different metrices for two different normal distributions
µ1= 0
µ2 =0
sig1= 1
sig2= 2 # sig2 is greater than sig 1
eps=0.2
x= np.linspace(min([µ1,µ2])-10*max([sig1,sig2]),max([µ1,µ2])+10*max([sig1,sig2]),1000000)

y1= stats.norm.pdf(x,µ1,sig1)
y2= stats.norm.pdf(x,µ2,sig2)
# First we calculate the KL divergence / relative entropy
KL= 1/2*((sig1/sig2)**2+(µ2-µ1)**2/sig2**2-1+np.log(sig2**2/sig1**2))
#KL = 1/2*((sig2/sig1)**2+(µ1-µ2)**2/sig1**2-1+np.log(sig1**2/sig2**2))
print('KL divergence is: ',KL)
opt_func = lambda  lam,KL_div, eps_val : -(np.exp(-KL_div)*(lam+1)**eps_val-1)/(lam)
_,corr_eps_KL,_,_= scipy.optimize.fminbound(lambda lam: opt_func(lam,KL,eps),1e-6,1e6,full_output=True)
corr_eps_KL=corr_eps_KL*-1
print('Unambiguous eps for KL is: ',corr_eps_KL)
# Now we calculate the relative variation distance
RVD = sig2/sig1*np.exp(0.5*((µ2-µ1)**2/(sig2**2-sig1**2)))
corr_eps_RVD = eps/RVD
print('Relative variation distance is: ',RVD)
print('Unambiguous eps for RVD is: ',corr_eps_RVD)
# Now we calculate the Hellinger distance
HD = np.sqrt(1-np.sqrt((2*sig1*sig2)/(sig1**2+sig2**2))*np.exp(-0.25*(((µ2-µ1)**2)/(sig1**2+sig2**2))))
HD_numerical = np.sqrt(0.5*np.trapz((np.sqrt(y1)-np.sqrt(y2))**2,x))
HD_norm= np.sqrt(np.sqrt(0.5*np.trapz((np.sqrt(y1/y2)-np.sqrt(y2/y2))**2,x)))
print('Hellinger distance is: ',HD)
print('Hellinger distance numerical is: ',HD_numerical)
print('Hellinger distance normalized is: ',HD_norm)
corr_eps_HD = np.maximum(np.sqrt(eps)-HD,0)**2
print('Unambiguous eps for HD is: ',corr_eps_HD)
# Now we calculate chi_square distance
CSD=np.trapz(((y1-y2)**2)/(y2),x)
CSD_norm= np.trapz(((y1-y2)**2)/(y2*y2),x)

#CSD= np.trapz(((y2-y1)**2)/(y1),x)
print('Chi square distance is: ',CSD)
print('Chi square distance normalized is: ',CSD_norm)
if Jiang_CSD_metric:
    corr_eps_CSD = eps- (np.sqrt(CSD**2+4*CSD*(eps-eps**2))-(1-2*eps)*CSD)/(2*CSD+2)
else:
    corr_eps_CSD = eps+ CSD/2-np.sqrt(eps*CSD+CSD**2/4)

print('Unambiguous eps for CSD is: ',corr_eps_CSD)
# Now we calculate the total variation distance
TVD=np.trapz(np.abs((y1-y2)),x)/2
TVD_norm=np.trapz(np.abs((y1-y2)/y1),x)/2
print('Total variation distance is: ',TVD)
print('Total variation distance normalized is: ',TVD_norm)
corr_eps_TVD = eps-TVD
print('Unambiguous eps for TVD is: ',corr_eps_TVD)

# %%
# Now do the above calulations for a range of epsilons and a range of sig2
eps_grid=np.linspace(0,1,101)
sig2_grid=np.linspace(1,10,101)
eps_KL=np.zeros((len(sig2_grid),len(eps_grid)))
eps_RVD=np.zeros((len(sig2_grid),len(eps_grid)))
eps_HD=np.zeros((len(sig2_grid),len(eps_grid)))
eps_CSD=np.zeros((len(sig2_grid),len(eps_grid)))
eps_TVD=np.zeros((len(sig2_grid),len(eps_grid)))

for i in range(len(sig2_grid)):
    for j in range(len(eps_grid)):
        x= np.linspace(min([µ1,µ2])-10*max([sig1,sig2_grid[i]]),max([µ1,µ2])+10*max([sig1,sig2_grid[i]]),1000)
        y1= stats.norm.pdf(x,µ1,sig1)
        y2= stats.norm.pdf(x,µ2,sig2_grid[i])
        KL= 1/2*((sig1/sig2_grid[i])**2+(µ2-µ1)**2/sig2_grid[i]**2-1+np.log(sig2_grid[i]**2/sig1**2))
        RVD = sig2_grid[i]/sig1*np.exp(0.5*((µ2-µ1)**2/(sig2_grid[i]**2-sig1**2)))
        HD = np.sqrt(1-np.sqrt((2*sig1*sig2_grid[i])/(sig1**2+sig2_grid[i]**2))*np.exp(-0.25*((µ2-µ1)**2/(sig1**2+sig2_grid[i]**2))))
        CSD=np.trapz(((y1-y2)**2)/(y2),x)
        TVD=np.trapz(np.abs(y1-y2),x)/2
        _,corr_eps_KL,_,_= scipy.optimize.fminbound(lambda lam: opt_func(lam,KL,eps_grid[j]),1e-6,1e6,full_output=True)
        eps_KL[i,j]=-corr_eps_KL
        eps_RVD[i,j]=eps_grid[j]/RVD
        eps_HD[i,j]=np.maximum(np.sqrt(eps_grid[j])-HD,0)**2
        if Jiang_CSD_metric:
            eps_CSD[i,j] = eps_grid[j]- (np.sqrt(CSD**2+4*CSD*(eps_grid[j]-eps_grid[j]**2))-(1-2*eps_grid[j])*CSD)/(2*CSD+2)
        else:
            eps_CSD[i,j]=eps_grid[j]+ CSD/2-np.sqrt(eps_grid[j]*CSD+CSD**2/4)
        eps_TVD[i,j]=eps_grid[j]-TVD

# %% Do 4 subplots for different epsilons over sigma 2
eps_gridpoints=[1,10,25,50]
fig, ax= plt.subplots(2,2)
for i in range(2):
    for j in range(2):
        ax[i,j].plot(sig2_grid,eps_KL[:,eps_gridpoints[i*2+j]],label='KL')
        ax[i,j].plot(sig2_grid,eps_RVD[:,eps_gridpoints[i*2+j]],label='RVD')
        ax[i,j].plot(sig2_grid,eps_HD[:,eps_gridpoints[i*2+j]],label='HD')
        ax[i,j].plot(sig2_grid,eps_CSD[:,eps_gridpoints[i*2+j]],label='CSD')
        ax[i,j].plot(sig2_grid,eps_TVD[:,eps_gridpoints[i*2+j]],label='TVD')
        ax[i,j].set_ylabel('$\hat{\epsilon}$')
        ax[i,j].set_xlabel('$\sigma_2$')
        ax[i,j].set_title('$\epsilon$ ='+str(eps_grid[eps_gridpoints[i*2+j]]))
        ax[i,j].set_ylim([1e-6,1])
        ax[i,j].set_yscale('log')
ax[0,0].legend(ncol=2,handleheight=1, labelspacing=0.05)
fig.align_ylabels(ax[:,0])
fig.tight_layout()

# %% Do 4 subplots for different epsilons over sigma 2
eps_gridpoints=[1,10,25,50]
fig, ax= plt.subplots(2,2)
for i in range(2):
    for j in range(2):
        ax[i,j].plot(sig2_grid,eps_KL[:,eps_gridpoints[i*2+j]],label='KL')
        ax[i,j].plot(sig2_grid,eps_RVD[:,eps_gridpoints[i*2+j]],label='RVD')
        ax[i,j].plot(sig2_grid,eps_HD[:,eps_gridpoints[i*2+j]],label='HD')
        ax[i,j].plot(sig2_grid,eps_CSD[:,eps_gridpoints[i*2+j]],label='CSD')
        ax[i,j].plot(sig2_grid,eps_TVD[:,eps_gridpoints[i*2+j]],label='TVD')
        ax[i,j].set_ylabel('$\hat{\epsilon}$')
        ax[i,j].set_xlabel('$\sigma_2$')
        ax[i,j].set_title('$\epsilon$ ='+str(eps_grid[eps_gridpoints[i*2+j]]))
        ax[i,j].set_ylim([1e-6,1])
        ax[i,j].set_yscale('log')
ax[0,0].legend(ncol=2,handleheight=1, labelspacing=0.05)
fig.align_ylabels(ax[:,0])
fig.tight_layout()

# %% For a range aof eps and a range of ambiguity metrics, plot the rescaled epsilons
eps_list = np.linspace(0,1,101)
M_list=np.linspace(0,1,101)
eps_KL=np.zeros((len(eps_list),len(M_list)))
eps_RVD=np.zeros((len(eps_list),len(M_list)))
eps_HD=np.zeros((len(eps_list),len(M_list)))
eps_CSD=np.zeros((len(eps_list),len(M_list)))
eps_TVD=np.zeros((len(eps_list),len(M_list)))
for i in range(len(eps_list)):
    for j in range(len(M_list)):
        _,corr_eps_KL,_,_= scipy.optimize.fminbound(lambda lam: opt_func(lam,M_list[j],eps_list[i]),1e-6,1e6,full_output=True)
        eps_KL[i,j]=-corr_eps_KL
        eps_RVD[i,j]=eps_list[i]/(M_list[j]+1)
        eps_HD[i,j]=np.maximum(np.sqrt(eps_list[i])-M_list[j],0)**2
        if Jiang_CSD_metric:
            eps_CSD[i,j] = eps_list[i]- (np.sqrt(M_list[j]**2+4*M_list[j]*(eps_list[i]-eps_list[i]**2))-(1-2*eps_list[i])*M_list[j])/(2*M_list[j]+2)
        else:
            eps_CSD[i,j]=eps_list[i]+ M_list[j]/2-np.sqrt(eps_list[i]*M_list[j]+M_list[j]**2/4)
        eps_TVD[i,j]=eps_list[i]-M_list[j]

# %% Plot all the rescaled epsilons for the different metrics
fig, ax = plt.subplots(1,1,figsize=(10,5))
eps_idx=20
ax.plot(M_list,eps_KL[eps_idx,:],label='KL')
ax.plot(M_list,eps_RVD[eps_idx,:],label='RVD-1')
ax.plot(M_list,eps_HD[eps_idx,:],label='HD')
ax.plot(M_list,eps_CSD[eps_idx,:],label='CSD')
ax.plot(M_list,eps_TVD[eps_idx,:],label='TVD')
ax.set_ylim([1e-6,eps_list[eps_idx]])
ax.set_ylabel('$\hat{\epsilon}$')
ax.set_xlabel('M')
ax.set_title('Rescaled $\epsilon$ for different metrics')
# %% Do 4 subplots of the above
fig, ax= plt.subplots(2,2,figsize=(8,8))
eps_idx=[1,10,25,50]
for i in range(2):
    for j in range(2):
        ax[i,j].plot(M_list,eps_KL[eps_idx[i*2+j],:],label='KL')
        ax[i,j].plot(M_list,eps_RVD[eps_idx[i*2+j],:],label='RVD-1')
        ax[i,j].plot(M_list,eps_HD[eps_idx[i*2+j],:],label='HD')
        ax[i,j].plot(M_list,eps_CSD[eps_idx[i*2+j],:],label='CSD')
        ax[i,j].plot(M_list,eps_TVD[eps_idx[i*2+j],:],label='TVD')
        ax[i,j].set_ylabel('$\hat{\epsilon}$')
        ax[i,j].set_xlabel('M')
        ax[i,j].set_title('$\epsilon$ ='+str(eps_list[eps_idx[i*2+j]]))
        ax[i,j].set_ylim([1e-6,eps_list[eps_idx[i*2+j]]])
ax[0,0].legend(ncol=2,handleheight=1, labelspacing=0.05)
fig.align_ylabels(ax[:,0])
fig.tight_layout()
# %%
fig.savefig('PRL_over_M.pdf',bbox_inches='tight')
# %%
#fig.savefig('normal_distr_diff_mu.pdf',bbox_inches='tight')
# %%
# Build a family of normal distributions
np.random.seed(0)
N_distr= 25
µ_list = np.random.uniform(-1,1,N_distr)
sig_list = np.random.uniform(1,2,N_distr)
# Find the nominal distribution with the smallest maximum RVD
µ_nom = ca.SX.sym('µ_nom')
sig_nom = ca.SX.sym('sig_nom')
RVD_max = ca.SX.sym('RVD_max')
J=RVD_max
g=[]
lb_g=[]
ub_g=[]
for i in range(N_distr):
    µ=µ_list[i]
    sig=sig_list[i]
    RVD = sig_nom/sig*ca.exp(0.5*((µ_nom-µ)**2/(sig_nom**2-sig**2)))
    g.append(RVD_max-RVD)
    lb_g.append(0)
    ub_g.append(ca.inf)
g.append(sig_nom)
lb_g.append(np.max(sig_list))
ub_g.append(ca.inf)
g=ca.vertcat(*g)
lb_g=ca.vertcat(*lb_g)
ub_g=ca.vertcat(*ub_g)
nlp = {'x':ca.vertcat(µ_nom,sig_nom,RVD_max),'f':J,'g':g}
opts = {'ipopt': {'print_level': 3 }}
solver = ca.nlpsol('res', 'ipopt', nlp,opts)

res= solver(x0=[0,3,1],lbg=lb_g,ubg=ub_g)
µ_nom_RVD=res['x'][0]
sig_nom_RVD=res['x'][1]
M_RVD=res['x'][2]
# %%
print('The nominal distribution with the smallest maximum RVD is: µ=',µ_nom_RVD,' sig=',sig_nom_RVD,' with maximum RVD=',M_RVD)
# %%
# Now we calculate the KL divergence / relative entropy
µ_nom = ca.SX.sym('µ_nom')
sig_nom = ca.SX.sym('sig_nom')
M_max = ca.SX.sym('M_max')
J=M_max
g=[]
lb_g=[]
ub_g=[]
for i in range(N_distr):
    µ=µ_list[i]
    sig=sig_list[i]
    M = 1/2*((sig/sig_nom)**2+(µ-µ_nom)**2/sig_nom**2-1+ca.log(sig_nom**2/sig**2))
    g.append(M_max-M)
    lb_g.append(0)
    ub_g.append(ca.inf)
g.append(sig_nom)
lb_g.append(0)
ub_g.append(ca.inf)
g=ca.vertcat(*g)
lb_g=ca.vertcat(*lb_g)
ub_g=ca.vertcat(*ub_g)
nlp = {'x':ca.vertcat(µ_nom,sig_nom,M_max),'f':J,'g':g}
opts = {'ipopt': {'print_level': 3 }}
solver = ca.nlpsol('res', 'ipopt', nlp,opts)

res= solver(x0=[0,3,1],lbg=lb_g,ubg=ub_g)
µ_nom_KLD=res['x'][0]
sig_nom_KLD=res['x'][1]
M_KLD=res['x'][2]
# %%
print('The nominal distribution with the smallest maximum KL divergence is: µ=',µ_nom_KLD,' sig=',sig_nom_KLD,' with maximum KL divergence=',M_KLD)
# %%
# Now we calculate the Hellinger distance
µ_nom = ca.SX.sym('µ_nom')
sig_nom = ca.SX.sym('sig_nom')
M_max = ca.SX.sym('M_max')
J=M_max
g=[]
lb_g=[]
ub_g=[]
for i in range(N_distr):
    µ=µ_list[i]
    sig=sig_list[i]
    M = ca.sqrt(1-ca.sqrt((2*sig*sig_nom)/(sig**2+sig_nom**2))*ca.exp(-0.25*((µ-µ_nom)**2/(sig**2+sig_nom**2))))
    g.append(M_max-M)
    lb_g.append(0)
    ub_g.append(ca.inf)
g.append(sig_nom)
lb_g.append(0)
ub_g.append(ca.inf)
g=ca.vertcat(*g)
lb_g=ca.vertcat(*lb_g)
ub_g=ca.vertcat(*ub_g)
nlp = {'x':ca.vertcat(µ_nom,sig_nom,M_max),'f':J,'g':g}
opts = {'ipopt': {'print_level': 3 }}
solver = ca.nlpsol('res', 'ipopt', nlp,opts)

res= solver(x0=[0,3,1],lbg=lb_g,ubg=ub_g)
µ_nom_HD=res['x'][0]
sig_nom_HD=res['x'][1]
M_HD=res['x'][2]

# %%
print('The nominal distribution with the smallest maximum Hellinger distance is: µ=',µ_nom_HD,' sig=',sig_nom_HD,' with maximum Hellinger distance=',M_HD)
# %%
# Now we calculate the chi_square distance
# But for this we need to evaluate the integral numerically
µ_nom = ca.SX.sym('µ_nom')
sig_nom = ca.SX.sym('sig_nom')
M_max = ca.SX.sym('M_max')
x_linspace = np.linspace(np.min([µ_list])-20*np.max([sig_list]),np.max([µ_list])+20*np.max([sig_list]),100)
# Define a normal distribution as a casadi function
x = ca.SX.sym('x')
ND = ca.Function('ND',[x,µ_nom,sig_nom],[ca.exp(-0.5*((x-µ_nom)/sig_nom)**2)/(sig_nom*ca.sqrt(2*ca.pi))])
J=M_max
g=[]
lb_g=[]
ub_g=[]
for i in range(N_distr):
    µ=µ_list[i]
    sig=sig_list[i]
    M=0
    print('Iteration: ',i)
    for j in range(len(x_linspace)-1):
        M+= 0.5*(x_linspace[j+1]-x_linspace[j])*((ND(x_linspace[j],µ,sig)-ND(x_linspace[j],µ_nom,sig_nom))**2)/(ND(x_linspace[j],µ_nom,sig_nom))
        M+= 0.5*(x_linspace[j+1]-x_linspace[j])*((ND(x_linspace[j+1],µ,sig)-ND(x_linspace[j+1],µ_nom,sig_nom))**2)/(ND(x_linspace[j+1],µ_nom,sig_nom))
    
    g.append(M_max-M)
    lb_g.append(0)
    ub_g.append(ca.inf)
g=ca.vertcat(*g)
lb_g=ca.vertcat(*lb_g)
ub_g=ca.vertcat(*ub_g)
nlp = {'x':ca.vertcat(µ_nom,sig_nom,M_max),'f':J,'g':g}
opts = {'ipopt': {'print_level': 3 }}
solver = ca.nlpsol('res', 'ipopt', nlp,opts)
# %% Solve
res= solver(x0=[0,3,1],lbg=lb_g,ubg=ub_g)
µ_nom_CSD=res['x'][0]
sig_nom_CSD=res['x'][1]
M_CSD=res['x'][2]
# %%
print('The nominal distribution with the smallest maximum chi_square distance is: µ=',µ_nom_CSD,' sig=',sig_nom_CSD,' with maximum chi_square distance=',M_CSD)

# %% Now we calculate the total variation distance
µ_nom = ca.SX.sym('µ_nom')
sig_nom = ca.SX.sym('sig_nom')
M_max = ca.SX.sym('M_max')
x_linspace = np.linspace(np.min([µ_list])-20*np.max([sig_list]),np.max([µ_list])+20*np.max([sig_list]),100)
# Define a normal distribution as a casadi function
x = ca.SX.sym('x')
ND = ca.Function('ND',[x,µ_nom,sig_nom],[ca.exp(-0.5*((x-µ_nom)/sig_nom)**2)/(sig_nom*ca.sqrt(2*ca.pi))])
J=M_max
g=[]
lb_g=[]
ub_g=[]
for i in range(N_distr):
    µ=µ_list[i]
    sig=sig_list[i]
    M=0
    print('Iteration: ',i)
    for j in range(len(x_linspace)-1):
        M+= 0.5*(x_linspace[j+1]-x_linspace[j])*(0.5*ca.fabs(ND(x_linspace[j],µ,sig)-ND(x_linspace[j],µ_nom,sig_nom)))
        M+= 0.5*(x_linspace[j+1]-x_linspace[j])*(0.5*ca.fabs(ND(x_linspace[j+1],µ,sig)-ND(x_linspace[j+1],µ_nom,sig_nom)))
    
    g.append(M_max-M)
    lb_g.append(0)
    ub_g.append(ca.inf)
g.append(sig_nom)
lb_g.append(1)
ub_g.append(ca.inf)
g=ca.vertcat(*g)
lb_g=ca.vertcat(*lb_g)
ub_g=ca.vertcat(*ub_g)
nlp = {'x':ca.vertcat(µ_nom,sig_nom,M_max),'f':J,'g':g}
opts = {'ipopt': {'print_level': 3 }}
solver = ca.nlpsol('res', 'ipopt', nlp,opts)
# %% Solve
res= solver(x0=[0.01,1.6,0.5],lbg=lb_g,ubg=ub_g)
µ_nom_TVD=res['x'][0]
sig_nom_TVD=res['x'][1]
M_TVD=res['x'][2]

# %%
print('The nominal distribution with the smallest maximum total variation distance is: µ=',µ_nom_TVD,' sig=',sig_nom_TVD,' with maximum total variation distance=',M_TVD)
# %% Plot all the rescaled epsilons for the different metrics
eps_list = np.linspace(0,1,101)
eps_KL=np.zeros((len(eps_grid),1))
eps_RVD=np.zeros((len(eps_grid),1))
eps_HD=np.zeros((len(eps_grid),1))
eps_CSD=np.zeros((len(eps_grid),1))
eps_TVD=np.zeros((len(eps_grid),1))
for i in range(len(eps_list)):
    eps=eps_list[i]

    _,corr_eps_KL,_,_= scipy.optimize.fminbound(lambda lam: opt_func(lam,M_KLD,eps_grid[i]),1e-6,1e6,full_output=True)
    eps_KL[i]=-corr_eps_KL
    eps_RVD[i]=eps_grid[i]/M_RVD
    eps_HD[i]=np.maximum(np.sqrt(eps_grid[i])-M_HD,0)**2
    if Jiang_CSD_metric:
        eps_CSD[i] = eps_grid[i]- (np.sqrt(M_CSD**2+4*M_CSD*(eps_grid[i]-eps_grid[i]**2))-(1-2*eps_grid[i])*M_CSD)/(2*M_CSD+2)
    else:
        eps_CSD[i]=eps_grid[i]+ M_CSD/2-np.sqrt(eps_grid[i]*M_CSD+M_CSD**2/4)
    eps_TVD[i]=eps_grid[i]-M_TVD
print('PRL for RVD for an epsilon of 0.01 is: ',eps_RVD[1])
print('PRL for KL for an epsilon of 0.01 is: ',eps_KL[1])
print('PRL for HD for an epsilon of 0.01 is: ',eps_HD[1])
print('PRL for CSD for an epsilon of 0.01 is: ',eps_CSD[1])
print('PRL for TVD for an epsilon of 0.01 is: ',eps_TVD[1])
# %%
fig, ax = plt.subplots(1,1,figsize=(10,5))
ax.plot(eps_list,eps_KL,label='KL')
ax.plot(eps_list,eps_RVD,label='RVD')
ax.plot(eps_list,eps_HD,label='HD')
ax.plot(eps_list,eps_CSD,label='CSD')
ax.plot(eps_list,eps_TVD,label='TVD')
ax.set_ylabel('$\hat{\epsilon}$')
ax.set_xlabel('$\epsilon$')
ax.set_title('Rescaled $\epsilon$ for different metrics')
ax.set_ylim([1e-6,1])
ax.set_yscale('log')
ax.legend()
fig.tight_layout()


# %% Plot eps_hat/eps
with plt.style.context(['science','ieee']):
    fig, ax = plt.subplots(1,1,figsize=(3.5,2))
    ax.plot(eps_list,[eps_RVD[i]/eps_list[i] for i in range(len(eps_list))],label='Relative variation distance')
    ax.plot(eps_list,[eps_KL[i]/eps_list[i] for i in range(len(eps_list))],label='Kullback-Leibler distance')
    ax.plot(eps_list,[eps_HD[i]/eps_list[i] for i in range(len(eps_list))],label='Hellinger distance')
    ax.plot(eps_list,[eps_CSD[i]/eps_list[i] for i in range(len(eps_list))],label='Chi square distance')
    ax.plot(eps_list,[eps_TVD[i]/eps_list[i] for i in range(len(eps_list))],color='c',ls= (0, (5, 10)),label='Total variation distance')
    ax.set_ylabel('PRL over risk level $\hat{\epsilon}_M/\epsilon$')
    ax.set_xlabel('Risk level $\epsilon$')
    #ax.set_title('Rescaled $\epsilon$ for different metrics')
    ax.set_ylim([1e-2,1])
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()


# %%
fig.savefig('normal_distr_eps_hat_normalized.pdf',bbox_inches='tight')
# %%
# %% Do similarly for a range of epsilons and a range of sig2 with normalized metrics
µ1= 0
µ2 =0
sig1= 1
eps_grid=np.linspace(0,1,101)
sig2_grid=np.linspace(1,10,101)
KL = np.zeros((len(sig2_grid),len(eps_grid)))
HD = np.zeros((len(sig2_grid),len(eps_grid)))
CSD = np.zeros((len(sig2_grid),len(eps_grid)))
TVD = np.zeros((len(sig2_grid),len(eps_grid)))

KL_norm=np.zeros((len(sig2_grid),len(eps_grid)))
RVD=np.zeros((len(sig2_grid),len(eps_grid)))
HD_norm=np.zeros((len(sig2_grid),len(eps_grid)))
CSD_norm=np.zeros((len(sig2_grid),len(eps_grid)))
TVD_norm=np.zeros((len(sig2_grid),len(eps_grid)))

for i in range(len(sig2_grid)):
    
    x= np.linspace(min([µ1,µ2])-5*max([sig1,sig2_grid[i]]),max([µ1,µ2])+5*max([sig1,sig2_grid[i]]),10000)
    y1= stats.norm.pdf(x,µ1,sig1)
    y2= stats.norm.pdf(x,µ2,sig2_grid[i])
    KL[i,:]= 1/2*((sig1/sig2_grid[i])**2+(µ2-µ1)**2/sig2_grid[i]**2-1+np.log(sig2_grid[i]**2/sig1**2))
    KL_norm[i,:]= np.trapz(y1/y2*np.log(y1/y2),x)
    RVD[i,:] = sig2_grid[i]/sig1*np.exp(0.5*((µ2-µ1)**2/(sig2_grid[i]**2-sig1**2)))
    HD[i,:] = np.sqrt(1-np.sqrt((2*sig1*sig2_grid[i])/(sig1**2+sig2_grid[i]**2))*np.exp(-0.25*((µ2-µ1)**2/(sig1**2+sig2_grid[i]**2))))
    HD_norm[i,:]= np.sqrt(np.sqrt(0.5*np.trapz((np.sqrt(y1/y2)-np.sqrt(1))**2,x)))
    CSD[i,:]=np.trapz(((y1-y2)**2)/(y2),x)
    CSD_norm[i,:]= np.trapz(((y1-y2)**2)/(y2*y2),x)
    TVD[i,:]=np.trapz(np.abs(y1-y2),x)/2
    TVD_norm[i,:]=np.trapz(np.abs((y1/y2-1)),x)/2
        #_,corr_eps_KL,_,_= scipy.optimize.fminbound(lambda lam: opt_func(lam,KL[i,j],eps_grid[j]),1e-6,1e6,full_output=True)
        #eps_KL[i,j]=-corr_eps_KL
        #eps_RVD[i,j]=eps_grid[j]/RVD[i,j]
        #eps_HD[i,j]=np.maximum(np.sqrt(eps_grid[j])-HD[i,j],0)**2
        #eps_CSD[i,j]=eps_grid[j]+ CSD[i,j]/2-np.sqrt(eps_grid[j]*CSD[i,j]+CSD[i,j]**2/4)
        #eps_TVD[i,j]=eps_grid[j]-TVD[i,j]

# %% Do 4 subplots for different normalized metrics over sigma 2
fig, ax= plt.subplots(1,1)

ax.plot(sig2_grid,KL_norm[:,0],label='KL_norm')
ax.plot(sig2_grid,RVD[:,0],label='RVD')
ax.plot(sig2_grid,HD_norm[:,0],label='HD_norm')
ax.plot(sig2_grid,CSD_norm[:,0],label='CSD_norm')
ax.plot(sig2_grid,TVD_norm[:,0],label='TVD_norm')
ax.set_ylabel('Metrics normed')
ax.set_xlabel('$\sigma_2$')
#ax[i,j].set_ylim([1e-6,1])
ax.set_yscale('log')
ax.legend()
fig.align_ylabels()
fig.tight_layout()

# %%
fig, ax= plt.subplots(1,1)

ax.plot(sig2_grid,KL[:,0],label='KL')
ax.plot(sig2_grid,RVD[:,0],label='RVD')
ax.plot(sig2_grid,HD[:,0],label='HD')
ax.plot(sig2_grid,CSD[:,0],label='CSD')
ax.plot(sig2_grid,TVD[:,0],label='TVD')
ax.set_ylabel('Metrics')
ax.set_xlabel('$\sigma_2$')
#ax[i,j].set_ylim([1e-6,1])
ax.set_yscale('log')
ax.legend()
fig.align_ylabels()
fig.tight_layout()
# %%
M=1e-9
x=np.linspace(1e-15,1,100000)-1
y=1-(M/(M+1))-(np.sqrt(M**2/x**2+4*M*(1/x-1))-np.sqrt(M**2/x**2))/(2*M+2)
y_tseng=(x+M/2-np.sqrt((M*x+M**2/4)))/x
# %%
plt.plot(x,y)
plt.plot(x,y_tseng)


# %% Get some normal distributions, for which the RVD is smaller than 2
# Setup optimization problem finding the biggest sigam for given mu
µ = ca.SX.sym('µ')
µ1 = ca.SX.sym('µ1')
sigma = ca.SX.sym('sigma')
sigma_max = ca.SX.sym('sigma_max')
RVD_max = ca.SX.sym('RVD_max')
J = sigma
g = []
lb_g = []
ub_g = []

g.append(RVD_max-sigma_max/sigma*ca.exp(0.5*((µ-µ1)**2/(sigma_max**2-sigma**2))))
lb_g.append(0)
ub_g.append(0)
g.append(sigma_max-sigma)
lb_g.append(0)
ub_g.append(ca.inf)
g = ca.vertcat(*g)
lb_g = ca.vertcat(*lb_g)
ub_g = ca.vertcat(*ub_g)
nlp = {'x':ca.vertcat(sigma),'f':J,'g':g,'p':ca.vertcat(µ,µ1,sigma_max,RVD_max)}
opts = {'ipopt': {'print_level': 5 }}
solver = ca.nlpsol('res', 'ipopt', nlp,opts)
# %% Solve the optimization problem
µ1_val = 0
sigma_max_val = 1
RVD_max_val =[2]

µ_list=np.arange(-2,2,0.1)
sig_list = np.zeros((len(µ_list),len(RVD_max_val)))
for j in range(len(RVD_max_val)):
    for i in range(len(µ_list)):
        res = solver(x0=0.1,lbx=1e-12,ubx=ca.inf,lbg=lb_g,ubg=ub_g,p=ca.vertcat(µ_list[i],µ1_val,sigma_max_val,RVD_max_val[j]))
        if solver.stats()['success']:
            sig_list[i,j] = res['x'][0]
        else:
            sig_list[i,j] = -1


# %% Plot the normal distributions
with plt.style.context(['science','ieee']):
    if len(RVD_max_val)>1:
        fig, ax = plt.subplots(1,len(RVD_max_val),sharex=True,sharey=True)
        # Plot the base normal distribution in thick black
        x = np.linspace(-3,3,1000)
        for j in range(len(RVD_max_val)):
            ax[j].plot(x,stats.norm.pdf(x,µ1_val,sigma_max_val),'k',linewidth=2,label='$f_{\hat{\mathcal{P}}}(\delta)$')
            ax[j].plot(x,RVD_max_val[j]*stats.norm.pdf(x,µ1_val,sigma_max_val),'r',linewidth=2,label='$M_{RVD} f_{\hat{\mathcal{P}}}(\delta)$')
            # Plot the normal distributions for the different µ values in grey thin lines
            for i in range(len(µ_list)):
                if sig_list[i,j]>0:
                    y = stats.norm.pdf(x,µ_list[i],sig_list[i,j])
                    if i == µ_list.size//2:
                        ax[j].plot(x,y,'grey',ls='-',alpha=0.5,label='$f_{\mathcal{P}}(\delta)$')
                    else:
                        ax[j].plot(x,y,'grey',ls='-',alpha=0.5)
            ax[j].set_xlabel('$\delta$')

        ax[0].set_ylabel('f($\delta$)')
        ax[0].legend()
        ax[1].legend()
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=(1.75,1.75))
        # Plot the base normal distribution in thick black
        x = np.linspace(-3,3,1000)
        ax.plot(x,stats.norm.pdf(x,µ1_val,sigma_max_val),'k',linewidth=2,label='$f_{\hat{\mathcal{P}}}(\delta)$')
        ax.plot(x,RVD_max_val[j]*stats.norm.pdf(x,µ1_val,sigma_max_val),'r',linewidth=2,label='$M_{RVD} f_{\hat{\mathcal{P}}}(\delta)$')
        # Plot the normal distributions for the different µ values in grey thin lines
        for i in range(len(µ_list)):
            if sig_list[i,0]>0:
                y = stats.norm.pdf(x,µ_list[i],sig_list[i,0])
                if i == µ_list.size//2:
                    ax.plot(x,y,'grey',ls='-',alpha=0.5,label='$f_{\mathcal{P}}(\delta)$')
                else:
                    ax.plot(x,y,'grey',ls='-',alpha=0.5)
        ax.set_xlabel('$\delta$')
        ax.set_ylabel('f($\delta$)')
        ax.set_ylim([0,3])
        ax.legend()
        fig.tight_layout()





# %%
fig.savefig('normal_distr_diff_mu_RVD_{}.pdf'.format(RVD_max_val[0]),bbox_inches='tight')

# %% Get multiple gaussian distributions for which the kullbach leibler divergence is smaller than 0.2
# Setup optimization problem finding the biggest sigam for given mu
µ = ca.SX.sym('µ')
µ1 = ca.SX.sym('µ1')
sigma = ca.SX.sym('sigma')
sigma_max = ca.SX.sym('sigma_max')
KL_max = ca.SX.sym('RVD_max')
J = sigma
g = []
lb_g = []
ub_g = []

g.append(KL_max-(1/2*((sigma/sigma_max)**2+(µ-µ1)**2/sigma_max**2-1+ca.log(sigma_max**2/sigma**2))))
lb_g.append(0)
ub_g.append(0)
# g.append(sigma_max-sigma)
# lb_g.append(0)
# ub_g.append(ca.inf)
g = ca.vertcat(*g)
lb_g = ca.vertcat(*lb_g)
ub_g = ca.vertcat(*ub_g)
nlp = {'x':ca.vertcat(sigma),'f':J,'g':g,'p':ca.vertcat(µ,µ1,sigma_max,KL_max)}
opts = {'ipopt': {'print_level': 5 }}
solver = ca.nlpsol('res', 'ipopt', nlp,opts)
# %% Solve the optimization problem
µ1_val = 0
sigma_max_val = 1
KL_max_val =[0.2]
µ_list=np.arange(-2,2,0.1)
sig_list = np.zeros((len(µ_list),len(RVD_max_val)))
for j in range(len(KL_max_val)):
    for i in range(len(µ_list)):
        res = solver(x0=0.1,lbx=1e-12,ubx=ca.inf,lbg=lb_g,ubg=ub_g,p=ca.vertcat(µ_list[i],µ1_val,sigma_max_val,KL_max_val[j]))
        if solver.stats()['success']:
            sig_list[i,j] = res['x'][0]
        else:
            sig_list[i,j] = -1
# %%
if len(KL_max_val)>1:
    fig, ax = plt.subplots(1,len(KL_max_val),sharex=True,sharey=True,figsize=(6,3))
    # Plot the base normal distribution in thick black
    x = np.linspace(-3,3,1000)
    for j in range(len(RVD_max_val)):
        ax[j].plot(x,stats.norm.pdf(x,µ1_val,sigma_max_val),'k',linewidth=2,label='$f_{\hat{\mathcal{P}}}(\delta)$')
        # Plot the normal distributions for the different µ values in grey thin lines
        for i in range(len(µ_list)):
            if sig_list[i,j]>0:
                y = stats.norm.pdf(x,µ_list[i],sig_list[i,j])
                if i == µ_list.size//2:
                    ax[j].plot(x,y,'grey',alpha=0.5,label='$f_{\mathcal{P}}(\delta)$')
                else:
                    ax[j].plot(x,y,'grey',alpha=0.5)
        ax[j].set_xlabel('$\delta$')

    ax[0].set_ylabel('f($\delta$)')
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
else:
    fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=(3,3))
    # Plot the base normal distribution in thick black
    x = np.linspace(-3,3,1000)
    ax.plot(x,stats.norm.pdf(x,µ1_val,sigma_max_val),'k',linewidth=2,label='$f_{\hat{\mathcal{P}}}(\delta)$')
    # Plot the normal distributions for the different µ values in grey thin lines
    for i in range(len(µ_list)):
        if sig_list[i,0]>0:
            y = stats.norm.pdf(x,µ_list[i],sig_list[i,0])
            if i == µ_list.size//2:
                ax.plot(x,y,'grey',alpha=0.5,label='$f_{\mathcal{P}}(\delta)$')
            else:
                ax.plot(x,y,'grey',alpha=0.5)
    ax.set_xlabel('$\delta$')
    ax.set_ylabel('f($\delta$)')
    ax.set_ylim([0,2])
    ax.legend()
    fig.tight_layout()
# %%
