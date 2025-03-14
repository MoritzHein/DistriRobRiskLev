# %%
import numpy as np
import casadi as ca
import casadi.tools as ca_tools
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib as mpl
import control 
import scipy
import scienceplots
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})
# %%
'''Some functions for the scenario program'''
def calc_beta(N,eps,d):
    summand2=0
    for k in range(0,d):
        term2=scipy.special.comb(N,k)*eps**k*(1-eps)**(N-k)
        summand2+=term2
    P_viol=summand2
    return P_viol
def calc_bet(N,eps,d):
    return scipy.special.betainc(N-d+1,d,1-eps)

def calc_my_value_old(N,d,gamma,eps):
    P_viol=0
    for i in range(d,N+1):
        term1=scipy.special.comb(N,i)*gamma**i*(1-gamma)**(N-i)
        summand2=0
        for k in range(0,d):
            term2=scipy.special.comb(i,k)*eps**k*(1-eps)**(i-k)
            summand2+=term2
        P_viol+=term1*summand2
    summand3=0
    for i in range(0,d):
        term3=scipy.special.comb(N,i)*gamma**i*(1-gamma)**(N-i)
        summand3+=term3
    P_viol+=summand3
    return P_viol

def calc_my_value(N,d,gamma,eps):
    P_viol=0
    for i in range(d,N+1):
        term1=scipy.special.bdtr(i,N,gamma)-scipy.special.bdtr(i-1,N,gamma)
        #term1=scipy.special.comb(N,i)*gamma**i*(1-gamma)**(N-i)
        summand2=scipy.special.betainc(i-d+1,d,1-eps)
        P_viol+=term1*summand2
    summand3=scipy.special.betainc(N-d+1,d,1-gamma)
    P_viol+=summand3
    return P_viol

def calc_expected_eps(N,d,gamma):
    deps=0
    for i in range(d,N+1):
        term1=scipy.special.bdtr(i,N,gamma)-scipy.special.bdtr(i-1,N,gamma)
        #term1=scipy.special.comb(N,i)*gamma**i*(1-gamma)**(N-i)
        summand2=d/(i+1)
        deps+=term1*summand2
    return deps+scipy.special.bdtr(d-1,N,gamma)

def get_min_N_SP(eps,bet,d):
    N1 = d
    N2 = np.ceil(2/eps*(d-1+np.log(1/bet)))
    while N2-N1>1:
        N = np.floor((N1+N2)/2)
        if calc_beta(N,eps,d) > bet:
            N1=N
        else:
            N2=N
    N = N2
    return N

def get_min_N_gamma(eps,bet,d,gamma):
    N1 = d
    N2 = int(np.ceil(2/eps*(d-1+np.log(1/bet))/gamma))
    bet_tes=calc_my_value(N2,d,gamma,eps)
    assert bet_tes<bet, "N2 too small, real beta is {}".format(bet_tes)
    while N2-N1>1:
        N = int(np.floor((N1+N2)/2))
        if calc_my_value(N,d,gamma,eps) > bet:
            N1=N
        else:
            N2=N
    N = N2
    return N

def my_density_function(eps_eval,N,d,gamma):
    density=0
    for i in range(d,N+1):
        term1=scipy.special.bdtr(i,N,gamma)-scipy.special.bdtr(i-1,N,gamma)
        term2=scipy.special.bdtr(d,i,eps_eval)-scipy.special.bdtr(d-1,i,eps_eval)
        density+=term1*d*term2/eps_eval
    return density

def SP_density_function(eps_eval,N,d):
    return d*(scipy.special.bdtr(d,N,eps_eval)-scipy.special.bdtr(d-1,N,eps_eval))/eps_eval

# %% Custom Density Function triangular with 0.25 at the edges
def pdf1(x):
    # takes an input between -inf and inf and returns the according PDF
    # if x<=-1:
    #     y=0
    # elif x<0:
    #     y=0.75+0.5*x
    # elif x<1:
    #     y=0.75-0.5*x
    # else:
    #     x=0
    y=0+((-1<=x)&(x<0))*(0.75+0.5*x)+((0<=x)&(x<1))*(0.75-0.5*x)
    return y

def cdf1(x):
    y=(x<-1)*0+((-1<=x)&(x<0))*(0.5+0.75*x+0.25*x**2)+((0<=x)&(x<1))*(0.5+0.75*x-0.25*x**2)+1*(x>=1)
    return y

def inv_cdf1(x): 
    # x can be just between 0 and 1
    y=(x<=0.5)*(-1.5+np.sqrt(2.25-2+4*x))+(x>0.5)*(1.5-np.sqrt(2.25+2-4*x))
    return y

gamma=0.5

# %% Custom Density Function triangular with 0.75 at the edges and 0.25 in the middle
def pdf2(x):
    # takes an input between -inf and inf and returns the according PDF
    # if x<=-1:
    #     y=0
    # elif x<0:
    #     y=0.75+0.5*x
    # elif x<1:
    #     y=0.75-0.5*x
    # else:
    #     x=0
    y=0+((-1<=x)&(x<0))*(0.25-0.5*x)+((0<=x)&(x<1))*(0.25+0.5*x)
    return y

def cdf2(x):
    y=(x<-1)*0+((-1<=x)&(x<0))*(0.5+0.25*x-0.25*x**2)+((0<=x)&(x<1))*(0.5+0.25*x+0.25*x**2)+1*(x>=1)
    return y

def inv_cdf2(x): 
    # x can be just between 0 and 1
    y=(x<=0.5)*(0.5-np.sqrt(np.sqrt((2.25-4*x)**2)))+(x>0.5)*(-0.5+np.sqrt(np.sqrt((0.25-2+4*x)**2)))
    return y

gamma=0.5
# %% Custom Density Function uniformly half of the time, the other half just 0
def inv_cdf3(x):
    # x can be just between 0 and 1
    y=(x<=0.25)*(-1+4*x)+((x>0.25)&(x<=0.75))*0+(x>0.75)*(4*x-3)
    return y
gamma=0.5

# %% Custom Density Function uniformly half of the time, the other half 0.5, so with an offset
def inv_cdf4(x):
    # x can be just between 0 and 1
    y=(x<=0.375)*(-1+4*x)+((x>0.375)&(x<=0.85))*0.5+(x>0.8755)*(4*x-3)
    return y
gamma=0.5

def inv_cdf_uni(x):
    # x can be just between 0 and 1
    y=-1+2*x
    return y
def pdf_uni(x):
    y= (x<=-1)*0+(-1<x)*(x<=1)*0.5+(x>=1)*0
    return y
def inv_cdf5(x):
    # uniform, but not around zero
    y=(x<=0.5)*(-1+x)+(x>0.5)*(x)
    return y

def pdf5(x):
    # uniform, but not around zero
    y=(x<=-1)*0+(-1<x)*(x<=-0.5)*(1)+(1>=x)*(x>0.5)*1
    return y

# %%
def get_custom_samples(lb,ub,size):
    s_uni=np.random.uniform(0,1,size)
    #y=inv_cdf1(s_uni)
    y=inv_cdf5(s_uni)
    #Now scale y from -1,1 to lb and ub
    y*=(ub-lb)/2
    y+=(lb+ub)/2
    return y

# %% system model 2D linear double integrator
#T=0.1
nx = 2
nu = 1
nw = 2
A=np.array([[1,1],[0,1]])
B=np.array([[0.5],[1]])
C = np.array([[1, 0]])
D = np.array([[0]])
# casadi function
x = ca.SX.sym('x', nx)
u = ca.SX.sym('u',nu)
x_next = ca.mtimes(A, x) + B*u
system = ca.Function('sys', [x, u], [x_next])
# Feedback controller - LQR
K,_,_ = control.dlqr(A, B, np.eye(2), 1)
K=-K.reshape(1,2)
print(K)
K = np.round(K,2)
A_K = A + B@K
print(np.linalg.eig(A_K))
system_fb = ca.Function('sys_fb', [x,u], [A_K@x+B@u])
# Disturbances: Additive noise
w_scale=0.2
# %% Simulate a trajectory
x0 = np.array([[1], [1]])
N = 100
x_traj = np.zeros((nx, N+1))
x_traj[:,0] = x0.flatten()
for i in range(N):
    x_traj[:,i+1] = (system(x_traj[:,i], K@x_traj[:,i]) + w_scale*np.random.randn(nx,1)).full().flatten()  
# Plot
fig, ax = plt.subplots()
ax.plot(x_traj[0,:], x_traj[1,:], 'r.-')




# %%

# Constraint: x1 < 2, x2 < 2, x1 > -0.5, x2 > -0.5, u < 5, u > -5
h= []
h.append(x[0] - 2)
h.append(x[1] - 2)
h.append(-x[0] -0.5)
h.append(-x[1] -0.5)
h_N = ca.vertcat(*h)
nh_N = h_N.shape[0]
h.append(u-1)
h.append(-u-1),
h = ca.vertcat(*h)
nh_k = h.shape[0]
con = ca.Function('con', [x,u], [h])
ter_con = ca.Function('ter_con', [x], [h_N])
P_create = ca.SX.sym('P', nx, nx)
K_create = ca.SX.sym('K', nu, nx)
H= []
for i in range(nh_k):
    H.append( (ca.gradient(h[i], ca.vertcat(x,u))).T@ca.vertcat(np.eye(nx), K_create)@P_create@(ca.vertcat(np.eye(nx), K_create).T)@ca.gradient(h[i], ca.vertcat(x,u)))
H = ca.vertcat(*H)
H_fun = ca.Function('H', [x, u, P_create,K_create], [H])
HN = []
for i in range(nh_N):
    HN.append( (ca.gradient(h_N[i], x)).T@P_create@(ca.gradient(h_N[i], x)))
HN = ca.vertcat(*HN)
HN_fun = ca.Function('HN', [x, P_create], [HN])
# cost function
Q = np.eye(2)
R = 1
x_ref = np.array([[-0.5], [-0.5]])
#x_ref = np.zeros((2,1))
stage_cost = (x-x_ref).T@Q@(x-x_ref) + u*R@u
stage_cost_fcn = ca.Function('stage_cost', [x, u], [stage_cost])
terminal_cost = 5*x.T@Q@x
terminal_cost_fcn = ca.Function('terminal_cost', [x], [terminal_cost])
#%% Setup ellispoidal optimization problem
N=2
N_samp = 1000
print('Mean eps: ',N*nu/(N_samp+1))

opt_x = ca_tools.struct_symSX([ca_tools.entry('x',repeat = [N+1,N_samp], shape=(nx)),
                                 ca_tools.entry('u', repeat = [N], shape=(nu))])
opt_p = ca_tools.struct_symSX([ca_tools.entry('w', repeat = [N_samp], shape=(nw)),
                               ca_tools.entry('x_init', repeat = [], shape=(nx))])    
#%%
J = 0
g = []
lb_g = []
ub_g = []

x_init = opt_p['x_init']
for s in range(N_samp):
    g.append(opt_x['x',0,s] - x_init)
    lb_g.append(np.zeros((nx,1)))
    ub_g.append(np.zeros((nx,1)))

    for i in range(N):
        # Cost
        J += stage_cost_fcn(opt_x['x',i,s], opt_x['u',i])
        # if i >0:
        #     J+= (opt_x['u',i]-opt_x['u',i-1]).T@R@(opt_x['u',i]-opt_x['u',i-1])
        # else:
        #     J+= opt_x['u',i].T@R@opt_x['u',i]
        # Nominal dynamics
        g.append(opt_x['x',i+1,s] - system_fb(opt_x['x',i,s], opt_x['u',i])-opt_p['w',s])
        lb_g.append(np.zeros((nx,1)))
        ub_g.append(np.zeros((nx,1)))
        # Constraint
        g.append(con(opt_x['x',i,s], K@opt_x['x',i,s]+opt_x['u',i]))
        lb_g.append(-np.ones((nh_k,1))*ca.inf)
        ub_g.append(np.zeros((nh_k,1)))

    # Terminal cost
    J += terminal_cost_fcn(opt_x['x',N,s])
    # Terminal constraint
    g.append(ter_con(opt_x['x',N,s]))
    lb_g.append(-np.ones((nh_N,1))*ca.inf)
    ub_g.append(np.zeros((nh_N,1)))

g = ca.vertcat(*g)
lb_g = ca.vertcat(*lb_g)
ub_g = ca.vertcat(*ub_g)

prob = {'f': J, 'x': opt_x.cat, 'g': g,'p':opt_p.cat}
solver_opt = {'ipopt.linear_solver': 'MA27','ipopt.print_level':0, 'print_time':0, 'ipopt.tol':1e-8}
solver = ca.nlpsol('solver', 'ipopt', prob, solver_opt)
# %% Test solver
x_init = np.array([[2.0], [-0.5]])
#x_init = np.array([[1.0], [1.0]])
# Initialize random samples
w_samp = w_scale*np.random.uniform(-1,1,(nw,N_samp))
opt_p_num = opt_p(0)

for s in range(N_samp):
    opt_p_num['w',s] = w_samp[:,s]
opt_p_num['x_init'] = x_init
opt_x_init = opt_x(0)
# %% def func to get u_traj
def get_u_traj():
    # Get opt_p_num
    w_samp = w_scale*np.random.uniform(-1,1,(nw,N_samp))
    opt_p_num = opt_p(0)
    for s in range(N_samp):
        opt_p_num['w',s] = w_samp[:,s]
    opt_p_num['x_init'] = x_init
    opt_x_init = opt_x(0)
    # Solve
    res = solver(x0=opt_x_init,p=opt_p_num,lbg=lb_g, ubg=ub_g)
    print(solver.stats()['return_status'])
    opt_x_num = opt_x(res['x'])
    u_traj = np.array(ca.horzcat(*opt_x_num['u',:]))
    if solver.stats()['return_status'] == 'Solve_Succeeded':
        return u_traj
    else:
        return None
# %%

opt_x_init['x'] = x_init

opt_x_init['u'] = np.zeros((1,1))
res = solver(x0=opt_x_init,p=opt_p_num,lbg=lb_g, ubg=ub_g)
opt_x_num = opt_x(res['x'])
print(solver.stats()['return_status'])
# %% Plot the ellipsoids
fig, ax = plt.subplots(2,2)
# Phase plot all predictions
for s in range(N_samp):
    x_traj = np.array(ca.horzcat(*opt_x_num['x',:,s]))
    ax[0,0].plot(x_traj[0,:], x_traj[1,:], '-', alpha=0.1)
# Plot the constraints
ax[0,0].plot([2,2],[-0.5,2],'k')
ax[0,0].plot([-0.5,-0.5],[-0.5,2],'k')
ax[0,0].plot([-0.5,2],[2,2],'k')
ax[0,0].plot([-0.5,2],[-0.5,-0.5],'k')
ax[0,0].set_xlabel('x1')
ax[0,0].set_ylabel('x2')
ax[0,1].set_title('input')
ax[0,1].set_xlabel('time')
ax[1,0].set_title('x1')
ax[1,0].set_xlabel('time')
ax[1,1].set_title('x2')
ax[1,1].set_xlabel('time')
fig.align_labels()
fig.tight_layout()

# %% Now validate the solution by running the closed loop system
N_val = 1000
u_traj = np.array(ca.horzcat(*opt_x_num['u',:]))
X_traj = np.zeros((nx,N+1,N_val))
X_traj[:,0,:] = x_init
W_traj = np.zeros((nx,N_val))
W_traj[:,:] = w_scale*np.random.uniform(-1,1,(nx,N_val))
for i in range(N):
    X_traj[:,i+1,:] = (system_fb(X_traj[:,i,:], u_traj[:,i]) + W_traj[:,:])
# %% def func to validate
def validate(u_traj,N_val,mode):
    X_traj = np.zeros((nx,N+1,N_val))
    X_traj[:,0,:] = x_init
    W_traj = np.zeros((nx,N_val))
    if mode=='custom':
        W_traj[:,:] = w_scale*get_custom_samples(-1,1,(nx,N_val))
    if mode=='uniform':
        W_traj[:,:] = w_scale*np.random.uniform(-1,1,(nx,N_val))
    for i in range(N):
        #W_traj[:,i,:] = w_scale*np.random.uniform(-1,1,(nx,N_val))
        X_traj[:,i+1,:] = (system_fb(X_traj[:,i,:], u_traj[:,i]) + W_traj[:,:])
    # Check constraints
    n_viol_run = 0
    tol = 1e-6

    indi=0
    for i in range(N):
        indi+= np.any(con(X_traj[:,i,:], K@X_traj[:,i,:]+ u_traj[:,i])>tol,axis=0)
    indi+= np.any(ter_con(X_traj[:,N,:])>tol,axis=0)
    n_viol_run = np.sum(indi>0)
    return n_viol_run


# %%
# PLot the trajectories
fig, ax = plt.subplots(2,2)
n_run_viol = 0
tol = 1e-6
for s in range(N_val):
    ax[0,0].plot(X_traj[0,:,s], X_traj[1,:,s], '-', alpha=0.1)
    n_viol_this_run = 0
    for i in range(N):
        if np.any(con(X_traj[:,i,s], K@X_traj[:,i,s]+u_traj[:,i])>tol):
            n_viol_this_run += 1
    if np.any(ter_con(X_traj[:,N,s])>tol):
        n_viol_this_run += 1
    if n_viol_this_run >= 1:
        n_run_viol += 1
# Plot the constraints
ax[0,0].plot([2,2],[-0.5,2],'k')
ax[0,0].plot([-0.5,-0.5],[-0.5,2],'k')
ax[0,0].plot([-0.5,2],[2,2],'k')
ax[0,0].plot([-0.5,2],[-0.5,-0.5],'k')
ax[0,0].set_xlabel('x1')
ax[0,0].set_ylabel('x2')
ax[0,1].set_title('input')
ax[0,1].set_xlabel('time')
ax[1,0].set_title('x1')
ax[1,0].set_xlabel('time')
ax[1,1].set_title('x2')
ax[1,1].set_xlabel('time')
fig.align_labels()
fig.tight_layout()

# %%
print(n_run_viol/N_val)
# %% Sample multiple trajectories and get their violation probability
N_val = 40000
n_run_viol_DR =[]
n_run_viol_nom = []
for i in range(800):
    print('Iteration: ', i)
    u_traj = get_u_traj()
    if u_traj is not None:
        n_run_viol_DR.append(validate(u_traj,N_val,'custom')/N_val)
        n_run_viol_nom.append(validate(u_traj,N_val,'uniform')/N_val)
# %%
fig, ax = plt.subplots()
ax.hist(n_run_viol_DR,density=True,color='tab:blue',alpha=0.5)
ax.hist(n_run_viol_nom,density=True,color='tab:orange',alpha=0.5)
#Plot the density
eps= np.linspace(1e-6,0.05,200)
ax.set_xlim([0,0.03])
density = np.zeros(eps.shape)
density_gamma = np.zeros(eps.shape)
for i in range(eps.shape[0]):
    density[i] = SP_density_function(eps[i],N_samp,nu*N)
    density_gamma[i] = my_density_function(eps[i],N_samp,nu*N,gamma**nx)
ax.plot(eps,density,'tab:orange',label='Nominal violation probability density')
ax.plot(eps,density_gamma,'tab:blue',label='DR violation probability density')
ax.axvline(np.mean(n_run_viol_DR), color='tab:blue',ls=':',label='Empirical violation probability DR')
ax.axvline(np.mean(n_run_viol_nom), color='tab:orange',ls=':',label='Empirical violation probability nominal')
ax.axvline(calc_expected_eps(N_samp,nu*N,gamma**nx), color='tab:blue',ls='--',label='Expected violation probability DR')
ax.axvline((nu*N)/(N_samp+1), color='tab:orange',ls='--',label='Expected violation probability nominal')
ax.set_xlabel('Violation Probability')
ax.set_ylabel('Density')
ins = ax.inset_axes([0.5,0.2,0.3,0.3])
# PLot DR density
x=np.linspace(-1,1,1000)
ins.plot(x,1/w_scale**2*pdf5(x/w_scale),'tab:blue',label='$f_{\mathcal{P}_{bor}}(w)$')
# Plot nominal density
ins.plot([-1,-w_scale,-w_scale, w_scale, w_scale,1],[0,0,1/(2*w_scale)**2,1/(2*w_scale)**2,0,0],'tab:orange',label='$f_{\hat{\mathcal{P}}}(w)$')
ins.plot([-1,-w_scale,-w_scale, w_scale, w_scale,1],2**nx*np.array([0,0,1/(2*w_scale)**2,1/(2*w_scale)**2,0,0]),'tab:orange',ls='--',label='$M_{RVD} f_{\hat{\mathcal{P}}}(w)$')
ins.set_xlim([-0.3,0.3])
ins.set_xlabel('$w_i$')
ins.set_ylabel('Density')
ins.legend(loc=[0.9,0.1])
ax.legend()
fig.tight_layout()
#fig.savefig('violation_density.pdf')
# %%
with plt.style.context(['science', 'ieee']):

    fig, ax = plt.subplots(figsize=(4,3))
    ax.hist(n_run_viol_DR,density=True,color='tab:blue',alpha=0.5,label='Empirical density of $ V_{\mathcal{P}} (\hat{x}_s(\mathbb{D}_N))$')
    ax.hist(n_run_viol_nom,density=True,color='tab:orange',alpha=0.5, label='Empirical density of $ V_{\hat{\mathcal{P}}} (\hat{x}_s(\mathbb{D}_N))$')
    #Plot the density
    eps= np.linspace(1e-6,0.05,200)
    ax.set_xlim([0,0.03])
    density = np.zeros(eps.shape)
    density_gamma = np.zeros(eps.shape)
    for i in range(eps.shape[0]):
        density[i] = SP_density_function(eps[i],N_samp,nu*N)
        density_gamma[i] = my_density_function(eps[i],N_samp,nu*N,gamma**nx)
    ax.plot(eps,density,'tab:orange',label='Density of $ V_{\hat{\mathcal{P}}} (\hat{x}_s(\mathbb{D}_N))$')
    ax.plot(eps,density_gamma,'tab:blue',ls='-',label='Density of $ V_{\mathcal{P}} (\hat{x}_s(\mathbb{D}_N))$')
    ax.axvline(np.mean(n_run_viol_DR), color='tab:blue',ls=':',label='Empirical mean of $ V_{\mathcal{P}} (\hat{x}_s(\mathbb{D}_N))$')
    ax.axvline(np.mean(n_run_viol_nom), color='tab:orange',ls=':',label='Empirical mean of $ V_{\hat{\mathcal{P}}} (\hat{x}_s(\mathbb{D}_N))$')
    ax.axvline(calc_expected_eps(N_samp,nu*N,gamma**nx), color='tab:blue',ls='--',label='Expected mean of $ V_{\mathcal{P}} (\hat{x}_s(\mathbb{D}_N))$')
    ax.axvline((nu*N)/(N_samp+1), color='tab:orange',ls='--',label='Expected mean of $ V_{\hat{\mathcal{P}}} (\hat{x}_s(\mathbb{D}_N))$')
    ax.set_xlabel('Violation Probability')
    ax.set_ylabel('Density')
    ins = ax.inset_axes([0.475,0.16,0.3,0.25])
    # PLot DR density
    x=np.linspace(-1,1,1000)
    ins.plot(x,1/w_scale**2*pdf5(x/w_scale),'tab:blue',label='$f_{\mathcal{P}}(w)$')
    # Plot nominal density
    ins.plot([-1,-w_scale,-w_scale, w_scale, w_scale,1],[0,0,1/(2*w_scale)**2,1/(2*w_scale)**2,0,0],'tab:orange',ls='-',label='$f_{\hat{\mathcal{P}}}(w)$')
    #ins.plot([-1,-w_scale,-w_scale, w_scale, w_scale,1],2**nx*np.array([0,0,1/(2*w_scale)**2,1/(2*w_scale)**2,0,0]),'tab:orange',ls='--',label='$M_{RVD} f_{\hat{\mathcal{P}}}(w)$')
    ins.set_xlim([-0.3,0.3])
    ins.set_xlabel('$w_i$')
    ins.set_ylabel('Density')
    ins.legend(loc=[0.9,0.3])
    ax.legend()
    fig.tight_layout()
    fig.savefig('violation_density.pdf')
