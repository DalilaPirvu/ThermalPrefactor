from plotting import *
from bubble_tools import *

### Params
nLat     = 4096
nTimeMAX = 262144

tempList = np.array([0.15])

temp = tempList[0]
lenLat = 100.
m2     = 1. - temp*3./2.
phieq  = 0.

dx     = lenLat/(nLat-1)
dk     = 2.*np.pi/lenLat
knyq   = nLat//2+1
dtout  = dx
lightc = dx/dtout

# Lattice
lattice = np.arange(nLat)
xlist   = lattice*dx
klist   = np.roll((lattice - nLat//2+1)*dk, nLat//2+1)

#### SPECTRA
norm = 1./np.sqrt(nLat-1)
w2 = lambda m2: m2 + (2./dx**2.) * (1. - np.cos(klist * dx))
w2standard = lambda m2: m2 + klist

eigenbasis  = lambda te,m2: np.array([norm * np.sqrt(te/dx) / np.sqrt(w2(m2)[k]) if kk!=0. else 0. for k,kk in enumerate(klist)])
pspec       = lambda te,m2: np.abs(eigenbasis(te,m2))**2.
fluct_stdev = lambda te,m2: np.sqrt(np.sum(pspec(te,m2)))

### POTENTIAL
Vfree = lambda x:   0.5*x**2.
Vquar = lambda x:   0.5*x**2. + 0.25*x**4.
V     = lambda x:   0.5*x**2. - 0.25*x**4.
Vinv  = lambda x: - 0.5*x**2. + 0.25*x**4.
dV    = lambda x:       x     -      x**3.

### Paths to files
root_dir      = '/gpfs/dpirvu/prefactor/'
batch_params  = lambda nL,m2,te: 'x'+str(int(nL))+'_m2eff'+str('%.4f'%m2)+'_T'+str('%.4f'%te) 
triage_pref   = lambda minS,maxS,nTM: '_minSim'+str(minS)+'_maxSim'+str(maxS)+'_up_to_nTMax'+str(nTM)
sims_interval = lambda minS,maxS: '_minSim'+str(minS)+'_maxSim'+str(maxS)

sim_location       = lambda nL,m2,te,sim: root_dir + batch_params(nL,m2,te) + '_sim'+str(sim)+'_fields.dat'
clean_sim_location = lambda nL,m2,te,sim: root_dir + batch_params(nL,m2,te) + '_clean_sim'+str(sim)+'_fields.npy'
rest_sim_location  = lambda nL,m2,te,sim: root_dir + batch_params(nL,m2,te) + '_rest_sim'+str(sim)+'_fields.npy'

directions_file = lambda nL,m2,te: root_dir + batch_params(nL,m2,te) + '_directions.npy'
velocities_file = lambda nL,m2,te: root_dir + batch_params(nL,m2,te) + '_velocitiesCOM.npy'
average_file    = lambda nL,m2,te: root_dir + batch_params(nL,m2,te) + '_average_bubble.npy'

sims_decayed_file   = lambda nL,m2,te,minS,maxS,nTM: root_dir + batch_params(nL,m2,te) + triage_pref(minS,maxS,nTM) + '_sims_decayed.npy'
sims_notdecayed_file= lambda nL,m2,te,minS,maxS,nTM: root_dir + batch_params(nL,m2,te) + triage_pref(minS,maxS,nTM) + '_sims_notdecayed.npy'
decay_times_file    = lambda nL,m2,te,minS,maxS,nTM: root_dir + batch_params(nL,m2,te) + triage_pref(minS,maxS,nTM) + '_timedecays.npy'
init_cond_file      = lambda nL,m2,te,minS,maxS,nTM: root_dir + batch_params(nL,m2,te) + triage_pref(minS,maxS,nTM) + '_initconds.npy'

powspec_tlist_file= lambda nL,m2,te,minS,maxS: root_dir + batch_params(nL,m2,te) + sims_interval(minS,maxS) + '_powspec.npy'
varians_tlist_file= lambda nL,m2,te,minS,maxS: root_dir + batch_params(nL,m2,te) + sims_interval(minS,maxS) + '_variances.npy'
stdinit_file      = lambda nL,m2,te,minS,maxS: root_dir + batch_params(nL,m2,te) + sims_interval(minS,maxS) + '_fld_init_std.npy'
emt_tlist_file    = lambda nL,m2,te,minS,maxS: root_dir + batch_params(nL,m2,te) + sims_interval(minS,maxS) + '_emt.npy'
stdemt0_tlist_file= lambda nL,m2,te,minS,maxS: root_dir + batch_params(nL,m2,te) + sims_interval(minS,maxS) + '_stdemt0.npy'
toten_tlist_file  = lambda nL,m2,te,minS,maxS: root_dir + batch_params(nL,m2,te) + sims_interval(minS,maxS) + '_toten.npy'

instanton_file = lambda nL,m2,te: root_dir + batch_params(nL,m2,te) + '_instanton_profile.npy'
crittimes_file = lambda nL,m2,te: root_dir + batch_params(nL,m2,te) + '_critical_timeslice.npy'
critenerg_file = lambda nL,m2,te: root_dir + batch_params(nL,m2,te) + '_critical_energy.npy'

# field, momentum
normal = np.array([phieq, 0., 0.])

def get_model(tmp):
    if tmp == 0: minSim, maxSim = 0, 1000
    elif tmp == 1: minSim, maxSim = 0, 10
    else:
        return 'Experiment not implemented.'

    temp = tempList[tmp]
    sigmafld = fluct_stdev(temp, m2)

    right_Vmax = sco.minimize_scalar(Vinv, bounds=(0., 2.), method='bounded')
    right_Vmax = right_Vmax.x
    return temp, sigmafld, minSim, maxSim, right_Vmax