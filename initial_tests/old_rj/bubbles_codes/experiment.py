from plotting import *
from bubble_tools import *

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

### Params
nLat     = 1024
nTimeMAX = 3*nLat
bytesUNDECAYED = 248512797

gclList  = np.array([0.3]) # for free field g=0.1
tempList = np.array([10.])

#### CONSTANTS
lenLat = 5.
m2 = 1.

phieq  = 0.
alph   = 16.

dx     = lenLat/nLat
dk     = 2.*np.pi/lenLat
knyq   = nLat//2+1

dt     = dx/alph
dtout  = dt*alph
lightc = dx/dtout

dxbar = lambda g: dx / g
dtbar = lambda g: dtout / g

# Lattice
lattice = np.arange(nLat)
xlist   = lattice*dx
klist   = np.roll((lattice - nLat//2+1)*dk, nLat//2+1)

inv_phases = np.exp(1j*np.outer(xlist, klist))
dit_phases = nLat**-1. * np.exp(-1j*np.outer(xlist, klist))

#### SPECTRA
norm = (2. * lenLat)**-0.5
w2 = lambda tmp, m2: klist**2. + m2

eigenbasis  = lambda tmp,m2,g,te: np.array([norm/(w2(tmp, m2)[k]**0.5) * (2.*(g**2.)*te)**0.5 if kk!=0. else 0. for k,kk in enumerate(klist)])
pspec       = lambda tmp,m2,g,te: np.abs(eigenbasis(tmp, m2, g, te))**2.
fluct_stdev = lambda tmp,m2,g,te: np.sqrt(np.sum(pspec(tmp, m2, g, te)))

### POTENTIAL
Vfree = lambda x,g:   0.5*x**2.# + 0.25*x**4.
V     = lambda x,g:   0.5*x**2. - 0.25*x**4.
Vinv  = lambda x,g: - 0.5*x**2. + 0.25*x**4.
dV    = lambda x,g:   x - x**3.

### Paths to files
root_dir      = '/gpfs/dpirvu/prefactor/'
batch_params  = lambda nL,g,m2,te: 'x'+str(int(nL))+'_g'+str('%.4f'%g)+'_m2'+str('%.4f'%m2)+'_T'+str('%.4f'%te) 
triage_pref   = lambda minS,maxS,nTM: '_minSim'+str(minS)+'_maxSim'+str(maxS)+'_up_to_nTMax'+str(nTM)
sims_interval = lambda minS,maxS: '_minSim'+str(minS)+'_maxSim'+str(maxS)

sim_location       = lambda nL,g,m2,te,sim: root_dir + batch_params(nL,g,m2,te) + '_sim'+str(sim)+'_fields.dat'
clean_sim_location = lambda nL,g,m2,te,sim: root_dir + batch_params(nL,g,m2,te) + '_clean_sim'+str(sim)+'_fields.npy'
rest_sim_location  = lambda nL,g,m2,te,sim: root_dir + batch_params(nL,g,m2,te) + '_rest_sim'+str(sim)+'_fields.npy'

directions_file = lambda nL,g,m2,te: root_dir + batch_params(nL,g,m2,te) + '_directions.npy'
velocities_file = lambda nL,g,m2,te: root_dir + batch_params(nL,g,m2,te) + '_velocitiesCOM.npy'
average_file    = lambda nL,g,m2,te: root_dir + batch_params(nL,g,m2,te) + '_average_bubble.npy'

sims_decayed_file    = lambda nL,g,m2,te,minS,maxS,nTM: root_dir + batch_params(nL,g,m2,te) + triage_pref(minS,maxS,nTM) + '_sims_decayed.npy'
sims_notdecayed_file = lambda nL,g,m2,te,minS,maxS,nTM: root_dir + batch_params(nL,g,m2,te) + triage_pref(minS,maxS,nTM) + '_sims_notdecayed.npy'
decay_times_file     = lambda nL,g,m2,te,minS,maxS,nTM: root_dir + batch_params(nL,g,m2,te) + triage_pref(minS,maxS,nTM) + '_timedecays.npy'

powspec_tlist_file = lambda nL,g,m2,te,minS,maxS: root_dir + batch_params(nL,g,m2,te) + sims_interval(minS,maxS) + '_powspec.npy'
varians_tlist_file = lambda nL,g,m2,te,minS,maxS: root_dir + batch_params(nL,g,m2,te) + sims_interval(minS,maxS) + '_variances.npy'
stdinit_file       = lambda nL,g,m2,te,minS,maxS: root_dir + batch_params(nL,g,m2,te) + sims_interval(minS,maxS) + '_fld_init_std.npy'
emt_tlist_file     = lambda nL,g,m2,te,minS,maxS: root_dir + batch_params(nL,g,m2,te) + sims_interval(minS,maxS) + '_emt.npy'
stdemt0_tlist_file = lambda nL,g,m2,te,minS,maxS: root_dir + batch_params(nL,g,m2,te) + sims_interval(minS,maxS) + '_stdemt0.npy'
toten_tlist_file   = lambda nL,g,m2,te,minS,maxS: root_dir + batch_params(nL,g,m2,te) + sims_interval(minS,maxS) + '_toten.npy'

instanton_file = lambda nL,g,m2,te: root_dir + batch_params(nL,g,m2,te) + '_instanton_profile.npy'
crittimes_file = lambda nL,g,m2,te: root_dir + batch_params(nL,g,m2,te) + '_critical_timeslice.npy'
critenerg_file = lambda nL,g,m2,te: root_dir + batch_params(nL,g,m2,te) + '_critical_energy.npy'

labl = lambda g,te: r'$g={:.1f}, T={:.1f}$'.format(g, te)

# Important: standard order or columns in .dat files is:
# field, momentum, gradient field squared
normal = np.array([phieq, 0., 0.])

def get_model(tmp):
    if tmp == 0: minSim, maxSim = 0, 5000
    elif tmp == 1: minSim, maxSim = 0, 100
    else:
        return 'Experiment not implemented.'

    gg  = gclList[tmp]
    temp = tempList[tmp]
    sigmafld = fluct_stdev(tmp, m2, gg, temp)

    #right_Vmax = 0.

    right_Vmax = sco.minimize_scalar(Vinv, args=gg, bounds=(0., 2.), method='bounded')
    right_Vmax = right_Vmax.x
    return gg, temp, sigmafld, minSim, maxSim, right_Vmax