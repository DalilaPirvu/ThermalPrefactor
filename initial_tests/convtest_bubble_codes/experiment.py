import os, sys
sys.path.append('/home/dpirvu/project/prefactor/convtest_bubbles_codes')
from bubble_tools import *
from plotting import *

### Lattice Params
nLat = 2048
lenLat = 64.
phieq  = 0.
dx     = lenLat/nLat
dk     = 2.*np.pi/lenLat
knyq   = nLat//2+1
kspec  = knyq * dk
dtout  = dx
lightc = dx/dtout

# Lattice
lattice = np.arange(nLat)
xlist   = lattice*dx
klist   = np.roll((lattice - nLat//2+1)*dk, nLat//2+1)

#### SPECTRA
w2    = lambda m2: m2 + (2./dx**2.) * (1. - np.cos(klist * dx) )
w2std = lambda m2: m2 + klist**2.
pspec = lambda te,m2: np.array([te / lenLat / w2(m2)[k] if kk!=0. and kk < kspec else 0. for k,kk in enumerate(klist)])
stdev = lambda te,m2: np.sqrt(np.sum(pspec(te,m2)))

def get_general_model(case='minus'):
    if case=='minus':
        nTimeMAX = 51200
        tempList = np.array([0.1, 0.2])
        minSim, maxSim = 0, [1,10]
        massq  = lambda te: 1. - te*3./2.

        V     = lambda x:   0.5*x**2. - 0.25*x**4.
        Vinv  = lambda x: - 0.5*x**2. + 0.25*x**4.
        dV    = lambda x:       x     -      x**3.

        alphaList = [5., 10., 15., 20., 25., 30.]

    right_Vmax = sco.minimize_scalar(Vinv, bounds=(0., 2.), method='bounded')
    right_Vmax = right_Vmax.x
    return tempList, massq, right_Vmax, V, dV, Vinv, nTimeMAX, minSim, maxSim, alphaList

def get_model(tempList, massq, right_Vmax, V, dV, Vinv, nTimeMAX, minSim, maxSim, alphaList, tmp, case='free'):
    temp = tempList[tmp]
    maxSim = maxSim[tmp]
    m2 = massq(temp)
    sigmafld = stdev(temp, m2)
    return temp, m2, sigmafld, maxSim


### Paths to files
root_dir      = '/gpfs/dpirvu/prefactor/'
batch_params  = lambda nL,m2,al,te: 'x'+str(int(nL))+'_alpha'+str('%.4f'%al)+'_m2eff'+str('%.4f'%m2)+'_T'+str('%.4f'%te) 
triage_pref   = lambda minS,maxS,nTM: '_minSim'+str(minS)+'_maxSim'+str(maxS)+'_up_to_nTMax'+str(nTM)
sims_interval = lambda minS,maxS: '_minSim'+str(minS)+'_maxSim'+str(maxS)

sim_location       = lambda nL,m2,al,te,sim: root_dir + batch_params(nL,m2,al,te) + '_sim'      +str(sim)+'_fields.dat'
clean_sim_location = lambda nL,m2,al,te,sim: root_dir + batch_params(nL,m2,al,te) + '_clean_sim'+str(sim)+'_fields.npy'
rest_sim_location  = lambda nL,m2,al,te,sim: root_dir + batch_params(nL,m2,al,te) + '_rest_sim' +str(sim)+'_fields.npy'

directions_file = lambda nL,m2,al,te: root_dir + batch_params(nL,m2,al,te) + '_directions.npy'
velocities_file = lambda nL,m2,al,te: root_dir + batch_params(nL,m2,al,te) + '_velocitiesCOM.npy'
average_file    = lambda nL,m2,al,te: root_dir + batch_params(nL,m2,al,te) + '_average_bubble.npy'

sims_decayed_file   = lambda nL,m2,al,te,minS,maxS,nTM: root_dir + batch_params(nL,m2,al,te) + triage_pref(minS,maxS,nTM) + '_sims_decayed.npy'
sims_notdecayed_file= lambda nL,m2,al,te,minS,maxS,nTM: root_dir + batch_params(nL,m2,al,te) + triage_pref(minS,maxS,nTM) + '_sims_notdecayed.npy'
decay_times_file    = lambda nL,m2,al,te,minS,maxS,nTM: root_dir + batch_params(nL,m2,al,te) + triage_pref(minS,maxS,nTM) + '_timedecays.npy'
init_cond_file      = lambda nL,m2,al,te,minS,maxS,nTM: root_dir + batch_params(nL,m2,al,te) + triage_pref(minS,maxS,nTM) + '_initconds.npy'

powspec_tlist_file= lambda nL,m2,al,te,minS,maxS: root_dir + batch_params(nL,m2,al,te) + sims_interval(minS,maxS) + '_powspec.npy'
varians_tlist_file= lambda nL,m2,al,te,minS,maxS: root_dir + batch_params(nL,m2,al,te) + sims_interval(minS,maxS) + '_variances.npy'
stdinit_file      = lambda nL,m2,al,te,minS,maxS: root_dir + batch_params(nL,m2,al,te) + sims_interval(minS,maxS) + '_fld_init_std.npy'
emt_tlist_file    = lambda nL,m2,al,te,minS,maxS: root_dir + batch_params(nL,m2,al,te) + sims_interval(minS,maxS) + '_emt.npy'
stdemt0_tlist_file= lambda nL,m2,al,te,minS,maxS: root_dir + batch_params(nL,m2,al,te) + sims_interval(minS,maxS) + '_stdemt0.npy'
toten_tlist_file  = lambda nL,m2,al,te,minS,maxS: root_dir + batch_params(nL,m2,al,te) + sims_interval(minS,maxS) + '_toten.npy'

instanton_file = lambda nL,m2,al,te: root_dir + batch_params(nL,m2,al,te) + '_instanton_profile.npy'
crittimes_file = lambda nL,m2,al,te: root_dir + batch_params(nL,m2,al,te) + '_critical_timeslice.npy'
critenerg_file = lambda nL,m2,al,te: root_dir + batch_params(nL,m2,al,te) + '_critical_energy.npy'

# field, momentum
normal = np.array([phieq, 0., 0.])