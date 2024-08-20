import os,sys
sys.path.append('/home/dpirvu/python_stuff/')
sys.path.append('/home/dpirvu/project/prefactor/bubbles_codes')

from plotting import *
from bubble_tools import *
from experiment import *

# Classify decays
get_partial_stats = True
get_all_stats = False

tmp=0
temp, sigmafld, minSim, maxSim, right_Vmax = get_model(tmp)
exp_params = np.asarray([nLat, m2, temp])
print('Experiment', exp_params)

undecayed = []
if get_partial_stats:
    aa=0
    div=10

    simList = np.array(np.linspace(minSim, maxSim, div+1), dtype='int')
    divdata = np.array([(ii,jj) for ii,jj in zip(simList[:-1], simList[1:])])
    asim, bsim = divdata[aa]

    for sim in np.arange(asim, bsim):
        path2sim = sim_location(*exp_params, sim)
        path2CLEANsim = clean_sim_location(*exp_params, sim)

        if os.path.exists(path2sim):
            tdecay, seeds, real, outcome = get_realisation(nLat, sim, nTimeMAX, path2sim)
            real = centre_bubble(real)
            np.save(path2CLEANsim, [sim, tdecay, outcome, seeds, real])
            print('Simulation', sim, ', outcome', outcome, ', tdecay:', tdecay)
        else:
            undecayed.append(sim)

print('Undecayed simulations', undecayed)

# Code below centralizes all results. Smaller files load faster.
if get_all_stats:
    outcomes_sims, decay_times, init_conditions = [], [], []
    for sim in range(minSim, maxSim):
        path2CLEANsim = clean_sim_location(*exp_params, sim)
        if os.path.exists(path2CLEANsim):
            sim, tdecay, outcome, seeds, real = np.load(path2CLEANsim, allow_pickle=True)

            fftfld = np.abs(np.fft.fft(seeds[0,:], axis=-1)/nLat)**2.
            fftmom = np.abs(np.fft.fft(seeds[1,:], axis=-1)/nLat)**2.
            fftgrd = np.abs(np.fft.fft(seeds[2,:], axis=-1)/nLat)**2.

            outcomes_sims.append([sim, outcome])
            decay_times.append([sim, tdecay])
            init_conditions.append([sim, seeds[0,:], seeds[1,:], seeds[2,:], fftfld, fftmom, fftgrd])
            print(sim, outcome, tdecay)

    np.save(sims_decayed_file(*exp_params, minSim, maxSim, nTimeMAX), outcomes_sims)
    np.save(decay_times_file(*exp_params, minSim, maxSim, nTimeMAX), decay_times)
    np.save(init_cond_file(*exp_params, minSim, maxSim, nTimeMAX), init_conditions)
    print('All saved.')

print('All Done.')
