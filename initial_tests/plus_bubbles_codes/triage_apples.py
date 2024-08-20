import os,sys
sys.path.append('/home/dpirvu/python_stuff/')
sys.path.append('/home/dpirvu/project/prefactor/free_bubbles_codes')

from plotting import *
from bubble_tools import *
from experiment import *

# Classify decays
get_partial_stats = True
get_all_stats = True

case = 'plus'
tmp=0

general = get_general_model(case)
tempList, massq, right_Vmax, V, dV, Vinv, nTimeMAX, minSim, maxSim = general
temp, m2, sigmafld = get_model(*general, tmp, case='free')
exp_params = np.array([nLat, m2, temp])
print('General params', tempList, right_Vmax, nTimeMAX, minSim, maxSim)
print('Experiment', exp_params)

if get_partial_stats:
    aa=0
    div=1

    simList = np.array(np.linspace(minSim, maxSim, div+1), dtype='int')
    divdata = np.array([(ii,jj) for ii,jj in zip(simList[:-1], simList[1:])])
    asim, bsim = divdata[aa]

    for sim in np.arange(asim, bsim):
        path2sim = sim_location(*exp_params, sim)
        path2CLEANsim = clean_sim_location(*exp_params, sim)

        if os.path.exists(path2sim):
            tdecay, seeds, real, outcome = get_realisation(nLat, sim, nTimeMAX, path2sim)
            if outcome!=2: real = centre_bubble(real)

            print('Simulation', sim, ', outcome', outcome, ', tdecay:', tdecay, np.shape(seeds), np.shape(real))
            np.save(path2CLEANsim, np.array([sim, tdecay, outcome, seeds, real]))

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
