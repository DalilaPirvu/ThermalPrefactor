import os,sys
sys.path.append('/home/dpirvu/python_stuff/')
sys.path.append('/home/dpirvu/project/prefactor/plus_bubbles_codes')

from plotting import *
from bubble_tools import *
from experiment import *

get_energy  = True
get_powspec = True

get_partial_stats = True
get_all_stats = True

case = 'plus'
tmp=2

general = get_general_model(case)
tempList, massq, right_Vmax, V, dV, Vinv, nTimeMAX, minSim, maxSim = general
temp, m2, sigmafld = get_model(*general, tmp, case)
exp_params = np.asarray([nLat, m2, temp])
print('General params', tempList, right_Vmax, nTimeMAX, minSim, maxSim)
print('Experiment', exp_params)

aa=0
div=1

simList = np.array(np.linspace(minSim, maxSim, div+1), dtype='int')
divdata = np.array([(ii,jj) for ii,jj in zip(simList[:-1], simList[1:])])

if get_partial_stats:
    asim, bsim = divdata[aa]

    TMAX = 80

    TEM_data = np.zeros((bsim-asim, TMAX))  
    EMT_data = np.zeros((bsim-asim, TMAX))  
    PS_data  = np.zeros((bsim-asim, 2, TMAX, nLat))
    TEM_data[:], EMT_data[:], PS_data[:] = 'nan', 'nan', 'nan'

    for sind, sim in enumerate(np.arange(asim, bsim)):
        path2sim      = sim_location(*exp_params, sim)
        path2CLEANsim = clean_sim_location(*exp_params, sim)
        if not os.path.exists(path2CLEANsim): continue

        tdecay, seeds, real, outcome = get_realisation(nLat, sim, phieq, path2sim)
        nC, nT, nN = np.shape(real)
        if nT>TMAX:
            real = real[:,:TMAX]
            cds = np.arange(TMAX)
        else:
            cds = np.arange(nT)
        nC, nT, nN    = np.shape(real)
        fld, mom, grd = real[0], real[1], real[2]
        print('Simulation, duration, cds:', sim, nC, nN, nT)

        if get_energy:
            TEM_data[sind, cds] = np.sum(0.5*mom**2. + 0.5*grd**2. + V(fld), axis=-1)
            EMT_data[sind, cds] = np.sum(mom * grd, axis=-1)

        if get_powspec:
            fftfld, fftmom = np.empty((2, TMAX, nLat))
            fftfld[:], fftmom[:] = 'nan', 'nan'
            fftfld[cds, :] = np.abs(np.fft.fft(fld, axis=-1)/nLat)**2.
            fftmom[cds, :] = np.abs(np.fft.fft(mom, axis=-1)/nLat)**2.
            fftfld[cds, 0], fftmom[cds, 0] = 0., 0. # subtracting the mean
            PS_data[sind, 0] = fftfld
            PS_data[sind, 1] = fftmom

    if get_energy:
        np.save(toten_tlist_file(*exp_params, asim, bsim), TEM_data)
        np.save(emt_tlist_file(*exp_params, asim, bsim), EMT_data)

    if get_powspec:
        np.save(powspec_tlist_file(*exp_params, asim, bsim), PS_data)

# once finished, merge partial lists
if get_all_stats:
    ALL_TEM_data = np.load(toten_tlist_file(*exp_params, *divdata[0]))
    ALL_EMT_data = np.load(emt_tlist_file(*exp_params, *divdata[0]))
    ALL_PS_data  = np.load(powspec_tlist_file(*exp_params, *divdata[0]))

    for inds in divdata[1:]:
        print(inds)
        ALL_TEM_data = np.concatenate((ALL_TEM_data, np.load(toten_tlist_file(*exp_params, *inds)))  , axis=0)
        ALL_EMT_data = np.concatenate((ALL_EMT_data, np.load(emt_tlist_file(*exp_params, *inds)))    , axis=0)
        ALL_PS_data  = np.concatenate((ALL_PS_data,  np.load(powspec_tlist_file(*exp_params, *inds))), axis=0)

    np.save(toten_tlist_file(*exp_params, minSim, maxSim)  , ALL_TEM_data)
    np.save(emt_tlist_file(*exp_params, minSim, maxSim)    , ALL_EMT_data)
    np.save(powspec_tlist_file(*exp_params, minSim, maxSim), ALL_PS_data)

    if False:
        # optionally remove partial lists
        for inds in divdata:
            os.remove(toten_tlist_file(*exp_params, *inds))
            os.remove(emt_tlist_file(*exp_params, *inds))
            os.remove(powspec_tlist_file(*exp_params, *inds))

print('All Done.')