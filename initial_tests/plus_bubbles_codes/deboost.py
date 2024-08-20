import os,sys
sys.path.append('/home/dpirvu/project/prefactor/plus_bubbles_codes')
from plotting import *
from bubble_tools import *
from experiment import *

# Classify decays
tmp=0
gcl, temp, sigmafld, minSim, maxSim, right_Vmax = get_model(tmp)
exp_params = np.asarray([nLat, gcl, m2, temp])
print('Experiment', exp_params)

aa=0
div=4

decay_times = np.load(decay_times_file(*exp_params, minSim, maxSim, nTimeMAX))
done_sims   = np.array([sim for sim in decay_times[:,0] if os.path.exists(rest_sim_location(*exp_params, sim))])
decay_times = np.array([decay_times[sind] for sind, ss in enumerate(decay_times[:,0]) if ss not in done_sims])

minDecTime  = nLat
alltimes    = decay_times[:,1]
simList2Do  = decay_times[alltimes>=minDecTime, 0]
n2Do        = len(simList2Do)
print('N = ', n2Do,'simulations to deboost.')

ranges2Do   = np.array(np.linspace(0, n2Do, div+1), dtype='int')
if len(ranges2Do) > 1:
    divdata    = np.asarray([(ii,jj) for ii,jj in zip(ranges2Do[:-1], ranges2Do[1:])])
    asim, bsim = divdata[aa]
else:
    asim, bsim = 0, n2Do

ranList = simList2Do[asim : bsim]
random.shuffle(ranList)
print('Here we\'re deboosting the following sims:', asim, bsim, ranList)

threshm, threshM = right_Vmax.x + 1.*sigmafld, right_Vmax.x + 3.*sigmafld
ampList  = np.linspace(threshm, threshM, 10)

crit_rad = 50
winsize  = np.array(np.linspace(crit_rad*2, crit_rad*3, 5), dtype='int')
crit_thresh = right_Vmax.x + 2.*sigmafld

plots=False

print('Looking at at lambda, T, phi0, m2, sigma:', lamb, temp, phi0, m2(lamb), sigmafld)

for sim in ranList:
    print('Starting simulation, temp, lambda:', sim, temp, lamb)
    path2CLEANsim = clean_sim_location(*exp_params, sim)
    fullreal, sim, tdecay, outcome = np.load(path2CLEANsim)

    fullreal = fullreal[:,-nLat:-nLat//4,nLat//4:-nLat//4] # this is to speed up the boosting
    
    bubble = fullreal[:1]
    bool, vellist = True, []
    try:
        beta = find_COM_vel(bubble, ampList, winsize, nLat, lightc, phieq, crit_thresh, crit_rad, 1., plots)
        if np.isnan(beta):
            print('Simulation, temp, lambda:', sim, temp, lamb, '. Dead end at step 0.')
            bool = False
    except:
        print('Some error within first vCOM detection. Skipped sim', sim)
        continue

    while np.abs(beta) >= 0.03:
        roundlist = np.array([round(vv, 2) for vv in vellist])
        if round(beta, 2) in roundlist:
            beta = beta/2.
            if np.abs(beta) <= 0.015: break
        elif len(vellist)!=0 and np.abs(beta) > 0.1:
            if np.sign(beta) != np.sign(vellist[-1]):
                beta = beta/2.
                if np.abs(beta) <= 0.015: break

        vellist.append(beta)
        try:
            bubble = boost_bubble(bubble, nLat, lightc, phieq, beta, crit_thresh, crit_rad, normal)
            beta = find_COM_vel(bubble, ampList, winsize, nLat, lightc, phieq, crit_thresh, crit_rad, 1., plots)
            if np.isnan(beta):
                print('Simulation, temp, lambda:', sim, temp, lamb, 'skipped. Dead end.')
                bool = False
                break
        except:
            print('Some error with deboost / finding the velocity. Skipped sim', sim)
            bool = False
            break

    if bool:
        print('Simulation, temp, lambda:', sim, temp, lamb, 'doing final step.')
        vellist.append(beta)
        totbeta   = get_totvel_from_list(vellist)

        finalreal = fullreal[:1]
        finalreal = boost_bubble(finalreal, nLat, lightc, phieq, totbeta, crit_thresh, crit_rad, normal)
        finalbeta = find_COM_vel(finalreal, ampList, winsize, nLat, lightc, phieq, crit_thresh, crit_rad, 1., plots)        

        endbeta  = get_totvel_from_list([totbeta, finalbeta])

        fullreal = boost_bubble(fullreal, nLat, lightc, phieq, endbeta, crit_thresh, crit_rad, normal)
        fullreal = space_save(fullreal, lightc, phieq, crit_thresh, crit_rad, nLat)

        if np.abs(finalbeta) < 0.1:
            path2RESTsim = rest_sim_location(*exp_params, sim)
            np.save(path2RESTsim, np.array([sim, fullreal, endbeta]))
            print('Saved. Total final velocity, remeasured velocity after initial deboosting, before final check, vellist:', endbeta, finalbeta, totbeta, vellist)
        else:
            print('Discarded as inconsistent.')

            
# Centralize all total velocities. Compute average bubbles.
getvels = False
getavs  = False
considerMaxVel = 0.9 # excludes faster bubbles from average

for tmp in range(len(tempList)):
    phi0, temp, lamb, sigmafld, minSim, maxSim, right_Vmax = get_model(tmp)
    exp_params = np.array([nLat, lamb, phi0, temp])

    crit_thresh = right_Vmax.x+2.*sigmafld
    win = 300
    critSize = 20

    if getvels:
        ALLvels = []
        for sim in range(minSim, maxSim):
            path2RESTsim = rest_sim_location(*exp_params, sim)
            if os.path.exists(path2RESTsim):
                sim, real, totalvCOM, finalv = np.load(path2RESTsim)
                ALLvels.append([sim, totalvCOM])

        np.save(velocities_file(*exp_params), ALLvels)
        print('Velocities saved.', exp_params)

    if getavs:
        ALLatrest = []
        for sim in range(minSim, maxSim):      
            path2RESTsim = rest_sim_location(*exp_params, sim)
            if os.path.exists(path2RESTsim):
                sim, real, totalvCOM, finalv = np.load(path2RESTsim)
                if np.abs(totalvCOM) < considerMaxVel:
                    ALLatrest.append([sim, real])

        stacks  = stack_bubbles(ALLatrest, win, phieq, crit_thresh, critSize)
        stacks  = average_stacks(stacks, normal)
        avstack = average_bubble_stacks(stacks)
        np.save(average_file(*exp_params), avstack)
        print('Average bubble saved.', exp_params)

print('All Done.')
