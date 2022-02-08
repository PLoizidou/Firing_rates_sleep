import bk.load
import bk.plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import neuroseries as nts

from scipy.stats import spearmanr,pearsonr, wilcoxon, zscore, ttest_ind

import os
import fun
import re
from itertools import chain

paths = pd.read_csv('Z:/All-Rats/Billel/session_indexing.csv',sep = ';')['Path']

# Importing pre and post RUN sleeps
sleep_pre=np.load('C://Users//Panagiota.Loizidou//Desktop//Amy-Hpc-sleep-dynamics-python-master//REM project//Sleeps_epochs.npy', allow_pickle=True)
sleep_post=np.load('C://Users//Panagiota.Loizidou//Desktop//Amy-Hpc-sleep-dynamics-python-master//REM project//Sleeps_epochs_post.npy', allow_pickle=True)
sleep=np.concatenate((sleep_pre,sleep_post), axis=0)


def flatten(t):
    return [item for sublist in t for item in sublist]

def extract(lst, i,e):
    return [item[i:e] for item in lst]


def parsing_sleep(path, pre_RUN=True):
    """
    input: the path of session you want to analyze
    ->  all states in chronological order in the pre run period 
        all states in chronological order in the post run period 
        sleeps sessions within the desgnated period (pre or post run) that are separated by more than
        30 seconds of wakefullness  
    """
    #loading data
    bk.load.current_session(path)
    states=bk.load.states()
    pre, post = bk.load.sleep()
    
    #turning them into a pd.DataFrame
    wake=np.insert(np.array(states['wake'],dtype='object'),2,'wake', axis=1)
    nrem=np.insert(np.array(states['sws'],dtype='object'), 2, 'nrem', axis=1)
    rem=np.insert(np.array(states['Rem'],dtype='object'), 2, 'rem', axis=1)
    drowsy=np.insert(np.array(states['drowsy'],dtype='object'), 2, 'drowsy', axis=1)
    whole_session=np.concatenate((wake, nrem, rem, drowsy))
    whole_session=pd.DataFrame(whole_session, columns=['start', 'stop', 'state'])
    whole_session_sorted=whole_session.sort_values('start', ignore_index=True)
    
    # separating pre and post RUN
    pre_run_period = whole_session_sorted.loc[whole_session_sorted['start']<int(pre['end'])]
    post_run_period = whole_session_sorted.loc[whole_session_sorted['start']>int(post['start'])]
    
    # reseting indexing for the post run sleep session
    post_run_period=post_run_period.reset_index(drop=True) 
    index=np.arange(0, len(post_run_period), 1)
    post_run_period=post_run_period.reindex(index)
    
    # chosing which period to parse
    if pre_RUN:
        l=pre_run_period
    else:
        l=post_run_period
       
    p=l[(((l['stop']-l['start'])/1e6)>30) & (l['state']=='wake')]
    k=[-1]+p.index.values.tolist()+[-1]    # list containing indexes of epochs of wake>30 s
    
    sleeps = [l.iloc[k[n]+1:k[n+1]] for n in range(len(k)-1)]
    sleeps=[[i] for i in sleeps if len(i)>0] #dropping empty periods
        
    return pre_run_period,post_run_period, sleeps

def get_duration(sleep, unit='s'):
    """
    input: only a specific sleep session (i.e. sleeps[0]) can be imported
    """
    session=nts.IntervalSet(start=sleep[0]['start'].iloc[0], end=sleep[0]['stop'].iloc[-1])
    return session.tot_length(unit)

def removing_short_sessions(path, pre_RUN=True, min_dur=30, path_list=False):
    '''
    min_dur: select how long the extended sleep should be (in minutes). 
    path_list returns a list containing n times the path where n is the number of long sleeps. 
    Used later for labeling purposes 
    -> weird dataframe (np.ndarray of np.ndarray of pd.DF). Might be easier to access it using [0]
    '''
    _,_, sleeps=parsing_sleep(path, pre_RUN)
    sleeps=np.array(sleeps, dtype=object)
    index_list=[]
    for i in range(len(sleeps)):
        if get_duration(sleeps[i])> (min_dur*60):  #*60 to make it into minutes 
            index_list.append(i)
            
    long_sleeps=sleeps[index_list]
    if path_list:
        paths=[path]
        paths=len(long_sleeps)*paths
        return long_sleeps, paths
    else:
        return long_sleeps
    

def get_table(path, pre_RUN=True, min_dur=30):
    """
    -> table with average firing rates for each state lasting longer than 30 seconds of the specified session
    """
    
    session, path, rat, day,n_channels=bk.load.current_session(path, return_vars=True)
    neurons,metadata = bk.load.loadSpikeData(bk.load.path)
    
    long_sleeps=removing_short_sessions(path, pre_RUN, min_dur)
    durations=[]
    rem=[]
    starts=[]
    stops=[]
    states=[]
    mean_FRs=np.zeros(len(neurons))
    BLA_Pyr=[]
    BLA_Int=[]
    Hpc_Pyr=[]
    Hpc_Int=[]
    mean_FR=[]
    index=[]
    path_s=[]
    
    for s in range(len(long_sleeps)):
        for i in range(len(long_sleeps[s][0])):
            epoch=nts.IntervalSet(start=long_sleeps[s][0].iloc[i].start, end=long_sleeps[s][0].iloc[i].stop)
            durations.append(epoch.tot_length('ms'))
            states.append(long_sleeps[s][0].iloc[i].state)
            starts.append(long_sleeps[s][0].iloc[i].start)
            stops.append(long_sleeps[s][0].iloc[i].stop)
            index.append(i)
            
            
            for n in range(len(neurons)):
                spk_time = neurons[n].restrict(epoch).as_units('s').index.values
                mean_firing_rate= len(spk_time)/epoch.tot_length('s')
                mean_FRs[n]=mean_firing_rate

            mean_FR.append(np.nanmean(mean_firing_rate))
            BLA_Pyr_n=np.nanmean(mean_FRs[(metadata.Region == 'BLA') & (metadata.Type == 'Pyr')])
            BLA_Int_n=np.nanmean(mean_FRs[(metadata.Region == 'BLA') & (metadata.Type == 'Int')])
            Hpc_Pyr_n=np.nanmean(mean_FRs[(metadata.Region == 'Hpc') & (metadata.Type == 'Pyr')])
            Hpc_Int_n=np.nanmean(mean_FRs[(metadata.Region == 'Hpc') & (metadata.Type == 'Int')])
            BLA_Pyr.append(float(BLA_Pyr_n))
            BLA_Int.append(float(BLA_Int_n))
            Hpc_Pyr.append(Hpc_Pyr_n)
            Hpc_Int.append(Hpc_Int_n)
    table=pd.DataFrame(np.column_stack([index, starts, stops, durations, states, mean_FR, BLA_Pyr,BLA_Int,Hpc_Pyr,Hpc_Int]), columns=['index', 'start', 'stop', 'duration', 'state', 'all_cells', 'BLA_Pyr','BLA_Int','Hpc_Pyr','Hpc_Int'])
    for col in table.columns:
        if col == 'state':
            pass
        else:
            table[col] = table[col].astype(float)
    
    #adding the path column (useful if anything weird is seen)
    table['path']=path
    return table

    
def separate_ES(path, pre_RUN=True, min_dur=30):
    """
    parses the table of a whole sleep (pre or post run) into 'extended sleep' tables
    *Not all sessions are supposed to have BLA and Hpc cells (because of recordings outside the BLA and Hpc)
    """
    
    sleep=get_table(path, pre_RUN, min_dur)
    individual_sessions=[]
    h=sleep[(sleep['index']==0.0)].index.values.tolist() + [len(sleep)+1]
    j = [sleep.iloc[h[n]:h[n+1]] for n in range(len(h)-1)]
    for u in j:
        individual_sessions.append(u)
    return individual_sessions

def separate_ES_multisession(paths=paths, pre_RUN=True, min_dur=30):
    """
    parses into 'extended sleep' for multiple sessions but always in the same period pre or post RUN
    """
    life_ruining=[] #add it to the return if you need to have it, but it is generally stable. 
                    #See list of sessions not used and reason why in Summer Internship-Part II
    sleep=[]
    for i in paths:
        try:
            sleep.append(get_table(i, pre_RUN, min_dur))
        except:
            life_ruining.append(i)
    
    individual_sessions=[]
    for s in range(len(sleep)):    
        h=sleep[s][(sleep[s]['index']==0.0)].index.values.tolist()+[-1]
        j = [sleep[s].iloc[h[n]:h[n+1]] for n in range(len(h)-1)]
        for u in j:
            individual_sessions.append(u)
    return individual_sessions

def get_first_last_avg_all_sessions(paths=paths, celltype='Pyr', region='Hpc', stage='nrem', pre_RUN=True, min_dur=30):    
    """
    -> list of tuples with the avg firing rate of the celltype specified during the first and last epochs of the stage specified in all ES. Includes only ES pre or post RUN
    """
    
    first_last_avg_all_sessions=[]
    
    # removing paths without any ES
    not_use=[]
    for i in range(len(paths)):
        try:
            ES=removing_short_sessions(paths[i], pre_RUN, min_dur)
        except:
            not_use.append(paths[i])  
    useful_paths=[ele for ele in paths if ele not in not_use]

    for p in range(len(useful_paths)):    
        # loading info about one session 
        bk.load.current_session(useful_paths[p])
        neurons,metadata = bk.load.loadSpikeData(bk.load.path)

        avg_first_lst = []
        avg_last_lst = []

        #selecting celltype of interest:
        neurons = neurons[(metadata.Type ==celltype) &(metadata.Region ==region)]

        long_sleeps=removing_short_sessions(useful_paths[p], pre_RUN, min_dur)  #going over all ES in useful sessions

        for n in range(len(long_sleeps)):    
            first_start = long_sleeps[n][0][long_sleeps[n][0]['state'] == stage].iloc[0]['start']/1e6
            first_stop = long_sleeps[n][0][long_sleeps[n][0]['state'] == stage].iloc[0]['stop']/1e6
            last_start = long_sleeps[n][0][long_sleeps[n][0]['state'] == stage].iloc[-1]['start']/1e6
            last_stop = long_sleeps[n][0][long_sleeps[n][0]['state'] == stage].iloc[-1]['stop']/1e6

            # defining the first and last epoch (REM or NREM) intervals
            first_epoch = nts.IntervalSet(start = [first_start], end = [first_stop], time_units = 's')
            last_epoch = nts.IntervalSet(start = [last_start], end = [last_stop], time_units = 's')

            # calculating the total spikes of all neurons of interest within each interval 
            total_spikes_first = total_spikes_last = 0
            for i in range(len(neurons)):
                total_spikes_first = total_spikes_first+len(neurons[i].restrict(first_epoch))
                total_spikes_last = total_spikes_last+len(neurons[i].restrict(last_epoch))


            # calculating the average firing rate per interval (total_spikes/(n_neurons*seconds)). If total is 0, the average is automatically set to 0.
            if total_spikes_first>0:
                avg_first = total_spikes_first/(len(neurons)*(first_stop-first_start))
            else:
                avg_first=0

            if total_spikes_last>0:
                avg_last = total_spikes_last/(len(neurons)*(first_stop-first_start))
            else:
                avg_last=0

            avg_first_lst.append(avg_first)
            avg_last_lst.append(avg_last)

        first_last_avg=list(zip(avg_first_lst,avg_last_lst))
        first_last_avg_all_sessions.extend(first_last_avg)
    
    return first_last_avg_all_sessions


def norm_FR(path, region='Hpc', celltype='Pyr', min=0, max=2040, bin=1, whole=True):
    """
    inputs:
    min, max, bin is in seconds
    if whole is True, the interval will be from 0 to the end of recording. It overwrites the min and max specified.
    -> Firing rate z scores for each cell of specified celltype at structure specified calculated using means and SDs in 1-min bins of the specified interval
    """
    
    bk.load.current_session(path)
    neurons,metadata = bk.load.loadSpikeData(bk.load.path)
    pre, post =bk.load.sleep()
    
    if whole:
        min=pre['start'].values/1e6   # from microseconds to seconds
        max=post['end'].values/1e6
    
    window = nts.IntervalSet(min,max,time_units = 's')
    
    n=[]
    for i in range(len(neurons)):
        n.append(neurons[i].restrict(window).as_units('s').index)
    n=sorted(n,key=len)     # sorted based on highest firing rate overall 
    n=np.array(n,dtype=object)
    
    bins=np.arange(min,max,bin)
    hist=[]
    for i in range(len(neurons)):
        j,e=np.histogram(n[i],bins)
        hist.append(j)
    z=zscore(hist, axis=1)
    
    fr_z=z[(metadata.Type==celltype) &(metadata.Region==region)]
    
    return fr_z, e


def norm_FR_ES(paths=paths, region='Hpc', celltype='Pyr', min=0, max=2040, bin=1, min_dur=30, pre_post=False):
    """
    pre_post: select True if you want the pre and post RUN Es to be separated
    -> array with population firing rate for specified celltype and brainregion for each ES
    """

    # removing paths without any ES both for pre and post RUN
    not_use_pre=[]
    for i in range(len(paths)):
        try:
            ES=removing_short_sessions(paths[i], pre_RUN=True, min_dur=min_dur)
            if ES[0]==0:
                not_use_pre.append(paths[i])  
        except:
            not_use_pre.append(paths[i])  
    useful_paths_pre=[ele for ele in paths if ele not in not_use_pre]
    # useful_paths_pre.remove("Z:\Rat11\Rat11-20150401")

    not_use_post=[]
    for i in range(len(paths)):
        try:
            ES=removing_short_sessions(paths[i], pre_RUN=False, min_dur=min_dur)
            if ES[0]==0:
                not_use_post.append(paths[i]) 
        except:
            not_use_post.append(paths[i])  
    useful_paths_post=[ele for ele in paths if ele not in not_use_post]

    # Normalizing within an ES
    all_ES_pre=[]
    all_ES_post=[]
    
    for i in range(len(useful_paths_pre)): # going over all sessions
        ES_pre=removing_short_sessions(useful_paths_pre[i])  #going over all ES in useful sessions
        for n in range(len(ES_pre)):
            start=ES_pre[n][0]['start'].iloc[0]/1e6
            stop=ES_pre[n][0]['stop'].iloc[-1]/1e6
            c, e=norm_FR(useful_paths_pre[i],region, celltype, min=start, max=stop, whole=False)
            mean_c=np.nanmean((c), axis=0)  #taking the average of all cells for each time bin
            all_ES_pre.append(mean_c)

    for i in range(len(useful_paths_post)):
        ES_post=removing_short_sessions(useful_paths_post[i], pre_RUN=False)
        for n in range(len(ES_post)):
            start=ES_post[n][0]['start'].iloc[0]/1e6
            stop=ES_post[n][0]['stop'].iloc[-1]/1e6
            c, e=norm_FR(useful_paths_post[i],region, celltype, min=start, max=stop, whole=False)
            mean_c=np.nanmean((c), axis=0)
            all_ES_post.append(mean_c)
    
    if pre_post:
        return all_ES_pre, all_ES_post

    all_ES=all_ES_pre+all_ES_post
    
    return all_ES