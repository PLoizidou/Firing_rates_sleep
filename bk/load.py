import numpy as np
import pandas as pd
import neuroseries as nts
import scipy.io
import sys
import time
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
import xml.etree.ElementTree as ET
import os
import bk.compute
import re

global session, path, rat, day, n_channels


def sessions():
    return pd.read_csv("Z:/All-Rats/Billel/session_indexing.csv", sep=";")




def current_session(path_local = 'Z:\Rat08\Rat08-20130713', return_vars=False):
    #Author : BK 08/20
    #Input Path to the session to load
    #output : True if loading was done correctly
    #Variable are stored in global variables.
    
        #Create Global variable that allow for all function to know in wich session we are this usefull only for variable that are going to be recurentyly used. Do not overuse this functionnality as it can add inconstansies. 
    global session, path, rat, day,n_channels

    
    session_index = pd.read_csv("Z:/All-Rats/Billel/session_indexing.csv", sep=";")
   
    path = path_local
    os.chdir(path)


    session = path.split('\\')[2]
    rat = session_index['Rat'][session_index['Path'] == path].values[0]
    day = session_index['Day'][session_index['Path'] == path].values[0]

    n_channels = xml(session)['nChannels']
    
    
    print('Rat : ' + str(int(rat)) + ' on day : ' + str(int(day)))
    print('Working with session ' + session + ' @ ' + path)
    
    if return_vars==True:
        return session, path, rat, day,n_channels
    return;



def current_session_linux(
    base_folder="/home/billel/Data/GG-Dataset/", local_path="Rat08/Rat08-20130713"
):
    # Author : BK 08/20
    # Input Path to the session to load
    # output : True if loading was done correctly
    # Variable are stored in global variables.

    # Create Global variable that allow for all function to know in wich session we are this usefull only for variable that are going to be recurentyly used. Do not overuse this functionnality as it can add inconstansies.
    global base, session, path, rat, day, n_channels
    base = base_folder

    os.chdir(base)
    session_index = pd.read_csv("relative_session_indexing.csv")

    session = local_path.split("/")[1]
    rat = session_index["Rat"][session_index["Path"] == local_path].values[0]
    day = session_index["Day"][session_index["Path"] == local_path].values[0]
    path = os.path.join(base, local_path)
    os.chdir(path)

    n_channels = xml()["nChannels"]

    print("Rat : " + str(int(rat)) + " on day : " + str(int(day)))
    print("Working with session " + session + " @ " + path)

    return True


def xml(session):
    os.chdir(path)
    tree = ET.parse(session + ".xml")
    root = tree.getroot()

    xmlInfo = {}
    for elem in root:
        for subelem in elem:
            try:
                xmlInfo.update({subelem.tag: int(subelem.text)})
            except:
                pass
    return xmlInfo


def batch(func, *args, verbose=False, linux=False):

    # Author : BK
    # Date : 08/20

    # Input Function
    # Output : Output of the function

    # This function batch over all rat / all session and return output of the functions.
    print(args)
    t = time.time()

    if linux:
        current_session_linux()
        os.chdir(base)
        session_index = pd.read_csv("relative_session_indexing.csv")
    else:
        session_index = pd.read_csv("Z:/All-Rats/Billel/session_indexing.csv", sep=";")

    error = []
    output_dict = {}
    for path in tqdm(session_index["Path"]):

        if linux:
            session = path.split("/")[1]
        else:
            session = path.split("\\")[2]
        print("Loading Data from " + session)

        try:
            output = func(os.path.join(path), *args)
            output_dict.update({session: output})
            if not verbose:
                clear_output()
        except:
            error.append(session)
            print("Error in session " + session)
            if not verbose:
                clear_output()
    print("Batch finished in " + str(time.time() - t))

    if error:
        print("Some session were not processed correctly")
        print(error)
        print(len(error) / len(session_index["Path"]) * 100, "%")

    return output_dict


def get_raw_data_directory(raw_data_directory="\\\AGNODICE\IcyBox"):
    return raw_data_directory


def get_session_path(session_name):
    # Author : Anass
    rat = session_name[0:5]  # "Rat08"
    rat_path = os.path.join(get_raw_data_directory(), rat)
    session_path = os.path.join(rat_path, session_name)
    return session_path


def shank_to_structure(rat, day, shank):
    structures_path = os.path.join(base, "All-Rats/Structures/structures.mat")
    structures = scipy.io.loadmat(structures_path)
    useless = ["__header__", "__version__", "__globals__", "basal", "olfact"]
    for u in useless:
        del structures[u]

    for stru, array in structures.items():
        filtered_array = array[np.all(array == [rat, day, shank], 1)]
        if np.any(filtered_array):
            return stru


def channels():

    tree = ET.parse(session + ".xml")
    root = tree.getroot()
    shank_channels = {}

    i = 0
    for anatomicalDescription in root.iter("anatomicalDescription"):
        for channelGroups in anatomicalDescription.iter("channelGroups"):
            for group in channelGroups.iter("group"):
                i += 1
                channels = []
                for channel in group.iter("channel"):
                    channels.append(int(channel.text))
                stru = shank_to_structure(rat, day, i)
                if (i == 21) or (i == 22):
                    stru = "Accel"
                shank_channels.update({i: [channels, stru]})
    return shank_channels


def best_channel(shank):
    rat_shank_channels = os.path.join(base, "All-Rats/Rat_Shank_Channels.csv")
    rat_shank_channels = pd.read_csv(rat_shank_channels)

    chan = rat_shank_channels[
        (rat_shank_channels["rat"] == rat) & (rat_shank_channels["shank"] == shank)
    ]
    if not np.isnan(shank):
        return chan["channel"].values[0]
    else:
        return np.nan


def bla_shanks():
    bla_shanks = os.path.join(base, "All-Rats/BLA_Shanks.csv")
    bla_shanks = pd.read_csv(bla_shanks)

    left_chan = bla_shanks[(bla_shanks["Rat"] == rat) & (bla_shanks["Day"] == day)][
        "Left"
    ].values[0]
    right_chan = bla_shanks[(bla_shanks["Rat"] == rat) & (bla_shanks["Day"] == day)][
        "Right"
    ].values[0]
    try:
        left_chan = int(left_chan)
    except:
        pass

    try:
        right_chan = int(right_chan)
    except:
        pass

    return {"left": left_chan, "right": right_chan}


def bla_channels():
    return {
        "left": best_channel(bla_shanks()["left"]),
        "right": best_channel(bla_shanks()["right"]),
    }


def random_channel(stru):
    chans = channels()

    chan = []
    for shank, (channel, s) in chans.items():
        if s == stru:
            chan.append(channel)

    chan = np.random.choice(np.array(chan).ravel())
    return chan


def pos(save=False):
    # BK : 04/08/2020
    # Return a NeuroSeries DataFrame of position whith the time as index

    #     session_path = get_session_path(session_name)
    import csv

    pos_clean = scipy.io.loadmat(path + "/posClean.mat")["posClean"]
    #     if save == True :
    #         with open('position'+'.csv', 'w') as csvfile:
    #             filewriter=csv.writer(csvfile)
    return nts.TsdFrame(
        t=pos_clean[:, 0], d=pos_clean[:, 1:], columns=["x", "y"], time_units="s"
    )


def state_vector():
    names = {"wake": 1, "drowsy": 2, "nrem": 3, "intermediate": 4, "rem": 5}

    states = scipy.io.loadmat(bk.load.session + "-states.mat")["states"][0]
    states = np.array(states, np.object)

    for name, number in names.items():
        states[np.where(states == number)] = name
    return states


def states():
    # BK : 17/09/2020
    # Return a dict with variable from States.
    #     if session_path == 0 : session_path = get_session_path(session_name)
    states = scipy.io.loadmat(path + "/States.mat")

    useless = ["__header__", "__version__", "__globals__"]
    for u in useless:
        del states[u]
    states_ = {}
    for state in states:
        states_.update(
            {
                state: nts.IntervalSet(
                    states[state][:, 0], states[state][:, 1], time_units="s"
                )
            }
        )

    return states_


def ripples():
    ripples_ = scipy.io.loadmat(f"{bk.load.session}-RippleFiring.mat")["ripples"][
        "allsws"
    ][0][0]
    #     ripples_ = pd.DataFrame(data = ripples,columns=['start','peak','stop'])

    columns = ["start", "peak", "stop"]

    ripples = {}
    for i, c in zip(range(ripples_.shape[1]), columns):
        ripples.update({c: nts.Ts(ripples_[:, i], time_units="s")})
    return ripples


def ripple_channel():
    with open(f"{session}.rip.evt", "r") as f:
        rip = f.readline()
    chan = re.findall("\d+", rip)[-1]

    try:
        chan = int(chan)
    except:
        pass

    return chan

    return chan


def events(filename):
    if not filename.endswith(".evt"):
        filename += ".evt"

    print("Not done yet")
    return 0


def run_intervals():
    trackruntimes = scipy.io.loadmat(session + "-TrackRunTimes.mat")["trackruntimes"]
    trackruntimes = nts.IntervalSet(
        trackruntimes[:, 0], trackruntimes[:, 1], time_units="s"
    )

    return trackruntimes


def sleep():
    runs = scipy.io.loadmat("runintervals.mat")["runintervals"]
    pre_sleep = nts.IntervalSet(start=runs[0, 1], end=runs[1, 0], time_units="s")
    post_sleep = nts.IntervalSet(start=runs[1, 1], end=runs[2, 0], time_units="s")

    return pre_sleep, post_sleep


def laps():
    laps = {}
    danger = scipy.io.loadmat(f"{session}-LapType.mat")["aplaps"][0][0][0]
    safe = scipy.io.loadmat(f"{session}-LapType.mat")["safelaps"][0][0][0]

    danger = nts.IntervalSet(danger[:, 0], danger[:, 1], time_units="s")
    safe = nts.IntervalSet(safe[:, 0], safe[:, 1], time_units="s")

    laps.update({"danger": danger, "safe": safe})
    return laps


def spikes():
    return loadSpikeData(path)


def loadSpikeData(path, index=None, fs=20000):
    ### Adapted from Viejo github https://github.com/PeyracheLab/StarterPack/blob/master/python/wrappers.py
    ### Modified by BK 06/08/20
    ### Modification are explicit with comment
    """
    if the path contains a folder named /Analysis, 
    the script will look into it to load either
        - SpikeData.mat saved from matlab
        - SpikeData.h5 saved from this same script
    if not, the res and clu file will be loaded 
    and an /Analysis folder will be created to save the data
    Thus, the next loading of spike times will be faster
    Notes :
        If the frequency is not givne, it's assumed 20kH
    Args:
        path : string

    Returns:
        dict, array    
    """

    #     try session:
    #     except: print('Did you load a session first?')

    if not os.path.exists(path):
        print("The path " + path + " doesn't exist; Exiting ...")
        sys.exit()
    if os.path.exists(path + "//" + session + "-neurons.npy"):
        print("Data already saved in Numpy format, loading them from here:")
        print(session + "-neurons.npy")
        neurons = np.load(path + "//" + session + "-neurons.npy", allow_pickle=True)
        print(session + "-metadata.npy")
        shanks = np.load(path + "//" + session + "-metadata.npy", allow_pickle=True)
        shanks = pd.DataFrame(
            shanks, columns=["Rat", "Day", "Shank", "Id", "Region", "Type"]
        )
        return neurons, shanks

    files = os.listdir(path)
    # Changed 'clu' to '.clu.' same for res as in our dataset we have file containing the word clu that are not clu files
    clu_files = np.sort([f for f in files if ".clu." in f and f[0] != "."])
    res_files = np.sort([f for f in files if ".res." in f and f[0] != "."])

    # Changed because some files have weird names in GG dataset because of some backup on clu/res files
    # Rat10-20140627.clu.10.07.07.2014.15.41 for instance

    clu_files = clu_files[[len(i) < 22 for i in clu_files]]
    res_files = res_files[[len(i) < 22 for i in res_files]]

    clu1 = np.sort([int(f.split(".")[-1]) for f in clu_files])
    clu2 = np.sort([int(f.split(".")[-1]) for f in res_files])

    #     if len(clu_files) != len(res_files) or not (clu1 == clu2).any():
    #         print("Not the same number of clu and res files in "+path+"; Exiting ...")
    #         sys.exit()
    #   Commented this because in GG dataset their .clu.12.54.21.63 files that mess up everything ...

    count = 0
    spikes = []
    basename = clu_files[0].split(".")[0]
    idx_clu_returned = []
    for i, s in zip(range(len(clu_files)), clu1):
        clu = np.genfromtxt(
            os.path.join(path, basename + ".clu." + str(s)), dtype=np.int32
        )[1:]
        print("Loading " + basename + ".clu." + str(s))
        if np.max(clu) > 1:
            res = np.genfromtxt(os.path.join(path, basename + ".res." + str(s)))
            tmp = np.unique(clu).astype(int)
            idx_clu = tmp[tmp > 1]
            idx_clu_returned.extend(
                idx_clu
            )  # Allow to return the idx of each neurons on it's shank. Very important for traceability
            idx_col = np.arange(count, count + len(idx_clu))
            tmp = pd.DataFrame(
                index=np.unique(res) / fs,
                columns=pd.MultiIndex.from_product([[s], idx_col]),
                data=0,
                dtype=np.uint16,
            )

            for j, k in zip(idx_clu, idx_col):
                tmp.loc[res[clu == j] / fs, (s, k)] = np.uint16(k + 1)
            spikes.append(tmp)
            count += len(idx_clu)

    # Returning a list instead of dict in order to use list of bolean.
    toreturn = []
    shank = []
    for s in spikes:
        shank.append(s.columns.get_level_values(0).values)
        sh = np.unique(shank[-1])[0]
        for i, j in s:
            toreturn.append(
                nts.Tsd(
                    t=s[(i, j)].replace(0, np.nan).dropna().index.values, time_units="s"
                )
            )
            # To return was change to nts.Tsd instead of nts.Ts as it has bug for priting (don't know where it is coming from)

    del spikes
    shank = np.hstack(shank)

    neurons = np.array(toreturn, dtype="object")
    shanks = np.array([shank, idx_clu_returned]).T

    print()
    print("Saving data in Numpy format :")

    print("Saving " + session + "-neurons.npy")
    np.save(path + "//" + session + "-neurons", neurons)

    print("Saving " + session + "-neuronsShanks.npy")
    np.save(path + "//" + session + "-neuronsShanks", shanks)

    return (
        neurons,
        shanks,
    )  # idx_clu is returned in order to keep indexing consistent with Matlab code.


def loadLFP(path, n_channels=90, channel=64, frequency=1250.0, precision="int16"):

    """
    LEGACY
    """
    # From Guillaume Viejo
    import neuroseries as nts

    if type(channel) is not list:
        f = open(path, "rb")
        startoffile = f.seek(0, 0)
        endoffile = f.seek(0, 2)
        bytes_size = 2
        n_samples = int((endoffile - startoffile) / n_channels / bytes_size)
        duration = n_samples / frequency
        interval = 1 / frequency
        f.close()
        with open(path, "rb") as f:
            print("opening")
            data = np.fromfile(f, np.int16).reshape((n_samples, n_channels))[:, channel]
            timestep = np.arange(0, len(data)) / frequency
        return nts.Tsd(timestep, data, time_units="s")
    elif type(channel) is list:
        f = open(path, "rb")
        startoffile = f.seek(0, 0)
        endoffile = f.seek(0, 2)
        bytes_size = 2

        n_samples = int((endoffile - startoffile) / n_channels / bytes_size)
        duration = n_samples / frequency
        f.close()
        with open(path, "rb") as f:
            data = np.fromfile(f, np.int16).reshape((n_samples, n_channels))[:, channel]
            timestep = np.arange(0, len(data)) / frequency
        return nts.TsdFrame(timestep, data, time_units="s")


def recording_length(fs=1250):

    data = np.memmap(session + ".lfp", np.uint16)
    data = data.reshape(-1, n_channels)

    rec_len = len(data) / fs
    del data

    return rec_len


def lfp(
    channel,
    start=0,
    stop=1e8,
    fs=1250.0,
    n_channels_local=None,
    precision=np.int16,
    dat=False,
    verbose=False,
    memmap=False,
    p=None,
    volt_step=0.195,
):

    if (np.isnan(channel)) or (channel is None):
        return None

    if p is None:
        p = session + ".lfp"
        if dat:
            p = session + ".dat"

    if n_channels_local is None:
        n_channels = xml()["nChannels"]
    else:
        n_channels = n_channels_local

    if verbose:
        print("Load data from " + p)
        print(f"File contains {n_channels} channels")

    # From Guillaume viejo
    import neuroseries as nts

    bytes_size = 2
    start_index = int(start * fs * n_channels * bytes_size)
    stop_index = int(stop * fs * n_channels * bytes_size)
    # In order not to read after the file
    if stop_index > os.path.getsize(p):
        stop_index = os.path.getsize(p)
    fp = np.memmap(
        p, precision, "r", start_index, shape=(stop_index - start_index) // bytes_size
    )
    if memmap == True:
        print("/!\ memmap is not compatible with volt_step /!\ ")
        return fp.reshape(-1, n_channels)[:, channel]
    data = np.array(fp).reshape(len(fp) // n_channels, n_channels)

    if type(channel) is not list:
        timestep = np.arange(0, len(data)) / fs + start
        return nts.Tsd(timestep, data[:, channel], time_units="s")
    elif type(channel) is list:
        timestep = np.arange(0, len(data)) / fs + start
        return nts.TsdFrame(timestep, data[:, channel], time_units="s")


def lfp_in_intervals(channel, intervals):
    t = np.array([])
    lfps = np.array([])

    for start, stop in zip(intervals.as_units("s").start, intervals.as_units("s").end):
        start = np.round(start, decimals=1)
        stop = np.round(stop, decimals=1)
        lfp = bk.load.lfp(channel, start, stop)
        t = np.append(t, lfp.index)
        lfps = np.append(lfps, lfp.values)

    lfps = nts.Tsd(t, lfps)

    return lfps


#####


def digitalin_old(local_path=None, nchannels=16, Fs=20000):
    import pandas as pd
    if  local_path is None:
        local_path = session+'-digitalin.dat' 

    digital_word = np.fromfile(local_path, "uint16")
    sample = len(digital_word)
    time = np.arange(0, sample)
    time = time / Fs
    
    for i in range(nchannels):
        if i == 0:
            data = (digital_word & 2 ** i) > 0
        else:
            data = np.vstack((data, (digital_word & 2 ** i) > 0))


    return data


def digitalin(chan, Fs=20000):
    import pandas as pd
    
    local_path = session+'-digitalin.dat' 

    digital_word = np.fromfile(local_path, "uint16")
    data = (digital_word & 2 ** chan) > 0

    return data


def analogin(
    channel, start=0, stop=1e8, fs=20000, dat=False, verbose=False, memmap=False, p=None
):
    # FIXME
    ## Get analogin path
    if p is None:
        if not dat:
            p = f"{session}-analogin.lfp"
        else:
            p = f"{session}-analogin.dat"
    print(p)
    ## Get analogin n_channels :
    rec_len = recording_length()

    analogin = lfp(channel, start, stop, fs, 1, np.uint16, dat, verbose, memmap, p)
    return analogin


def freezing_intervals():
    if os.path.exists("freezing_intervals.npy"):
        freezing_intervals = np.load("freezing_intervals.npy")
        return nts.IntervalSet(
            start=freezing_intervals[:, 0], end=freezing_intervals[:, 1]
        )
    else:
        print("Could not find freezing_intervals.npy")
        return False


def DLC_pos(filtered=True, force_reload=False, save=False):
    """
    Load position from DLC files (*.h5) and returns it as a nts.TsdFrame
    """
    files = os.listdir()
    if ("positions.h5" in files) and (force_reload == False):
        print("hey listen")
        data = pd.read_hdf("positions.h5")
        pos = nts.TsdFrame(data)
        return pos

    for f in files:
        if filtered and f.endswith("filtered.h5"):
            filename = f
            break
        if not filtered and not f.endswith("filtered.h5") and f.endswith(".h5"):
            print(f)
            filename = f
            break
    data = pd.read_hdf(filename)
    data = data[data.keys()[0][0]]

    TTL = digitalin("digitalin.dat")[0, :]
    tf = bk.compute.TTL_to_times(TTL)

    if len(tf) > len(data):
        tf = np.delete(tf, -1)

    data.index = tf * 1_000_000

    if save:
        data.to_hdf("positions.h5", "pos")

    pos = nts.TsdFrame(data)
    return pos

