import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import neuroseries as nts

def rasterPlot(neurons,window,col = 'black',width= 0.5,height = 1,offsets = 1):
    window = nts.IntervalSet(window[:,0],window[:,1],time_units = 's')
    neurons_np = []
    
    if isinstance(neurons,nts.time_series.Tsd):
        neurons = [neurons]
    print(type(neurons))
    for neuron in neurons:
        neurons_np.append(neuron.restrict(window).as_units('s').index)

    neurons_np = np.array(neurons_np,dtype = 'object')
    
    
    plt.eventplot(neurons_np,color = col,linewidth = width,linelengths=height,lineoffsets=offsets)
    plt.ylabel('Neurons')
    plt.xlabel('Time(s)')
    
def intervals(intervals,col = 'orange',alpha = 0.5,time_units = 's'):
    
    if type(intervals) != nts.interval_set.IntervalSet:
        print(type(intervals))
        intervals = nts.IntervalSet(intervals['start'],intervals['end'])
    
    for interval in intervals.as_units(time_units).values:
        plt.axvspan(interval[0],interval[1], facecolor=col, alpha=alpha)