import numpy as np
import scipy.signal as signal
from sklearn.cluster import DBSCAN
from alphashape import alphashape


def circular_mask(shape, center, radius):
    Y, X = np.ogrid[:shape[0], :shape[1]]
    return ((X - center[0]) ** 2 + (Y - center[1]) ** 2) <= radius ** 2


def preprocessing(trial, border_mask, fs, fc, threshold):
    """
    Pre-processing of a single trial, including border removing, temporal filtering and the application of a threshold
    ------
    Parameters:
        border_mask = boolean mask to remove the border
        fs = frequency sampling
        fc = cutoff frequency for the lowpass filter
        threshold = {float between 0 and 100} percentile to keep
    ------
    """
    Trial=np.copy(trial)
    Trial[:, ~border_mask] = 0
    butt_filt = signal.butter(N=4, Wn=fc, btype='lowpass', fs=fs, output='sos')
    Trial=signal.sosfiltfilt(butt_filt, Trial, axis=0)
    thresh_value=np.percentile(Trial, threshold)
    Trial = Trial - thresh_value
    Trial[Trial<0]=0
    return Trial


def overlap_test(shape1,shape2,Theta):
    """
    Test if the percentage of overlap between two polygons exceed the threshold
    Returns a boolean
    """
    inter_area = shape1.intersection(shape2).area
    union_area = shape1.union(shape2).area
    return inter_area/union_area>Theta


def MCDA(Trial, Epsilon, MinPts, Theta, alpha=0.1):
    ''' Moving Cluster Detection Algo, first version of the original paper (straight-forward definition)
    ----------
    Parameters:
        Trial: {numpy array of format (Nt,Nx,Ny)} Preprocessed video data (filetred and cut above a threshold)
        Epsilon: {float} Distance above which two points are considered as neighbours in DBSCAN
        MinPts: {int} number of points above which an epsilon-neighbourhood is considered as dense in DBSCAN
        Theta: {float between 0 and 1} Threshold to consider two clusters as consecutives
        alpha: {float between 0 and 1} Parameter to compute the alpha shape and therefore the relative shared area between two clusters
    ----------
    Return :
        list of dicts containing a list of coordinates matrices
        = list of moving clusters
    '''
    N_timeslices,_,_= Trial.shape
    G=[] #list of previously detected moving clusters
    Output=[]
    for t in range(N_timeslices):
        #print(f'frame #{t}')
        pts_x, pts_y = np.where(Trial[t]>0) #indices of the points to cluster
        current_points = np.vstack((pts_x, pts_y)).T #matrix of coordinates of all points
        for g in G:
            g['extension'] = False #initiation of boolean variable indicating that the moving cluster has been extended
        G_next=[] #list of new moving clusters after considering the current timeslice
        dbscan = DBSCAN(eps=Epsilon, min_samples=MinPts)
        cluster_labels = dbscan.fit_predict(current_points)
        cluster_points = [current_points[cluster_labels == label] for label in np.unique(cluster_labels) if label != -1] #list of all clusters in current timeslice data
        for clust_pts in cluster_points: #clusters' assignement loop
            assigned=False
            for g in G: #previous moving clusters
                g_shape=alphashape(clust_pts,alpha)
                if overlap_test(g['alpha shapes'][-1], g_shape, Theta): #test validity for all uncompleted moving clusters
                    g['points'].append(clust_pts)
                    g['alpha shapes'].append(g_shape)
                    g['trace'].append(t)
                    g['extension']=True
                    G_next.append(g)
                    assigned=True
            if not assigned: #if this cluster has no antecedents, then it may be the first of a new moving cluster
                moving_cluster = { 'points':[clust_pts], 'trace':[t],'alpha shapes':[alphashape(clust_pts,alpha)], 'extension':True }
                G_next.append(moving_cluster)
        for g in G:
            if not g['extension']: #a moving cluster that hasn't been extended is either already completed or invalid depending on its length
                if len(g['trace'])>3: 
                    Output.append(g)
        G=G_next[:]
    for g in G: #last test to pick the moving clusters still extending in the last timeslice
        if len(g['trace'])>3: 
            Output.append(g)
    return Output

