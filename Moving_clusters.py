import numpy as np
from sklearn.cluster import DBSCAN
from alphashape import alphashape
import scipy.signal as signal 
import scipy.stats as stat


def preprocessing(trial, border_mask, fs, fc, threshold):
    """
    Pre-processing of a single trial, including border removing, temporal filtering and the application of a threshold
    """
    Trial=np.copy(trial)
    Trial[:, ~border_mask] = 0
    butt_filt = signal.butter(N=4, Wn=fc, btype='lowpass', fs=fs, output='sos')
    Trial=signal.sosfiltfilt(butt_filt, Trial, axis=0)
    thresh_value=np.percentile(Trial, threshold)
    Trial = Trial - thresh_value
    Trial[Trial<0]=0
    return Trial



def valid_moving_cluster(cluster1,cluster2,Theta,alpha):
    """
    Test if thpercentage of shared surface between two sets of points exceed the threshold
    """
    polygon1,polygon2 = alphashape(cluster1,alpha),alphashape(cluster2,alpha)
    inter_area = polygon1.intersection(polygon2).area
    union_area = polygon1.union(polygon2).area
    return inter_area/union_area>Theta




def MC1(Trial, epsilon, min_pts, Theta=.5, alpha=0.1):
    ''' Moving Cluster algo, first version of the paper (straight-forward definition)
    ----------
    Parameters:
        Trial: preprocessed video data (filetred and cut above a threshold), numpy array of format (Nt,Nx,Ny)
        epsilon:
        min_pts:
        Theta: Threshold of shared surface proportion to consider two clusters as consecutives
        alpha: parameter to compute the alpha shape and therefore the common surface between two clusters
    ----------
    Return :
        list of dicts containing a list of coordinates matrices
    '''
    N_timeslices,_,_= Trial.shape
    G=[] #list of moving clusters detected in data
    for t in range(N_timeslices):
        print(f'frame #{t}')
        pts_x, pts_y = np.where(Trial[t]>0) #indices of the points to cluster
        current_points = np.vstack((pts_x, pts_y)).T #matrix of coordinates of all points
        for g in G:
            g['extension'] = False #initiation of boolean variable indicating that the moving cluster has been extended
        G_next=[] #list of next moving clusters
        dbscan = DBSCAN(eps=epsilon, min_samples=min_pts)
        cluster_labels = dbscan.fit_predict(current_points)
        cluster_points = [current_points[cluster_labels == label] for label in np.unique(cluster_labels) if label != -1] #list of all clusters in current timeslice data
        for clust_pts in cluster_points: #clusters' assignement loop
            assigned=False
            for g in G: #previous moving clusters
                if valid_moving_cluster(g['points'][-1], clust_pts, Theta, alpha): #test validity for all moving clusters
                    print('valid mc detected')
                    g['points'].append(clust_pts)
                    g['trace'].append(t)
                    g['extension']=True
                    G_next.append(g)
                    assigned=True
            if not assigned: #if this cluster has no antecedents, then it may be the first of a new moving cluster
                G_next.append({'points':[clust_pts], 'trace':[t]})
        for g in G:
            if not g['extension']: #a moving cluster that hasn't been extended is either already completed or invalid depending on its length
                if len(g['points'])>1: 
                    G_next.append(g)
        G=G_next[:]
    return G