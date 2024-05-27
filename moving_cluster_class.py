import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
#______________________________________________________________________________
class Moving_Cluster:
    
    def __init__(self, list_points, trace, alpha_shapes, Fs, pixel_spacing, condition, session, trial, number):
        self.list_points = list_points
        self.trace = trace
        self.shapes = alpha_shapes
        self.fs = Fs
        self.pixel_spacing = pixel_spacing
        self.condition = condition
        self.session = session
        self.trial = trial
        self.number = number
        self.length = len(self.trace)
        self.timepoints = [t/self.fs for t in range(self.length)]
        self.centroids = np.mean(list_points, axis=0)
        self.source = self.centroids[0]
        self.surfaces = [shape.area*self.pixel_spacing**2 for shape in self.shapes]

    def expansion_test(self, sign_level):
        radial_speed = np.full(self.length,np.nan)
        pts_dist_to_centroid=[]
        for clust,center in zip(self.list_points,self.centroids):
            dist=np.linalg.norm(clust-center,axis=1) 
            pts_dist_to_centroid.append(dist)
        Rho=np.concatenate(pts_dist_to_centroid)
        T=[time for index,time in enumerate(self.trace) for _ in range(len(self.list_points[index]))]
        res_rho = stat.linregress(T,Rho,alternative='greater')
        a = res_rho.pvalue<sign_level
        res_surf=stat.spearmanr(self.trace, self.surfaces, alternative='greater')
        b = res_surf.pvalue<sign_level
        return (a and b, radial_speed)

    def translation_test(self,MinDist, sign_level):
        pts_stack = np.concatenate(self.list_points)
        T=[time for index,time in enumerate(self.trace) for _ in range(len(self.list_points[index]))]
        X,Y = pts_stack[:, 0], pts_stack[:, 1]
        linreg_X = stat.linregress(T,X)
        linreg_Y = stat.linregress(T,Y)
        centroid_speed = np.nan
        c = linreg_X.pvalue<sign_level or linreg_Y.pvalue<sign_level
        Dist = np.linalg.norm(np.diff(np.centroids,axis=0), axis=1)
        Instant_speed = np.append(Dist*self.fs*self.pixel_spacing/1000, np.nan)
        velocities = np.diff(pts_stack, axis=0)
        dot_products = np.einsum('ij,ij->i', velocities[:-1], velocities[1:])
        norms = np.linalg.norm(velocities[:-1], axis=1) * np.linalg.norm(velocities[1:], axis=1)
        # Compute directional persistence
        directional_persistence = np.mean(dot_products / norms)
        if c and d:
            centroid_speed = np.linalg.norm((linreg_X.slope, linreg_Y.slope))*self.pixel_spacing/1000*self.fs
        return (c and d, Instant_speed)

    def classify(self, MinDist, significance_level):
        self.expansion, self.radial_speed = self.expansion_test(MinDist, significance_level)
        self.translation, self.centroid_speed = self.translation_test(significance_level)

    def plot():
        pass 



    

