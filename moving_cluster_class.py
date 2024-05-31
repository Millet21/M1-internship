import numpy as np
import scipy.stats as stat
from astropy.stats import kuiper
from functools import partial
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import split
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
        self.timepoints = [t/self.fs for t in self.trace]
        self.duration = self.length/self.fs*1000
        self.centroids = [np.mean(points, axis=0) for points in self.list_points]
        self.source = self.centroids[0]
        self.surfaces = [shape.area*self.pixel_spacing**2 for shape in self.shapes]

    def expansion_test(self, sign_level):
        radius=[]
        for clust,center in zip(self.list_points,self.centroids):
            dist=np.linalg.norm(clust-center,axis=1) 
            radius.append(np.mean(dist))
        res_radius = stat.spearmanr(self.trace,radius,alternative='greater')
        radial_speed = np.append(np.diff(radius)*self.fs*self.pixel_spacing/1000, np.nan)
        res_surf=stat.spearmanr(self.trace, self.surfaces, alternative='greater')
        inter_test = res_radius.pvalue<sign_level and res_surf.pvalue<sign_level
        return (inter_test, radial_speed if inter_test else np.full(self.length,np.nan))

    def translation_test(self, MinDist, significance_level):
        none_list = np.full(self.length,np.nan)
        velocities = np.diff(self.centroids, axis=0)
        dot_products = np.einsum('ij,ij->i', velocities[:-1], velocities[1:])
        norms = np.linalg.norm(velocities, axis=1)
        norms_prod = norms[:-1]*norms[1:]
        directional_persistence = np.mean(dot_products / norms_prod)
        test_persistence = directional_persistence>0.5
        centroids_from_source = np.linalg.norm(self.centroids - self.source, axis=1)
        test_mindist = np.max(centroids_from_source)>MinDist
        inter_test = test_persistence and test_mindist
        return (inter_test, np.append(norms*self.fs*self.pixel_spacing/1000, np.nan) if inter_test else none_list)

    def classify(self, MinDist, significance_level):
        self.expansion, self.radial_speed = self.expansion_test(significance_level)
        self.translation, self.centroid_speed = self.translation_test(MinDist, significance_level)

    def propagation_direction(self, significance_level):
        pts_angle_from_source=[]
        for clust_pts in self.list_points:
            clust_from_source=clust_pts-self.source
            angle=np.arctan2(clust_from_source[:,0],clust_from_source[:,1])
            pts_angle_from_source.append(angle)
        Angles=np.concatenate(pts_angle_from_source)
        kuiper_test=kuiper(data=Angles, cdf=partial(stat.uniform.cdf, loc=-np.pi, scale=2*np.pi))
        self.isotropy =  kuiper_test[1]>significance_level
        self.direction = stat.circmean(Angles) if not self.isotropy else np.nan
    
    def border_crossing(self, BorderLine):
        cross_list, v2_area = [], []
        for shape in self.shapes:
            cross_list.append(shape.crosses(BorderLine))
            area_below_line = 0.
            if shape.crosses(BorderLine):
                split_polygons = split(shape, BorderLine)
                # Determine which part is below the line and calculate its area
                for poly in split_polygons:
                    # Check if the centroid of the polygon is below the line
                    if poly.centroid.y < min(BorderLine.bounds):
                        area_below_line += poly.area
            v2_area.append(area_below_line)
        self.border_cross = cross_list
        self.v2_area = v2_area

    def plot(self):
        pass 



    

