import numpy as np
import scipy.stats as stat
from astropy.stats import kuiper
from functools import partial
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import split
#from Classification_functions import dp_pdf
from scipy.signal import fftconvolve

#______________________________________________________________________________
class Moving_Cluster:
    
    def __init__(self, list_points, trace, alpha_shapes, Fs, pixel_spacing, condition, session, trial, number):
        self.list_points = list_points
        self.trace = trace
        self.shapes = alpha_shapes
        self.fs = Fs #in Hz
        self.pixel_spacing = pixel_spacing #in mm/px
        self.condition = condition
        self.session = session
        self.trial = trial
        self.number = number
        self.length = len(self.trace)
        self.timepoints = np.array([t/self.fs*1000 for t in self.trace]) #in ms
        self.duration = self.length/self.fs*1000 #in ms
        self.centroids = np.array([np.mean(points, axis=0) for points in self.list_points])
        self.source = self.centroids[0]
        self.surfaces = np.array([shape.area*self.pixel_spacing**2 for shape in self.shapes]) #in mm2

    def expansion_test(self, MinDist, significance_level=.01):
        mean_radius=[]
        for index,clust_shape in enumerate(self.shapes):
            contour_pts = np.array(clust_shape.exterior.xy)
            dist=np.linalg.norm(contour_pts.T-self.centroids[index],axis=1) 
            mean_radius.append(np.mean(dist))
        rho=np.array(mean_radius)
        max_ind=np.argmax(self.surfaces)
        expansion_speed = np.diff(rho[:max_ind+1])*self.fs*self.pixel_spacing/1000
        test_radius = rho[max_ind]-rho[0]>MinDist/self.pixel_spacing
        surf_regress = stat.spearmanr(self.trace[:max_ind+1], self.surfaces[:max_ind+1], alternative='greater').pvalue
        test_surf  = surf_regress<significance_level
        inter_test = test_radius and test_surf
        return  (inter_test, np.append(expansion_speed,np.full(self.length-len(expansion_speed),np.nan)) if inter_test else np.nan, test_surf, test_radius)

    def translation_test(self, MinDist, significance_level):
        source=self.centroids[0]
        #Directional persistence test: is the centroid moving directionnaly enough?
        velocities=np.diff(self.centroids, axis=0)
        dot_products = np.einsum('ij,ij->i', velocities[:-1], velocities[1:])#pairwise dot_products between consecutive velocity vectors
        norms = np.linalg.norm(velocities, axis=1)
        prod_norms=norms[:-1]*norms[1:] #normalisation factors
        directional_persistence = np.mean(dot_products/prod_norms)#equivalent to the mean cosine of the angles between consecutive velocity vectors
        xrange, density_function = dp_pdf(self.length) #computation of the analytic pdf of directional persistence under assumption of randomness of the movement
        dp_index=np.argmin((xrange-directional_persistence)**2)
        pval=np.trapz(density_function[dp_index:], xrange[dp_index:])#calculation of the p-value
        test_persistence = pval < significance_level
        #Distance test: iis the centroid moving more than a defined minimal distance?
        centroids_from_source = np.linalg.norm(self.centroids - source, axis=1)
        test_mindist = np.max(centroids_from_source)>MinDist/self.pixel_spacing
        #Intersection of these tests
        inter_test = test_persistence and test_mindist

        return (inter_test, np.append(norms*self.fs*self.pixel_spacing/1000, np.nan) if inter_test else np.nan, test_persistence, test_mindist)

    def classify(self, MinDist, significance_level):
        """
        MinDist in mm
        """ 
        self.expansion, self.radial_speed, self.test_surf, self.test_radius = self.expansion_test(MinDist,significance_level) #speeds in m/s
        self.translation, self.centroid_speed,self.test_persistence, self.test_mindist = self.translation_test(MinDist, significance_level) #speeds in m/s
        if self.expansion and self.translation:
            self.pattern_type='Complex'
        elif self.expansion and not self.translation:
            self.pattern_type='Radial'
        elif not self.expansion and self.translation:
            self.pattern_type='Plane'
        else:
            self.pattern_type='Static'

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
        self.border_cross =  [shape.crosses(BorderLine) for shape in self.shapes]
    
    def area_crossed(self, BorderLine): #a avancer plus tard
        """
        Evaluate in which visual area each cluster is localised based on the location of its centroid and the V1/V2 border
        The proportion of its shape in each area is also evaluated
        """
        v1_area, v2_area = [], []
        cross_list=[]
        for center, shape in zip(self.centroids,self.shapes):
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
        else:
            pass
        v1_area=np.array(v1_area)
        v2_area=np.array(v2_area)
        self.v1_proportion = v1_area/self.surfaces
        self.v2_area = v2_area/self.surfaces
        self.visual_area = "V1" or "V2"

    def plot(self):
        pass 


#____________________________________________________________________________________________________
#Useful functions for classsification

def pdf_cos_theta(y):
    # Probability density function of X = cos(Theta) if Theta follows a uniform law
    if abs(y)<1:
        return 1/(np.pi*np.sqrt(1 - y**2))
    else:
        return 0


def dp_pdf(N):
    # Probability density function of Y= mean( cos(Theta) ) if Theta follows a uniform law
    x = np.linspace(1-N, N-1, 100000)
    pdf_x = [pdf_cos_theta(y) for y in x]
    pdf_sum = pdf_x
    for _ in range(N-2):    # Compute the convolution of the pdf with itself N-2 times
        pdf_sum = fftconvolve(pdf_sum, pdf_x, mode='same')
        pdf_sum /= np.trapz(pdf_sum, x)  # Normalize the pdf after convolution
    pdf_sample_mean = (N-1)* pdf_sum # Scale the pdf to get the pdf of the sample mean
    scaled_x = x / (N-1)
    return scaled_x, pdf_sample_mean