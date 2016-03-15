#!/usr/bin/env python
"""
=================================================
    Zalando Data Science Teaser
=================================================
"""
# Author: James Lawlor <jalawlor@tcd.ie>
# Date: 2015-01-28

import numpy as np
from scipy.stats import lognorm, norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy.interpolate
from scipy import optimize

def t(xx,yy,inv=False):
    """
    Coordinate transform between Lat/Lon <--> Cartesian
    accurate for the Berlin Area.
    """
    SW_lat = 52.464011 #(Latitude)
    SW_lon = 13.274099 #(Longitude)

    if inv:     # Lat/Lon -> Cartesian
        lon = SW_lon + (xx/111.323)/(np.cos(SW_lat*np.pi/180.0))
        lat = (yy/111.323) + SW_lat
        return lat, lon
    else:       # Cartesian -> Lat/Lon
        x = (yy - SW_lon) * np.cos(SW_lat * np.pi / 180.0) * 111.323
        y = (xx - SW_lat) * 111.323
        return x, y

def euclid(x,y,x0=0,y0=0):
    """
    Euclidean distance between cartesian points (x,y) and (x0,y0)
    """
    return np.sqrt((x-x0)**2 + (y-y0)**2)

def p_bbg(x,y,
        center_x = 52.516288,
        center_y = 13.377689 ):

    """
    Log-normal distribution around the Brandenburg Gate (BBG)
    Returns the probability of the x,y input being the analyst
    location according to the BBG source.
    """
    x0 , y0 = t(center_x, center_y) # Transform lat,lon of the BBG to cartesian
    # parameters for the distribution, see accompanying pdf document
    mu = 1.4834  ; sigma = 0.3582  
    ln_bb = lognorm(s=sigma, scale = np.exp(mu)) # Creates log-norm distribution
    return ln_bb.pdf(euclid(x,y,x0,y0)) # Return distribution val at (x,y)

def d_to_seg(xx,yy,p0,p1):

    """
    Calculates Euclidean distance between a point (xx,yy)
    and a line segment with start & end points p0, p1
    for details of how this works please see the attached PDF
    """
    def g(x0,y0):   
        v = p1 - p0                     # Vector of line
        w = np.array([x0,y0]) - p0      # Vector from point to line start point
        
        v1 = np.dot(w,v)
        v2 = np.dot(v,v)
        # Point lies to left of line, closest point is p0
        if v1 <= 0.0:    return euclid(x0,y0,p0[0],p0[1])  
        # Point lies to right, closest point is p1.
        if v2 <= v1:     return euclid(x0,y0,p1[0],p1[1])   

        b = v1/v2             # If neither if statement is satisfied, closest
        pb = p0 + b*v         #    segment point must lie between p0 and p1
        return euclid(x0,y0,pb[0],pb[1])

    # As we work with meshgrid, this allows func to operate
    # on all XY values simultaneously
    vf = np.vectorize(g)     
    return vf(xx,yy)

def p_sat(x,y,
            start_x = 52.437385, start_y = 13.553989,
            end_x =52.590117, end_y = 13.39915):

    """
    normal distribution along satellite path, returns value of distribution at
    point (x,y) in Cartesian coordinates given satellite path between 
    (start_x,start_y) and (end_x,end_y) in Lat/Lon coordinate system.
    """
    sgm =  1.22451    # SD of norm for this data source. see pdf doc
    # Transform start and end points of satellite path to cartesian
    start_x , start_y = t(start_x,start_y)  
    end_x, end_y = t(end_x,end_y)
    #calculate distance d from (x,y) to path
    d = d_to_seg(x,y,np.array([start_x, start_y]), np.array([end_x, end_y]))    
    norm_sat = norm(0, sgm) # Create normal distribution
    return norm_sat.pdf(d)  # Return value of point (x,y) at distance d

def p_spree(x,y):

    """
    Normal distribution along the Spree.
    Similar to p_sat except we work with several joined line segments
    """ 
    # Load coords from external data file
    coords = np.loadtxt('spree_coords.dat',delimiter=',',dtype=float)  
    #Produce line segment representation of the spree:
    segments = [np.array(
                    [np.array(t(coords[q][0],coords[q][1])),
                    np.array(t(coords[q+1][0],coords[q+1][1]))]
                ) for q in range(len(coords)-1)]    
    sgm = 1.39289   # SD of norm, see PDF document

    # function measures (x,y) to all Spree line segments,
    # returns shortest distance
    def spree_f(xs,ys):
        return min([d_to_seg(xs,ys,s[0],s[1]) for s in segments ])          

    v_spree_f = np.vectorize(spree_f)
    d = v_spree_f(x,y)
    norm_spree = norm(0,sgm)
    return norm_spree.pdf(d)    

if __name__ == "__main__":
    
    delta = 5.0e-1
   
    # Build grid of XY cartesian coordinates 
    X, Y = np.meshgrid(np.arange(0.0, 20.01, delta),
             np.arange(-5.0, 15.01, delta))
    # Find the 3 distributions
    Z_spree = p_spree(X,Y)
    Z_sat = p_sat(X,Y)
    Z_bbg = p_bbg(X,Y)
    # Combined probability P 
    P = (Z_spree + Z_sat + Z_bbg )/3.0

    #################################
    # Optimisation
    #################################
    #################################
    def prob(z):
        X,Y = z
        Z_spree = p_spree(X,Y)
        Z_sat = p_sat(X,Y)
        Z_bbg = p_bbg(X,Y)
        return 1.0 - (Z_spree + Z_sat + Z_bbg)/3.0
    # Heizkraftwerk
    rranges = (slice(-0.0, 20.0, delta), slice(-5.0, 15.0, delta)) 
    # TU Berlin
    #rranges = (slice(-0.0, 10.0, delta), slice(-0.0, 10.0, delta)) 
    # Zalando CC office
    #rranges = (slice(-5.0, 14.0, delta), slice(-5.0, 15.0, delta))
    resbrute = optimize.brute(prob, rranges, args=(), full_output=True,
                                   finish=optimize.fmin)
    print 'Global Max:'
    print 'P = ', 1-resbrute[1]
    print 'Cartesian Coords = ', resbrute[0][0], resbrute[0][1]
    print 'Lat/Lon = ', t(resbrute[0][0], resbrute[0][1],True)

    ###########################################
    ###########################################
    #            HEAT MAP PLOT
    ###########################################
    ###########################################
#    
#    plt.figure()
#
#    rbf = scipy.interpolate.Rbf(X, Y, P, function='linear')
#    z = P
#    zi = rbf(X, Y)
#
#    plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
#               extent=[X.min(), X.max(), Y.min(), Y.max()],
#                 cmap=plt.get_cmap('hot'))
#    plt.grid(which='major', axis='both', linestyle='-', color='k')
#
#    # Brandenburg Gate point
#    bbg_x, bbg_y = t(52.516288,13.377689)
#    plt.plot(bbg_x,bbg_y,'o',color='r',markersize=15)
#
#    # Satellite path
#    sat_x_start, sat_y_start = t( 52.437385, 13.553989)
#    sat_x_end, sat_y_end= t(52.590117, 13.39915)
#    plt.plot([sat_x_start,sat_x_end], [sat_y_start, sat_y_end],
#                 color='r', linestyle='-', linewidth=5)
#
#    #Spree path
#    coords = np.loadtxt('spree_coords.dat',delimiter=',',dtype=float) 
#    segments = [np.array([np.array(t(coords[q][0],coords[q][1])),
#                    np.array(t(coords[q+1][0],coords[q+1][1]))])
#                     for q in range(len(coords)-1)]    
#    for s in segments:
#        plt.plot( [s[0][0],s[1][0]],[s[0][1],s[1][1]] ,
#                color = 'g', linestyle='-', linewidth=5    )
#
#    # Points of Interest
#    x, y = t(52.4911, 13.4948)  # Thermal plant
#    plt.plot(x,y,'v',color='b',markersize=15)
#    x, y = t(52.5094, 13.4367)  # TU Berlin
#    plt.plot(x,y,'v',color='b',markersize=15)
#    x, y = t(52.5247, 13.3222)  # Zalando
#    plt.plot(x,y,'v',color='b',markersize=15)
#
#    plt.xlabel('X (km)')
#    plt.ylabel('Y (km)')
#    cb = plt.colorbar()
#    cb.set_label("Probability")
#    plt.show()
