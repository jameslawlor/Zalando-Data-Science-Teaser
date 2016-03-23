plt.figure()

rbf = scipy.interpolate.Rbf(X, Y, P, function='linear')
z = P
zi = rbf(X, Y)

plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
           extent=[X.min(), X.max(), Y.min(), Y.max()],
             cmap=plt.get_cmap('hot'))
plt.grid(which='major', axis='both', linestyle='-', color='k')

# Brandenburg Gate point
bbg_x, bbg_y = t(52.516288,13.377689)
plt.plot(bbg_x,bbg_y,'o',color='r',markersize=15)

# Satellite path
sat_x_start, sat_y_start = t( 52.437385, 13.553989)
sat_x_end, sat_y_end= t(52.590117, 13.39915)
plt.plot([sat_x_start,sat_x_end], [sat_y_start, sat_y_end],
             color='r', linestyle='-', linewidth=5)

#Spree path
coords = np.loadtxt('spree_coords.dat',delimiter=',',dtype=float) 
segments = [np.array([np.array(t(coords[q][0],coords[q][1])),
                np.array(t(coords[q+1][0],coords[q+1][1]))])
                 for q in range(len(coords)-1)]    
for s in segments:
    plt.plot( [s[0][0],s[1][0]],[s[0][1],s[1][1]] ,
            color = 'g', linestyle='-', linewidth=5    )

# Points of Interest
x, y = t(52.4911, 13.4948)  # Thermal plant
plt.plot(x,y,'v',color='b',markersize=15)
x, y = t(52.5094, 13.4367)  # TU Berlin
plt.plot(x,y,'v',color='b',markersize=15)#
x, y = t(52.5247, 13.3222)  # Zalando
plt.plot(x,y,'v',color='b',markersize=15)
plt.xlabel('X (km)')
plt.ylabel('Y (km)')
cb = plt.colorbar()
cb.set_label("Probability")
plt.show()
