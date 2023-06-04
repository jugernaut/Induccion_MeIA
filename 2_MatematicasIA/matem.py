import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

def plot_vectors(vecs, lvecs = None, baseline = [], w = 0.01, ax = None, aspect='equal', limit=True):
    xmax, xmin = 0, 0
    ymax, ymin = 0, 0
    
    if ax == None:
        ax = plt.gca()
        
    for i, x in enumerate(vecs):
        color = 'C' + str(i)
        xmax = xmax if xmax > x[0] else x[0]
        ymax = ymax if ymax > x[1] else x[1]
        xmin = xmin if xmin < x[0] else x[0]
        ymin = ymin if ymin < x[1] else x[1]
        
        if len(baseline) == 0:
            x0, y0 = 0, 0
        else:
            x0 = baseline[i][0]
            y0 = baseline[i][1]

        if lvecs != None:
            ax.quiver(x0, y0, x[0], x[1], angles='xy', scale_units='xy', scale=1, width=w, color=color, label=lvecs[i])
        else:
            ax.quiver(x0, y0, x[0], x[1], angles='xy', scale_units='xy', scale=1, width=w, color=color)

    if limit:
        xoff = np.fabs(xmax - xmin) * 0.1
        yoff = np.fabs(ymax - ymin) * 0.1
        ax.set_xlim(xmin-xoff, xmax+xoff)
        ax.set_ylim(ymin-yoff, ymax+yoff)

    ax.set_aspect(aspect)
        
    if lvecs != None:
        ax.legend(loc='best', bbox_to_anchor=(0.75, 0., 0.5, 0.5))
#    plt.show()
    
def graf_surf(xg, yg, z, elev=5, azim=-30):
    ax = plt.axes(projection='3d')
    ax.plot_surface(xg,yg,z, alpha=0.95, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.view_init(elev, azim)

def graf_surfi(xg, yg, z):
    w = widgets.interactive(graf_surf,
                    xg = widgets.fixed(xg), yg = widgets.fixed(yg), z = widgets.fixed(z), 
                    elev = widgets.FloatSlider(min=0, max=180, value = 5, step=5),
                    azim = widgets.FloatSlider(min=0, max=180, value = -30, step=5))
    
    return w

def eigen_land(Mat):
    # Cálculo de eigenvectores
    w, v = np.linalg.eig(Mat)  # w: eigenvalues, v: eigenvectors

    # Impresión de los eigenvalores y eigenvectores
    print('eigenvalores = {}'.format(w))
    print('eigenvectores:\n {} \n {}'.format(v[:,0], v[:,1]))

    # Cálculo del ángulo entre los vectores.
    e0 = v[:,0] / np.linalg.norm(v[:,0])
    e1 = v[:,1] / np.linalg.norm(v[:,1])
    angulo = np.arccos(np.dot(e0, e1)) * 180 / np.pi
    print('ángulo entre eigenvectores = {}'.format(angulo)) 
    
    return w, v

def print_Aulu(A, w, v):
    for i, l in enumerate(w):
        print(chr(119860) + chr(119906) + ' = {}'.format(A @ v[:,i]))
        print(chr(120582) + chr(119906) + ' = {}'.format(l * v[:,i]))
        print()