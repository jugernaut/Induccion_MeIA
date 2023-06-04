import matplotlib.pyplot as plt

def graf_rectas(x, y0, y1, label0 = '$3x_0+2x_1=2$', label1='$2x_0+6x_1=-8$', aspect='equal'):
    plt.plot(x,y0,lw=3,c='seagreen',label = label0)
    plt.plot(x,y1,lw=3,c='mediumorchid',label = label1)
    if aspect == 'equal':
        plt.gca().set_aspect('equal')
        plt.legend(ncol = 1, frameon=True, loc='best', bbox_to_anchor=(0.95, 1.0))
    else:
        plt.legend(ncol = 2, frameon=True)

    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    plt.title('Cruce de rectas')
    plt.grid(True)

def graf_sol(sol):
    plt.scatter(sol[0], sol[1], fc='sandybrown', ec='k', s = 75, alpha=0.75, zorder=5, label='Solución')

def graf_eigen(xg, yg, u, v):  
    vec = plt.quiver(xg,yg,u,v,scale=10, zorder=6)
    
def graf_fcuad(sol, xg, yg, z, elev = 5, azim = -30, levels = 0, cmap='viridis'): 
    if levels == 0:
        ax = plt.axes(projection='3d')
        ax.plot_surface(xg, yg, z, cmap=cmap, alpha=0.75)
        ax.scatter(sol[0], sol[1], fc='sandybrown', ec='k', s = 75, alpha=0.75, zorder=5, label='Solución')
        ax.view_init(elev, azim)
        ax.set_xlabel('$x_0$')
        ax.set_ylabel('$x_1$')
        ax.set_zlabel('$f$')
    else:
        plt.contour(xg,yg,z,levels, cmap=cmap, linewidths=1.0, zorder=1)        

def graf_pasos(xsol, xs, ys, marker=False):
    plt.plot(xs, ys, '--', c='grey', lw=1.0, zorder=8)
    if marker:
        plt.scatter(xs[0], ys[0], fc='yellow', ec='k', s = 75, alpha=0.75, zorder=8, label='Sol. Inicial')
        plt.scatter(xs[1:], ys[1:], c='navy', s = 10, alpha=0.5, zorder=8)

def graf_error(d_error):
    max_iter = 0
    for i in d_error.keys():
        iter_k = len(d_error[i])
        max_iter = iter_k if max_iter < iter_k else max_iter
        plt.plot(list(range(1,iter_k+1)), d_error[i], '.-', label=i)
        plt.title('Error', loc='left')
    plt.yscale('log')
    plt.xlabel('Iteraciones')
    plt.xticks(ticks=list(range(1,max_iter+1)))
    plt.legend()

def savefig(filename):
    plt.savefig(filename)
    
    
