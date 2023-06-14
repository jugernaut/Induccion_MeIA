import numpy as np
import matplotlib
from numpy import ma
from matplotlib import cbook
from matplotlib.colors import Normalize


# Documentación en
# http://matplotlib.org/api/colors_api.html#matplotlib.colors.LinearSegmentedColormap
cdict = {'red': ((0.0, 0.0, 0.1),
                 (0.5, 1.0, 0.85),
                 (1.0, 1.0, 1.0)),
         'green': ((0.0, 0.0, 0.95),
                   (0.5, 1.0, 0.0),
                   (1.0, 0.5, 1.0)),
         'blue': ((0.0, 0.0, 1.0),
                  (0.5, 1.0, 1.0),
                  (1.0, 1.0, 1.0))}
bigradient = matplotlib.colors.LinearSegmentedColormap('bigradient_colormap', cdict, 256)



# Ligeramente modificada de:
# http://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

class MidPointNorm(Normalize):
    """ Normaliza los valores alrededor del punto medio indicado.
    """
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self, vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if vmin > midpoint:
            vmin = midpoint - 0.001
        if vmax < midpoint:
            vmax = midpoint + 0.001

        if not (vmin <= midpoint <= vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint
            resdat[resdat>0] /= abs(vmax - midpoint)
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = ma.array(resdat, mask=result.mask, copy=False)

        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        #if cbook.iterable(value):
        if np.iterable(value):
            val = ma.asarray(value)
            val = 2 * (val-0.5)
            val[val>0]  *= abs(vmax - midpoint)
            val[val<0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            value = 2 * (value - 0.5)
            if value < 0:
                return  value*abs(vmin-midpoint) + midpoint
            else:
                return  value*abs(vmax-midpoint) + midpoint


#
# Funciones para graficar el estado de la red neuronal
#
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json, matplotlib
plt.style.use('ggplot')
from IPython.core.pylabtools import figsize
figsize(11, 5)
colores = ["#348ABD", "#A60628","#06A628"]


import warnings
from .data import personas, relaciones

def plotNetworkActivity(redHinton, iEntrada, pyTorch=False):
    """ Grafica los valores de activación de cada neurona
    para la entrada en la columna iEntrada.
    """
    if pyTorch:
        if iEntrada > redHinton.a_1.size()[0]:
            raise IndexError("Ejemplar de entrenamiento inexistente " + str(iEntrada))
    else:
        if(iEntrada > redHinton.a_1.shape[1]):
            raise IndexError("Ejemplar de entrenamiento inexistente " + str(iEntrada))
    nRens = 5
    nCols = 2

    fig, axes = plt.subplots(figsize=(12,8))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    ax_a1 = plt.subplot2grid((nRens,nCols), (4,0))
    ax_a2 = plt.subplot2grid((nRens,nCols), (4,1))
    ax_b1 = plt.subplot2grid((nRens,nCols), (3,0))
    ax_b2 = plt.subplot2grid((nRens,nCols), (3,1))
    ax_c = plt.subplot2grid((nRens,nCols), (2,0), colspan=2)
    ax_d = plt.subplot2grid((nRens,nCols), (1,0), colspan=2)
    ax_e = plt.subplot2grid((nRens,nCols), (0,0), colspan=2)

    #fig, axes = plt.subplots(nRens, nCols, figsize=(12,3), sharex='col')

    if pyTorch:
        a1 = redHinton.a_1[iEntrada,:].detach().numpy()
        a2 = redHinton.a_2[iEntrada,:].detach().numpy()
        b1 = redHinton.b_1[iEntrada,:].detach().numpy()
        b2 = redHinton.b_2[iEntrada,:].detach().numpy()
        c = redHinton.c[iEntrada,:].detach().numpy()
        d = redHinton.d[iEntrada,:].detach().numpy()
        e = redHinton.e[iEntrada,:].detach().numpy()
    else:
        a1 = redHinton.a_1[:,iEntrada]
        a2 = redHinton.a_2[:,iEntrada]
        b1 = redHinton.b_1[:,iEntrada]
        b2 = redHinton.b_2[:,iEntrada]
        c = redHinton.c[:,iEntrada]
        d = redHinton.d[:,iEntrada]
        e = redHinton.e[:,iEntrada]

    # A1
    N = np.vstack((a1[0:12],a1[12:]))
    #print("a_1 = ", N)
    ax_a1.pcolormesh(N, cmap=cm.cool, norm=norm)
    ax_a1.set_xticks(np.arange(12) + 0.5)
    ax_a1.set_xticklabels(personas[0:12], minor=False, rotation=90, ha='center')
    ax_a1.set_yticks(np.arange(2))

    topax_a1 = ax_a1.twiny()
    topax_a1.set_xlim(0, 12)
    topax_a1.set_xticks(np.arange(12) + 0.5)
    topax_a1.set_xticklabels(personas[12:], minor=False, rotation=90, ha='center')

    # A2
    N = a2[np.newaxis]
    #print("a_2 = ", N)
    ax_a2.pcolormesh(N, cmap=cm.cool, norm=norm)
    ax_a2.set_xticks(np.arange(12) + 0.5)
    ax_a2.set_xticklabels(relaciones[0:12], minor=False, rotation=90, ha='left')
    ax_a2.set_yticks(np.arange(1))

    # B1
    N = b1[np.newaxis]
    #print("b_1 = ", N)
    ax_b1.pcolormesh(N, cmap=cm.cool, norm=norm)
    ax_b1.set_xticks(np.arange(6) + 0.5)
    ax_b1.set_xticklabels(np.arange(6))
    ax_b1.set_yticks(np.arange(1))

    # B2
    N = b2[np.newaxis]
    #print("b_2 = ", N)
    ax_b2.pcolormesh(N, cmap=cm.cool, norm=norm)
    ax_b2.set_xticks(np.arange(6) + 0.5)
    ax_b2.set_xticklabels(np.arange(6))
    ax_b2.set_yticks(np.arange(1))

    # C
    N = c[np.newaxis]
    #print("c = ", N)
    ax_c.pcolormesh(N, cmap=cm.cool, norm=norm)
    ax_c.set_xticks(np.arange(12) + 0.5)
    ax_c.set_xticklabels(np.arange(12))
    ax_c.set_yticks(np.arange(1))

    # D
    N = d[np.newaxis]
    #print("d = ", N)
    ax_d.pcolormesh(N, cmap=cm.cool, norm=norm)
    ax_d.set_xticks(np.arange(6) + 0.5)
    ax_d.set_xticklabels(np.arange(6))
    ax_d.set_yticks(np.arange(1))

    # E
    N = np.vstack((e[0:12],e[12:]))
    #print("e = ", N)
    ax_e.pcolormesh(N, cmap=cm.cool, norm=norm)
    ax_e.set_xticks(np.arange(12) + 0.5)
    ax_e.set_xticklabels(personas[0:12], minor=False, ha='center')
    ax_e.set_yticks(np.arange(2))
    topax_e = ax_e.twiny()
    topax_e.set_xlim(0, 12)
    topax_e.set_xticks(np.arange(12) + 0.5)
    topax_e.set_xticklabels(personas[12:], minor=False, ha='center')
    ax_e.set_title("Nivel de activación de cada neurona", y = 1.75)


    # Barra de color
    ax1 = fig.add_axes([1.0, 0, 0.025, 1.0]) # left, bottom, width, height
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cm.cool,
                                norm=norm,
                                orientation='vertical')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()


# Grafiquemos los valores de los pesos que conectan a las neuronas.

def muestraPesosWa_1(redHinton, pyTorch=False):
    """
    Crea una gráfica de colores, representando los valores en la matriz de pesos Wa_1
    :param redHinton:
    :return:
    """
    if pyTorch:
        Wa_1 = redHinton.zb_1.weight.data.detach().numpy()
    else:
        Wa_1 = redHinton.Wa_1

    fig, axes = plt.subplots(3,2, figsize=(12,3), sharex=True)
    norm = MidPointNorm(0)
    norm.autoscale(Wa_1)

    # Barra de color
    ax1 = fig.add_axes([1.0, 0, 0.025, 1.0]) # left, bottom, width, height
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=bigradient,
                                norm=norm,
                                orientation='vertical')


    N1 = np.vstack((Wa_1[0:1,0:12], Wa_1[0:1,12:24]))
    axes[0][0].pcolormesh(N1, cmap=bigradient, norm=norm)

    N2 = np.vstack((Wa_1[1:2,0:12], Wa_1[1:2,12:24]))
    axes[0][1].pcolormesh(N2, cmap=bigradient, norm=norm)
    N3 = np.vstack((Wa_1[2:3,0:12], Wa_1[2:3,12:24]))
    axes[1][0].pcolormesh(N3, cmap=bigradient, norm=norm)
    N4 = np.vstack((Wa_1[3:4,0:12], Wa_1[3:4,12:24]))
    axes[1][1].pcolormesh(N4, cmap=bigradient, norm=norm)
    N5 = np.vstack((Wa_1[4:5,0:12], Wa_1[4:5,12:24]))
    axes[2][0].pcolormesh(N5, cmap=bigradient, norm=norm)
    N6 = np.vstack((Wa_1[5:,0:12], Wa_1[5:,12:24]))
    axes[2][1].pcolormesh(N6, cmap=bigradient, norm=norm)

    #labels
    for i in range(2):
        newax = axes[0][i].twiny()
        newax.set_xlabel('Ingleses', color='red')
        newax.patch.set_visible(False)
        newax.xaxis.set_ticks_position('top')
        newax.xaxis.set_label_position('top')
        newax.set_xticks(np.arange((len(personas) + 1)//2))
        newax.set_xticklabels(personas[0:12], minor=False, rotation=90, ha='left')

        axes[2][i].set_xlabel('Italianos', color='red')
    for axRow in axes:
        axRow[0].set_xticks(np.arange(len(personas)/2))
        axRow[0].set_xticklabels(personas[0:12], minor=False, rotation=90, ha='left')
        axRow[1].set_xticks(np.arange(len(personas)/2))
        axRow[1].set_xticklabels(personas[12:], minor=False, rotation=90, ha='left')

    plt.title("Pesos de $A_1$ a $B_1$\nCada caja es una neurona en $B_1$", x = 0, y = 2.7)


def muestraPesosWa_2(redHinton, pyTorch=False):
    """
    Crea una gráfica de colores, representando los valores en la matriz de pesos Wa_2
    :param redHinton:
    :return:
    """
    if pyTorch:
        Wa_2 = redHinton.zb_2.weight.data.detach().numpy()
    else:
        Wa_2 = redHinton.Wa_2

    fig, axes = plt.subplots(3,2, figsize=(12,1.5), sharex=True, sharey=True)
    norm = MidPointNorm(0)
    norm.autoscale(Wa_2)

    plt.title("Pesos de $A_2$ a $B_2$\nCada caja es una neurona en $B_2$", x=0, y=3.5)

    # Barra de color
    ax1 = fig.add_axes([1.0, 0, 0.025, 1.0]) # left, bottom, width, height
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=bigradient,
                                norm=norm,
                                orientation='vertical')

    axes[0][0].pcolormesh(Wa_2[0:1,:], cmap=bigradient, norm=norm)
    axes[0][1].pcolormesh(Wa_2[1:2,:], cmap=bigradient, norm=norm)
    axes[1][0].pcolormesh(Wa_2[2:3,:], cmap=bigradient, norm=norm)
    axes[1][1].pcolormesh(Wa_2[3:4,:], cmap=bigradient, norm=norm)
    axes[2][0].pcolormesh(Wa_2[4:5,:], cmap=bigradient, norm=norm)
    axes[2][1].pcolormesh(Wa_2[5:,:], cmap=bigradient, norm=norm)

    #labels
    for i in range(2):
        axes[2][i].set_xlabel('Relaciones', color='red')
        axes[2][i].set_yticks(np.arange(1))
        axes[2][i].set_xticks(np.arange(len(relaciones)) + 0.5)
        axes[2][i].set_xticklabels(relaciones, minor=False, rotation=90, ha='left')


def muestraPesosWc(redHinton, pyTorch=False):
    """
    Crea una gráfica de colores, representando los valores en la matriz de pesos Wa_2
    :param redHinton:
    :return:
    """
    if pyTorch:
        Wc = redHinton.zd.weight.data.detach().numpy()
    else:
        Wc = redHinton.Wc

    fig, axes = plt.subplots(3,2, figsize=(12,1.5), sharex=True, sharey=True)
    norm = MidPointNorm(0)
    norm.autoscale(Wc)

    plt.title("Pesos de $C$ a $D$\nCada caja es una neurona en $D$", x=0, y=3.5)

    # Barra de color
    ax1 = fig.add_axes([1.0, 0, 0.025, 1.0]) # left, bottom, width, height
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=bigradient,
                                norm=norm,
                                orientation='vertical')

    axes[0][0].pcolormesh(Wc[0:1,:], cmap=bigradient, norm=norm)
    axes[0][1].pcolormesh(Wc[1:2,:], cmap=bigradient, norm=norm)
    axes[1][0].pcolormesh(Wc[2:3,:], cmap=bigradient, norm=norm)
    axes[1][1].pcolormesh(Wc[3:4,:], cmap=bigradient, norm=norm)
    axes[2][0].pcolormesh(Wc[4:5,:], cmap=bigradient, norm=norm)
    axes[2][1].pcolormesh(Wc[5:,:], cmap=bigradient, norm=norm)

    #labels
    for i in range(2):
        axes[2][i].set_xlabel('Neurona en D')
        axes[2][i].set_yticks(np.arange(1))
        axes[2][i].set_xticks(np.arange(len(relaciones)) + 0.5)
        axes[2][i].set_xticklabels(np.arange(len(relaciones)), minor=False, ha='left')


def muestraPesosWd(redHinton, pyTorch=False):
    """
    Crea una gráfica de colores, representando los valores en la matriz de pesos Wd
    :param redHinton:
    :return:
    """
    if pyTorch:
        Wd = redHinton.ze.weight.data.detach().numpy()
    else:
        Wd = redHinton.Wd

    fig, axes = plt.subplots(3,2, figsize=(12,3), sharex=True)
    norm = MidPointNorm(0)
    norm.autoscale(Wd)

    # Barra de color
    ax1 = fig.add_axes([1.0, 0, 0.025, 1.0]) # left, bottom, width, height
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=bigradient,
                                norm=norm,
                                orientation='vertical')


    N1 = np.vstack((Wd[0:12,0:1].T, Wd[12:24,0:1].T))
    axes[0][0].pcolormesh(N1, cmap=bigradient, norm=norm)

    N2 = np.vstack((Wd[0:12,1:2].T, Wd[12:24,1:2].T))
    axes[0][1].pcolormesh(N2, cmap=bigradient, norm=norm)
    N3 = np.vstack((Wd[0:12,2:3].T, Wd[12:24,2:3].T))
    axes[1][0].pcolormesh(N3, cmap=bigradient, norm=norm)
    N4 = np.vstack((Wd[0:12,3:4].T, Wd[12:24,3:4].T))
    axes[1][1].pcolormesh(N4, cmap=bigradient, norm=norm)
    N5 = np.vstack((Wd[0:12,4:5].T, Wd[12:24,4:5].T))
    axes[2][0].pcolormesh(N5, cmap=bigradient, norm=norm)
    N6 = np.vstack((Wd[0:12,5:].T, Wd[12:24,5:].T))
    axes[2][1].pcolormesh(N6, cmap=bigradient, norm=norm)

    #labels
    for i in range(2):
        newax = axes[0][i].twiny()
        newax.set_xlabel('Ingleses', color='red')
        newax.patch.set_visible(False)
        newax.xaxis.set_ticks_position('top')
        newax.xaxis.set_label_position('top')
        newax.set_xticks(np.arange((len(personas) + 1)//2))
        newax.set_xticklabels(personas[0:12], minor=False, rotation=90, ha='left')

        axes[2][i].set_xlabel('Italianos', color='red')
    for axRow in axes:
        axRow[0].set_xticks(np.arange(len(personas)/2))
        axRow[0].set_xticklabels(personas[0:12], minor=False, rotation=90, ha='left')
        axRow[1].set_xticks(np.arange(len(personas)/2))
        axRow[1].set_xticklabels(personas[12:], minor=False, rotation=90, ha='left')

    plt.title("Pesos de $D$ a $E$\nCada caja es una neurona en $D$", x = 0, y = 2.7)


def muestraPesos(redHinton, pyTorch=False):
    muestraPesosWa_1(redHinton, pyTorch)
    muestraPesosWa_2(redHinton, pyTorch)
    muestraPesosWc(redHinton, pyTorch)
    muestraPesosWd(redHinton, pyTorch)
