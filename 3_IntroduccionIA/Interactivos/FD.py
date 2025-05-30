#!/usr/bin/env python
# coding: utf-8

# Cálculo de derivadas numéricas
# Autor: Luis M. de la Cruz Salas
# Rev: mar jul  7 11:12:13 CDT 2020


import numpy as np
from pandas import DataFrame
from sympy import Symbol, diff, sin, cos, exp
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt
    
def forwardFD(u,x,h):
    """
    Esquema de diferencias finitas hacia adelante.
    
    Parameters
    ----------
    u : función. 
    Función a evaluar.
    
    x : array
    Lugar(es) donde se evalúa la función
    
    h : array
    Tamaño(s) de la diferencia entre u(x+h) y u(x).
    
    Returns
    -------
    Cálculo de la derivada numérica hacia adelante.
    """
    return (u(x+h)-u(x))/h

def backwardFD(u,x,h):
    """
    Esquema de diferencias finitas hacia atrás.
    
    Parameters
    ----------
    u : función. 
    Función a evaluar.
    
    x : array
    Lugar(es) donde se evalúa la función
    
    h : array
    Tamaño(s) de la diferencia entre u(x+h) y u(x).
    
    Returns
    -------
    Cálculo de la derivada numérica hacia atrás.
    """
    return (u(x)-u(x-h))/h


def centeredFD(u,x,h):
    """
    Esquema de diferencias finitas centradas.
    
    Parameters
    ----------
    u : función. 
    Función a evaluar.
    
    x : array
    Lugar(es) donde se evalúa la función
    
    h : array
    Tamaño(s) de la diferencia entre u(x+h) y u(x).
    
    Returns
    -------
    Cálculo de la derivada numérica centrada.
    """
    return (u(x+h)-u(x-h))/(2*h)

def line(x,x0,x1,y0,y1):
    """
    Dados dos puntos en el plano, cálcula la pendiente de
    la recta que pasa por esos puntos y regresa los puntos
    que pasan por esa recta.
    """
    return ((y1-y0)/(x1-x0))*(x-x0) + y0

def numericalDer(f, x0, h, aprox = 'All'):
    t = Symbol('x') # defino la variable dependiente
    fp = diff(f,t)  # calcula la derivada de la función f (simbólica)

    # Genero funciones lambda para evaluar la función y su derivada
    evalfp = lambdify(t, fp, modules=['numpy']) # la derivada
    evalf = lambdify(t, f, modules=['numpy']) # la función
    
    fp_value = evalfp(x0) # evalúo la derivada en x0
    
    # Calculo el error de las aproximacions y el valor exacto
    ef = np.fabs(fp_value - forwardFD(evalf, x0, h))
    eb = np.fabs(fp_value - backwardFD(evalf, x0, h))
    ec = np.fabs(fp_value - centeredFD(evalf, x0, h))
              
    # Arreglos para las gráficas de la línea y la función.
    xl = np.linspace(x0 - 0.5 * np.pi, x0 + 0.5 * np.pi, 50)
    xv = np.linspace(x0 - np.pi, x0 + np.pi, 50)
    
    # Coordenaas verticales de las líneas de cada aproximación
    lf = line(xl, x0, x0 + h, evalf(x0), evalf(x0 + h))
    lb = line(xl, x0, x0 - h, evalf(x0), evalf(x0 - h))
    lc = line(xl, x0-h, x0+h, evalf(x0-h), evalf(x0+h))

    # Punto donde se aproxima y el punto vecino.
    yv = evalf(xv)
    yp = evalfp(xv)

    ancho_linea = 2.0
    
    fig = plt.figure(figsize=(10,4))
    fig.suptitle('EF = {:10.4e}, EB = {:10.4e}, EC = {:10.4e}'.format(ef,eb,ec))
    
    ax1 = plt.subplot(1,2,1)
    plt.plot(xv, yv, '--', lw = 3, color='b', label = 'f = {}'.format(f))
    plt.scatter(x0, evalf(x0), facecolor ='b', edgecolor='k', zorder=10)
    
    if aprox == 'All':
        plt.scatter(x0+h, evalf(x0+h), facecolor ='w', edgecolor='k', zorder=10)
        plt.scatter(x0-h, evalf(x0-h), facecolor ='w', edgecolor='k', zorder=10)
        plt.scatter(x0-2*h, evalf(x0-2*h), facecolor ='w', edgecolor='k', zorder=10)    
        plt.plot(xl, lf, lw = ancho_linea, label="f' = Forward")
        plt.plot(xl, lb, lw = ancho_linea, label="f' = Backward") 
        plt.plot(xl, lc, lw = ancho_linea, label="f' = Centered")
        
    elif aprox == 'Forward':
        plt.scatter(x0+h, evalf(x0+h), facecolor ='w', edgecolor='k', zorder=10)   
        plt.plot(xl, lf, lw = ancho_linea, label="f' = Forward")
        
    elif aprox == 'Backward':
        plt.scatter(x0-h, evalf(x0-h), facecolor ='w', edgecolor='k', zorder=10)
        plt.plot(xl, lb, lw = ancho_linea, label="f' = Backward")
        
    elif aprox == 'Centered':
        plt.scatter(x0+h, evalf(x0+h), facecolor ='w', edgecolor='k', zorder=10)   
        plt.scatter(x0-h, evalf(x0-h), facecolor ='w', edgecolor='k', zorder=10)
        plt.plot(xl, lc, lw = ancho_linea, label="f' = Centered")
        
    plt.legend(ncol=2, loc=(0.0,-0.30))


    ax2 = plt.subplot(1,2,2)
    plt.plot(xv, yp, '--', lw = 3, color='k', label = "f'= {}".format(fp))
    
    if aprox == 'All':
        plt.plot(xv, forwardFD(evalf, xv, h), lw = ancho_linea)
        plt.plot(xv, backwardFD(evalf, xv, h), lw = ancho_linea)
        plt.plot(xv, centeredFD(evalf, xv, h), lw = ancho_linea)
    elif aprox == 'Forward':
        plt.plot(xv, forwardFD(evalf, xv, h), lw = ancho_linea)
    elif aprox == 'Backward':
        plt.plot(xv, backwardFD(evalf, xv, h), lw = ancho_linea)
    elif aprox == 'Centered':
        plt.plot(xv, centeredFD(evalf, xv, h), lw = ancho_linea)
        
    plt.legend()
    
    plt.show()
    return [ef, eb, ec]

#----------------------- TEST OF THE MODULE ----------------------------------   
if __name__ == '__main__':
    
    # Definimos un arreglo con diferentes tamaños de h:
    N = 5
    h = np.zeros(N)

    h[0] = 1.0
    for i in range(1,N):
        h[i] = h[i-1] * 0.5

    # Definimos un arreglo con valores de 1.0 (donde evaluaremos el cos(x)):
    x = np.ones(N)
    
    aprox = np.array([h,
             np.cos(x), 
             forwardFD(np.sin,x,h), 
             backwardFD(np.sin,x,h), 
             centeredFD(np.sin,x,h)])
    
    
    daprox = DataFrame(aprox.transpose(), 
                       columns = ['h', 'original', 'forward', 'backward', 'centered'])
    print(daprox)
    
    t = Symbol('x')
    print(numericalDer(sin(t) * t*t, x[0], h[0], 'Forward'))