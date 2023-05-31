import numpy as np

def jacobi(A,b,tol,kmax,xi, yi):
    N = len(b[0])
    xnew = np.zeros(N)
    xold = np.zeros(N)
    
    # Solución inicial
    xold[0] = xi
    xold[1] = yi
    
    xs = [xi]
    ys = [yi]
    
    e = 10
    error = [] 
    
    k = 0
    print('{:^2} {:^10} {:^12} {:^12}'.format(' i ', 'Error', 'x0', 'x1'))
    while(e > tol and k < kmax) :
        for i in range(0,N): # se puede hacer en paralelo
            xnew[i] = 0
            for j in range(0,i):
                xnew[i] += A[i,j] * xold[j]
            for j in range(i+1,N):
                xnew[i] += A[i,j] * xold[j]                
            xnew[i] = (b[0,i] - xnew[i]) / A[i,i]
        
        # Almacenamos la solución actual
        xs.append(xnew[0])
        ys.append(xnew[1])
        
        e = np.linalg.norm(xnew-xold,1)
        error.append(e)
        k += 1
        xold[:] = xnew[:]
        print('{:2d} {:10.9f} ({:10.9f}, {:10.9f})'.format(k, e, xnew[0], xnew[1]))
    return xnew, np.array(xs), np.array(ys), error, k


def gauss_seidel(A,b,tol,kmax,xi,yi):
    N = len(b[0])
    xnew = np.zeros(N)
    xold = np.zeros(N)
    
    # Solución inicial
    xold[0] = xi
    xold[1] = yi

    xs = [xi]
    ys = [yi]
    
    e = 10
    error = [] 
    
    k = 0
    print('{:^2} {:^10} {:^12} {:^12}'.format(' i ', 'Error', 'x0', 'x1'))
    while(e > tol and k < kmax) :
        for i in range(0,N): # se puede hacer en paralelo
            xnew[i] = 0
            for j in range(0,i):
                xnew[i] += A[i,j] * xnew[j]
            for j in range(i+1,N):
                xnew[i] += A[i,j] * xold[j]                
            xnew[i] = (b[0,i] - xnew[i]) / A[i,i]
            
        # Almacenamos la solución actual
        xs.append(xnew[0])
        ys.append(xnew[1])

        e = np.linalg.norm(xnew-xold,1)
        error.append(e)
        k += 1
        xold[:] = xnew[:]
        print('{:2d} {:10.9f} ({:10.9f}, {:10.9f})'.format(k, e, xnew[0], xnew[1]))
    return xnew, np.array(xs), np.array(ys), error, k

def sor(A,b,tol,kmax,w,xi,yi):
    N = len(b[0])
    xnew = np.zeros(N)
    xold = np.zeros(N)

    # Solución inicial
    xold[0] = xi
    xold[1] = yi

    xs = [xi]
    ys = [yi]
    
    e = 10
    error = [] 
    
    k = 0
    while(e > tol and k < kmax) :
        for i in range(0,N): # se puede hacer en paralelo
            sigma = 0
            for j in range(0,i):
                sigma += A[i,j] * xnew[j]
            for j in range(i+1,N):
                sigma += A[i,j] * xold[j]                
            sigma = (b[0,i] - sigma) / A[i,i]
            xnew[i] = xold[i] + w * (sigma -xold[i])
            
        # Almacenamos la solución actual
        xs.append(xnew[0])
        ys.append(xnew[1])
        
        e = np.linalg.norm(xnew-xold, 1)
        
        error.append(e)
        k += 1
        xold[:] = xnew[:]
        print('{:2d} {:10.9f} ({:10.9f}, {:10.9f})'.format(k, e, xnew[0], xnew[1]))
    return xnew, np.array(xs), np.array(ys), error, k

def steepest(A,b,x,tol,kmax):
    xs, ys = [x[0,0]], [x[1,0]]
    r = b.T - A @ x
    res = np.linalg.norm(r)
    res_list = []
    k = 0
    while(res > tol and k < kmax):
        alpha = r.T @ r / (r.T @ A @ r)
        x = x + r * alpha
        xs.append(x[0,0])
        ys.append(x[1,0])
        r = b.T - A @ x
        res = np.linalg.norm(r,1)
        res_list.append(res)
        k += 1
        print('{:2d} {:10.9f} ({:10.9f}, {:10.9f})'.format(k, res, x[0,0], x[1,0]))
    return x, np.array(xs), np.array(ys), res_list, k

def conjugateGradient(A,b,x,tol,kmax):
    xs, ys = [x[0,0]], [x[1,0]]
    
    r = b.T - A @ x
    d = r
    rk_norm = r.T @ r
    res = np.linalg.norm(rk_norm)
    res_list = []

    k = 0
    while(res > tol and k < kmax):
        alpha = float(rk_norm) / float(d.T @ A @ d)
        x = x + alpha * d
        xs.append(x[0,0])
        ys.append(x[1,0])
        r = r - alpha * A @ d
        res = np.linalg.norm(r)
        res_list.append(res)
        
        rk_old = rk_norm
        rk_norm = r.T @ r
        beta = float(rk_norm) / float(rk_old)
        d = r + beta * d
        k += 1
        print('{:2d} {:10.9f} ({:10.9f}, {:10.9f})'.format(k, res, x[0,0], x[1,0]))
    return x, np.array(xs), np.array(ys), res_list, k
