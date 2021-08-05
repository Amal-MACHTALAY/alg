#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 19:28:40 2021

@author: amal
"""

import numpy as np

def compute_residual(w,x,sigma):
    return -func(w)-((func(w+sigma*x)-func(w))/sigma)

def get_Hessenberg_matrix(H,m):
    h=np.zeros((m+1,m))
    for s in range(m):
        for k in range(s+2):
            h[k,s]=H[s][k]
    return h

def AxB(X,Y):
    XY=np.zeros((len(X),len(Y[0])))
    for i in range(len(X)):
    # iterate through columns of Y
        for j in range(len(Y[0])):
        # iterate through rows of Y
            for k in range(len(Y)):
                XY[i][j] += X[i][k] * Y[k][j]
    return XY


# def LU(L:'float32[:,:]',U:'float32[:,:]'):
def LU_decomposition(A):
    """ Compute LU Factorization """
    n = max(len(A),len(A[0]))
    L = np.zeros((n,n),dtype=np.float32)
    for i in range(n):
        L[i,i] = 1
    U = np.zeros((n,n),dtype= np.float32)
    U[:] = A
    # n = len(L)
    for i in range(n):
        p=U[i,i]
        for j in range(i+1,n):
            L[j,i] = (1/p)*U[j,i]
            for k in range(n):
                U[j,k] = U[j,k] - L[j,i]*U[i,k]
    return L,U



def LU_inverse(L,U):
    n = U.shape[0]
    u = U.copy()
    E1 = np.eye(n) # This E1 is used to find the inverse of U
    for j in range(n-1,-1,-1):
        E1[j,:] = E1[j,:]/u[j,j]
        u[j,:] = u[j,:]/u[j,j]
        for i in range(j-1,-1,-1):
            E1[i,:] = E1[i,:]-E1[j,:]*u[i,j]
            u[i,:] = u[i,:]-u[j,:]*u[i,j]
    
    m = L.shape[0]
    l = L.copy()
    E2 = np.eye(m) # This E2 is used to find the inverse of L
    for j in range(m):
        for i in range(j+1,m):
            E2[i,:] = E2[i,:]-E2[j,:]*l[i,j]
            l[i,:] = l[i,:]-l[j,:]*l[i,j]
        
    return E2,E1

def get_preconditionner(A):
    # A: Jacobian of fct, Ignoring the forward-backward coupling  parts
    # the *incomplete LU* decomposition
    L,U=LU_decomposition(A)
    inv_L,inv_U=LU_inverse(L,U)
    M=AxB(inv_U,inv_L)

    return M

def matvec(A,v):
    n =len(A)
    m = len(v)
    Av = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(m):
              s += A[i,j]*v[j]
        Av[i] = s
    return Av

def scalar(u,v):
    n = len(u)
    s = 0.0
    for i in range(n):
        s += u[i]*v[i]
    return s

def vecvec(u,v):
    n = len(u)
    m = len(v)
    uv = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
              uv[i,j]= u[i]*v[j]
    return uv

def norm_two(v):
    n=len(v)
    norm=v[0]**2
    for i in range(1,n):
        norm+=v[i]**2
    norm=np.sqrt(norm)
    return norm
    

def gmres(w0, fct, sigma, tol):
    while True :
        r=-fct(w0)
        v=[]
        v.append(r/norm_two(r))
        M=get_preconditionner(Jacobian(w0))
        H=[]
        j=0
        while True :
            hh=[]
            xj=matvec(M, v[j])
            q=(fct(w0+sigma*xj)-fct(w0))/sigma
            vj=q
            for i in range(j+1):
                hij=scalar(q,v[i])
                hh.append(hij)
                vj-=hij*v[i]
            hjpj=norm_two(vj)
            hh.append(hjpj)
            H.append(hh)
            m=j
            if hjpj==0.0 :
                break
            v.append(vj/hjpj)
            if norm_two(fct(w0)+q)<=tol:
                break
            j+=1
        h=get_Hessenberg_matrix(H,m)
        # calcul of beta*e1
        beta=np.zeros(m)
        beta[0]=norm_two(r)
        y=np.linalg.lstsq(h.transpose(),beta,rcond=-1)[0]
        w0_new=w0+matvec(np.asarray(v).transpose(),y)
        if norm_two(fct(w0))<=tol or norm_two(w0-w0_new)<=0.00000001 :
            break
        # tol=max(0.9*(norm_two(fct(w0_new))/norm_two(fct(w0)))**2,0.9*tol**2)
        # print(tol)
        w0=w0_new
    
    return w0_new


from scipy.optimize import fsolve

def func(x):
    return np.array([x[0] * np.cos(x[1]) - 4, x[1] * x[0] - x[1] - 5])

root = fsolve(func, [1, 1])

def Jacobian(x):
    return np.array([[np.cos(x[1]),-x[0] * np.sin(x[1])],[x[1],x[0]-1]])


result=gmres(np.array([0,0]), func, 0.01, 1e-10)
print("\n***************** Using Newton-GMRES ******************** \n")
print('x=',result)
print('f(x)=',func(result))

print("\n***************** Exact **********************************\n")
print('x=',root)
# array([6.50409711, 0.90841421])
print('f(x)=',func(root))  # func(root) should be almost 0.0.
