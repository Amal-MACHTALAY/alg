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
    n = len(v)
    Av = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
              s += A[i,j]*v[j]
        Av[i] = s
    return Av
    

def gmres(w0, fct, sigma, tol):
    a=0
    while True :
        a+=1
        print("*********************************************************\n")
        print(a)
        # q=(fct(w0+sigma*w0)-fct(w0))/sigma
        # print('q=',q)
        # r=-fct(w0)-q
        r=-fct(w0)
        # print('r=',r)
        v=[]
        v.append(r/np.linalg.norm(r,ord=2))
        M=get_preconditionner(Jacobian(w0))
        H=[]
        j=0
        while True :
            hh=[]
            print('************ j=',j)
            xj=matvec(M, v[j])
            q=(fct(w0+sigma*xj)-fct(w0))/sigma
            vj=q
            # print('q=',q)
            for i in range(j+1):
                # print('---------------i=',i)
                hij=np.dot(q,v[i])
                # print('hij=',hij)
                hh.append(hij)
                # print(vj,hij*v[i])
                vj-=hij*v[i]
            hjpj=np.linalg.norm(vj,ord=2)
            hh.append(hjpj)
            H.append(hh)
            print('H=',H)
            m=j
            # print(hjpj)
            if hjpj==0.0 :
                break
            print('yes1')
            v.append(vj/hjpj)
            print('v=',v)
            print('hh=',hh)
            print(np.linalg.norm(fct(w0)+q,ord=2))
            if np.linalg.norm(fct(w0)+q,ord=2)<=tol:
                break
            print('yes2')
            j+=1
        print('m=',m)
        h=get_Hessenberg_matrix(H,m)
        print('h=',h)
        # calcul of beta*e1
        beta=np.zeros(m)
        beta[0]=np.linalg.norm(r,ord=2)
        print('beta=',beta)
        # Minimize for y
        print('v=',v)
        y=np.linalg.lstsq(h.transpose(),beta,rcond=-1)[0]
        print('y=',y)
        # if len(y)>len(v):
        #     y=y[:-1]
        w0_new=w0+np.dot(np.asarray(v).transpose(),y)
        print('w0_new=',w0_new)
        print('w0=',w0)
        print('norm=',np.linalg.norm(fct(w0),ord=2))
        if np.linalg.norm(fct(w0),ord=2)<=tol:
            break
        tol=max(0.9*(np.linalg.norm(fct(w0_new),ord=2)/np.linalg.norm(fct(w0),ord=2))**2,0.9*tol**2)
        print(tol)
        w0=w0_new
        
    # Compute residual
    # res=np.linalg.norm(beta-np.dot(h,y),ord=2)
    # print('residual=',res)
    
    
    return w0_new,m


from scipy.optimize import fsolve

def func(x):

    return np.array([x[0] * np.cos(x[1]) - 4, x[1] * x[0] - x[1] - 5])

root = fsolve(func, [1, 1])

def Jacobian(x):
    return np.array([[np.cos(x[1]),-x[0] * np.sin(x[1])],[x[1],x[0]-1]])


# a=np.array([1, 1, 1])
# b=np.array([5, 6, 7])
# print(a)
# print(a+b)
# print(func(a))
# print(np.zeros(3))
result,k=gmres(np.array([0,0]), func, 0.1, 1e-16)
print("\n***************** GMRES ******************** \n")
print('k=',k)
print(result)
print(func(result))

print("\n************** Exact *********************\n")
print(root)
# array([6.50409711, 0.90841421])

print(func(root))  # func(root) should be almost 0.0.
# array([ True,  True])

# print("\n**************************************\n")
# w0=np.array([1,1])
# F=func(w0)
# print(F)
# q=(func(w0+0.1*w0)-func(w0))/0.1
# print(q)
# r=-F-q
# print(r)
# b=np.linalg.norm(r,ord=2)
# print(b)
# v0=r/b
# print(v0)
