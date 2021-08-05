#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 18:16:35 2021

@author: amal
"""

import numpy as np

def norm_two(v):
    n=len(v)
    norm=v[0]**2
    for i in range(1,n):
        norm+=v[i]**2
    norm=np.sqrt(norm)
    return norm

''' test '''   
a=np.arange(9) - 4
print(a)
print(np.linalg.norm(a,ord=2))
print(norm_two(a))


# def linear_least_squares(a,b):
# """ Computes the vector x that approximatively solves the equation ax = b. 
#     x minimizes the Euclidean 2-norm ||b-ax|| 
#     """
    


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
    """ Compute the (complete) LU Factorization """
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

# # the *incomplete LU* decomposition
# import scipy.sparse.linalg as spla
# M_ilu = spla.spilu(M) # Here, M_ilu is a sparse matrix in a CSC format
# # and M will now be the linear operator 
# M = spla.LinearOperator(M.shape, M_iLU.solve)


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

# np.linalg.inv(M)


def inv_matrix_LU_deco(M):
    L,U=LU_decomposition(M)
    inv_L,inv_U=LU_inverse(L,U)
    inv_M=AxB(inv_U,inv_L)
    return inv_M
 

m=np.array([[3,4,-1],[2,0,1],[1,3,-2]])
print(m)
l,u=LU_decomposition(m)
print("l=",l)
print("u=",u)
print(AxB(l,u))
inv_l,inv_u=LU_inverse(l,u)
print('U s inverse :\n',inv_u)
print('L s inverse :\n',inv_l)
print('\n',AxB(l,u))
# print(AxB(l,inv_l))
# print(AxB(inv_u,u))
inv_m=inv_matrix_LU_deco(m)
print('m s inverse :\n',inv_m)
print('Test :\n',AxB(m,inv_m))

def matvec(A,v):
    n = len(v)
    Av = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
              s += A[i,j]*v[j]
        Av[i] = s
    return Av

vec=np.array([1,1,1])
print(matvec(m, vec))
