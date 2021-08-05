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
 
print('\n***************')
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

print('\n***************')
vec=np.array([1,1,1])
vec2=np.array([1,2,3])
print(matvec(m, vec))
print(scalar(vec,vec2))
print(vecvec(vec,vec2))

def matT(A):
    n=len(A)
    m=len(A[0])
    At=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            At[i,j]=A[j,i]
    return At

print('\n***************')
print('m=',m)
print(matT(m))

def vec_asarray(v):
    vv=np.zeros((len(v),len(v[0])))
    for i in range(len(v)):
        for j in range(len(v[0])):
            vv[i,j]=v[i][j]
    return vv

print('\n***************')
v=[]
v.append(np.array([1,2]))
v.append(np.array([3,4]))
v.append(np.array([5,6]))
print('v=',v)
print(vec_asarray(v))
