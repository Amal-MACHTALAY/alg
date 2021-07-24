#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 19:28:40 2021

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

# """ test """  
# a=np.arange(9) - 4
# print(a)
# print(np.linalg.norm(a,ord=2))
# print(norm_two(a))


def AxB(X,Y):
    XY=np.zeros((len(X),len(Y[0])))
    for i in range(len(X)):
    # iterate through columns of Y
        for j in range(len(Y[0])):
        # iterate through rows of Y
            for k in range(len(Y)):
                XY[i][j] += X[i][k] * Y[k][j]
    return XY


def LU(A):
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

# """ test """  
# m=np.array([[3,4,-1],[2,0,1],[1,3,-2]])
# print(m)
# l,u=LU(m)
# print("l=",l)
# print("u=",u)
# print(AxB(l,u))

def gauss_jordan_inv(A,inv):
    """ Compute the inverse of A """
    n = len(A)

    #we need a matrix with double number of columns
    m = 2*n
    A1 = np.zeros((n,m))
    for i in range(n):
        for j in range(n):
            A1[i,j] = A[i,j]

    # making the other half an identity matrix of order n

    for i in range(n):        
         A1[i][i+n] = np.float32(1.0)

    # the gauss jordan elimination
    for i in range(n):
        if A1[i][i] == 0.0:
            print('Divide by zero detected!')
            return 
        for j in range(n):
            if i != j:
                ratio = A1[j][i]/A1[i][i]
                for k in range(2*n):
                    A1[j][k] = A1[j][k] - ratio * A1[i][k]

    # operating on rows to make them as identitie (1 in the diag)
    for i in range(n):
        divisor = A1[i][i]
        for j in range(2*n):
            A1[i][j] = A1[i][j]/divisor

    # Our matrix inverse
    for i in range(n):       
        for j in range(n):
            inv[i][j]=A1[i][j+n]

# """ test """  
# m=np.array([[3,4,-1],[2,0,1],[1,3,-2]])
# print(m)
# m_inv=np.zeros((len(m),len(m)))
# #print(m_inv)
# gauss_jordan_inv(m,m_inv)
# print(m_inv)