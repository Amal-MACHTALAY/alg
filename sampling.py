#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:49:44 2021

@author: amal
"""

import numpy as np
import matplotlib.pyplot as plt

""" random samples from 1d distribution """

def sample_1d(pdf, interval, nbr_samples, M, n_max=10**6):
    np.random.seed(45)
    """ genearte a list of random samples from a given pdf suggests random samples in 'interval' 
    and accepts-rejects the suggestion with probability 0<=pdf(x)<=M.
    We generate two numbers x,y from auniform distribution.  
    If pdf(x)≤y then we keep x as a sample, otherwise we reject it.
    In regions where the pdf is high, we are less likely to reject an x, 
    and so we will get more values in that region. 
    """
    samples=[]
    n=0
    while len(samples)<nbr_samples and n<n_max:
        x=np.random.uniform(low=interval[0],high=interval[1])
        new_sample=pdf(x)
        assert new_sample>=0 and new_sample<=M
        if np.random.uniform(low=0,high=M) <=new_sample:
            samples += [x]
        n+=1
    return sorted(samples)


""" test 1d """
L=1.0
rho_a=0.05; rho_b=0.95; gama=0.1
def rho_int(s): # initial density
    return rho_a+(rho_b-rho_a)*np.exp(-0.5*((s-0.5*L)/gama)**2) # 0<=rho<=rho_jam

pos0=sample_1d(rho_int,[0,L],1,20)
print(pos0)




""" random samples from 2d distribution """
def sample_2d(pdf, xy_min, xy_max, M, nbr_samples, n_max=10**6):
    #np.random.seed(34)
    """ genearte a list of random samples from a given 0<=pdf<=M
    suggests random samples between xy_min and xy_max 
    and accepts-rejects the suggestion with probability pdf(x) 
    """
    samples=np.zeros([nbr_samples,2])
    n=0
    i=0
    while i<nbr_samples and n<n_max:
        z = np.random.uniform(low=xy_min, high=xy_max, size=(1,2))
        z = np.squeeze(z) 
        new_sample = pdf(z)
        assert new_sample>=0 and new_sample<=M
        if np.random.uniform(low=0,high=M) <= new_sample:
            samples[i,0]=z[0]
            samples[i,1]=z[1]
            i+=1
        
        n+=1
    return samples

""" test 2d """
def f(x):
    y=(1/(2*math.pi))*math.exp(-(np.linalg.norm(x)**2)/2) #0<=y<=0.2
    return y

u=sample_2d(f, [-2, -2], [2, 2], 0.2, 100)
# print(u)
plt.scatter(u[:,0],u[:,1])
