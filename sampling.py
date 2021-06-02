#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:49:44 2021

@author: amal
"""

import numpy as np

""" random samples from 1d distribution """

def sample_1d(pdf, interval, nbr_samples, n_max=10**6):
    np.random.seed(45)
    """ genearte a list of random samples from a given pdf suggests random samples in 'interval' 
    and accepts-rejects the suggestion with probability 0<pdf(x)<=M.
    We generate two numbers x,y from auniform distribution.  
    If pdf(x)≤y then we keep x as a sample, otherwise we reject it.
    In regions where the pdf is high, we are less likely to reject an x, 
    and so we will get more values in that region. 
    """
    samples=[]
    M=1
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

pos0=sample_1d(rho_int,[0,L],20)
print(pos0)
