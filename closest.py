#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 17:37:14 2021

@author: amal
"""



def closest(lst, val):
    """ Find Closest number of 'val' in a list 'lst' """
    nearest=lst[0]
    for i in range (1,len(lst)):
        if abs(lst[i]-val)<=abs(nearest-val):
            nearest=lst[i]
    return nearest
        
                
        
# def closest(lst, val):
#     return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-val))]

""" test """
import numpy as np
l=np.linspace(0,1,20)
print(l)
print(closest(l, 0.8))

