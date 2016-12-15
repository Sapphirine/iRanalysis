#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 18:39:25 2016

@author: siddharthvarshney
"""

import matplotlib.pyplot as plt
axes = plt.gca()
axes.set_xlim([0,6])
axes.set_ylim([8,25])
plt.plot([0, 1, 2, 3, 4, 5, 6], [24.21406892399628, 23.017082000404983, 20.0050962096867, 17.107564293629007, 14.46677416312843, 11.526595685517064, 8.16106544617723], linestyle='--', marker='o', color='b')
plt.ylabel('Interest Rate')
plt.xlabel('Grade')
plt.show()