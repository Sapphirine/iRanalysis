#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 05:01:54 2016

@author: siddharthvarshney
"""

import csv
with open("loan.csv","rb") as source:
    rdr = csv.reader(source)
    with open("pre_processed_loan_data.csv","wb") as result:
        wtr = csv.writer( result )
        for r in rdr:
            wtr.writerow( (r[0], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9],r[12],r[13],r[14],r[16],r[17],r[23],r[24],r[25],r[26],r[27],r[30],r[31],r[32],r[33],r[34],r[35],r[36],r[37],r[38],r[39],r[40],r[41],r[42],r[43],r[44],r[45],r[46],r[48],r[56]) )