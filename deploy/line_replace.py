# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 21:03:58 2015

@author: Henry
"""


for i in range(1,64):
  f = open('eval_0.prototxt')
  fout =open('eval_'+str(i)+'.prototxt','wt')
  for line in f:
    if 'data/alisc/fimgeva_lmdb/fimgeva_0' in line:
      print i, line
      fout.write(line.replace('data/alisc/fimgeva_lmdb/fimgeva_0', 'data/alisc/fimgeva_lmdb/fimgeva_'+str(i)))
      print i,line
    else:
      fout.write(line)
  fout.close()
  f.close()
