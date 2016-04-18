# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 11:30:27 2015

@author: Henry
"""
##

import sys
from GenSimList import Load
from GenSimList import GenListsFile

imgInfotxt=Load('../finaldata/list_final_eva_image.txt')

#%%

total=len(imgInfotxt)
print total
print imgInfotxt[0]

#%%
label=0
chunk=50000
idx=0

for chi in range(total/chunk):
  fimgtrain = open('../finaldata/fimgeva_'+str(chi)+'.txt','w')
  for idx in range(chi*chunk,(chi+1)*chunk):
      fimgtrain.write(imgInfotxt[idx][0]+' '+str(label)+'\n')
  fimgtrain.close()
#%%


fimgtrain = open('../finaldata/fimgeva_'+str(chi+1)+'.txt','w')
for idx in range((chi+1)*chunk,total):
    fimgtrain.write(imgInfotxt[idx][0]+' '+str(label)+'\n')
fimgtrain.close()

