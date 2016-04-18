# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 11:30:27 2015

@author: Henry
"""
##

from GenSimList import Load
from GenSimList import pop_random2

imgInfotxt=Load('../finaldata/imgInfo.txt')

#%%

total=len(imgInfotxt)
print imgInfotxt[1][1]

#%%
print imgInfotxt[1]
print imgInfotxt[1116]
name=[]
catg=[]
attb=[]
for i in range(0,total):
  name.append(imgInfotxt[i][0])
  catg.append(imgInfotxt[i][1])
  attb.append(imgInfotxt[i][2])

#%%
print catg[0]
print catg.count(33)

catg.sort()

type(catg[0])
#%%
from collections import Counter

catgcU=Counter(catg)
subcatCouter=Counter(attb)
print catgcU
print subcatCouter

#%%
attbCount=[]
for key in subcatCouter:
  print subcatCouter[key]
  attbCount.append(subcatCouter[key])
#%%
import numpy as np
print np.median(attbCount)
  
print np.mean(attbCount)

#%%
subcatMap={}
for key in subcatCouter:
  subcatMap[key]=[]

for img in imgInfotxt:
  subcatMap[img[2]].append(img[0])
  

#%%
imgtrain={}
imgtest={}

for key in subcatMap:
  a=subcatMap[key]
  imgtrain[key]=[]
  imgtest[key]=[]
  if len(a)>9:
    imgtest[key].append(pop_random2(a))
    imgtest[key].append(pop_random2(a))
  imgtrain[key]=a

#%%
label=0
fimgtrain = open('fimgtrainc10b.txt','w')
fimgtest = open('fimgtestcp2b.txt','w')

for key in imgtrain:
  print key,label
  a=imgtrain[key]
  for repet in range(198424/len(a)):
    for ele in a:
      fimgtrain.write('/home/alisc/code/caffe/data/alisc/final_train_image/'+ele+'.jpg '+str(label)+'\n')
  for remain in range(198424-(repet+1)*len(a)):
    fimgtrain.write('/home/alisc/code/caffe/data/alisc/final_train_image/'+pop_random2(a)+'.jpg '+str(label)+'\n')
  for imgidt in imgtest[key]:
    fimgtest.write('/home/alisc/code/caffe/data/alisc/final_train_image/'+imgidt+'.jpg '+str(label)+'\n')    
  label=label+1 # python will convert \n to os.linesep





