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
import matplotlib.pyplot as plt
attbint = [int(i) for i in attb]
plt.hist(attbint,bins=559)
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
imgtrainb={}


for key in subcatMap:
  imgtrainb[key]=[]
  a=subcatMap[key]
  if len(a)>3230:
    for i in range(3230):
      imgtrainb[key].append(pop_random2(a))
  else:
    for repet in range(3230/len(a)):
      for ele in a:
        imgtrainb[key].append(ele)
    for remain in range(3230-(repet+1)*len(a)):
      imgtrainb[key].append(pop_random2(a))
#%%
imgtest={}

for key in imgtrainb:
  imgtest[key]=[]
  for i in range(30):
    imgtest[key].append(pop_random2(imgtrainb[key]))
  
#%%
label=0
fimgtrain = open('fimgtrainb3230.txt','w')
fimgtest = open('fimgtestb30.txt','w')

for key in imgtrainb:
  for imgid in imgtrainb[key]:
    fimgtrain.write(imgid+'.jpg '+str(label)+'\n')
  for imgidt in imgtest[key]:
    fimgtest.write(imgidt+'.jpg '+str(label)+'\n')    
  label=label+1 # python will convert \n to os.linesep
fimgtrain.close()
fimgtest.close()
#%%
label=0
fimgtest = open('fimgtestb30.txt','w')
for key in imgtest:
  for imgid in imgtest[key]:
    fimgtest.write(imgid+'.jpg '+str(label)+'\n')
  label=label+1# python will convert \n to os.linesep
fimgtest.close()
#%%
import random
label=0
fimgtest = open('imgtestcb.txt','w')
for key in imgtest:
  lenimgkey=len(imgtest[key])+0.0
  mylist=imgtest[key]
  if lenimgkey>8000:
    rand_smpl = [ mylist[i] for i in sorted(random.sample(xrange(int(lenimgkey)), int(lenimgkey/4))) ]
  elif lenimgkey>3000:#1378*0.15:
    rand_smpl = [ mylist[i] for i in sorted(random.sample(xrange(int(lenimgkey)), int(lenimgkey/3.75))) ]
  elif lenimgkey>206.7:
    rand_smpl = [ mylist[i] for i in sorted(random.sample(xrange(int(lenimgkey)), int(lenimgkey/2.5))) ]
  elif lenimgkey>13.6:
    rand_smpl = [ mylist[i] for i in sorted(random.sample(xrange(int(lenimgkey)), int(lenimgkey/2))) ]
  else:
    rand_smpl=mylist
    
  for imgid in rand_smpl:
    fimgtest.write(imgid+'.jpg '+str(label)+'\n')
  label=label+1# python will convert \n to os.linesep
fimgtest.close()

#%%
from sklearn.cross_validation import train_test_split
import numpy as np
a=subcatMap['150506']
print a 

atrain,atest=train_test_split(a,test_size=0.1, random_state=42)
print atrain,atest

#%%
print type(catgcU)
print catgcU[1]

#%%
print catg.index(33)
indices33 = [i for i, x in enumerate(catg) if x ==73]
attb33=[]
for i in range(0,len(indices33)):
  attb33.append(attb[indices33[i]])
  
print Counter(attb33)


#%%
import matplotlib.pyplot as plt
from numpy.random import normal
gaussian_numbers = normal(size=1000)
plt.hist(gaussian_numbers)
plt.title("Gaussian Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()




