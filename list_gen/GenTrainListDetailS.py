# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 11:30:27 2015

@author: Henry
"""
##

from GenSimList import Load

imgInfotxt=Load('../finaldata/imgInfo.txt')


#%%
total=len(imgInfotxt)
Nlist={}
for i in range(total):
  key=imgInfotxt[i][2]
  if len(imgInfotxt[i][3])>0:
    s=imgInfotxt[i][3].split(";")
    s.sort()
    for j in range(len(s)):
      ss=s[j].split(":")
      key=key+ss[0]+ss[1]
  if key in Nlist:
    Nlist[key].append(imgInfotxt[i][0])
  else:
    Nlist[key]=[]
    Nlist[key].append(imgInfotxt[i][0])
#%%
cout=0
Nlistkey=[]
for key in  Nlist:
  Nlistkey.append([key,len(Nlist[key])])
  if len(Nlist[key])>1000:
    print key,len(Nlist[key])
    cout=cout+1
print cout
  
#%%
from collections import Counter

Ct=Counter(Nlistkey)

#%%
label=0
fCt=open('fimgtrain100T.txt','w')

for key in Nlist:
  if len(Nlist[key])>100 :
    for imgid in Nlist[key]:
      fCt.write(imgid+'.jpg '+str(label)+'\n')
    label=label+1
fCt.close()

    
  

      





