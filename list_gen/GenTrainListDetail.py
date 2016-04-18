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
Nlistkey=[]
for key in  Nlist:
  Nlistkey.append(key)
  
#%%
from collections import Counter

Ct=Counter(Nlistkey)

#%%

fCt=open('Ct.txt','w')
NlistCt={}
for key in Nlist:
  NlistCt[key]=len(Nlist[key])
#%%  
th=0
fCt=open('Ct.txt','w')
for key in NlistCt:
  fCt.write(str(key)+':'+str(NlistCt[key])+'\n') 
  if NlistCt[key]>200:
    th=th+1
print th    


#%%
label=4740
fCt=open('fimgtrain1to9.txt','w')

for key in Nlist:
  if len(Nlist[key])>0 and len(Nlist[key])<10:
    for imgid in Nlist[key]:
      fCt.write(imgid+'.jpg '+str(label)+'\n')
    label=label+1
fCt.close()

#%%
label=0
fCt=open('fimgtrain100f2k.txt','w')
threshod=2000
for key in Nlist:
  if len(Nlist[key])>99:
    for i in range(min(threshod,len(Nlist[key]))):
      fCt.write(Nlist[key][i]+'.jpg '+str(label)+'\n')
    label=label+1
fCt.close()

#%%
label=4740
fCt=open('fimgtrain10000key.txt','w')

for key in Nlist:
  if len(Nlist[key])>10000:
    for imgid in Nlist[key]:
      fCt.write(imgid+'.jpg '+str(key)+'_'+str(len(Nlist[key]))+'_'+str(label)+'\n')
    label=label+1
fCt.close()

    
  

      





