# -*- coding: utf-8 -*-
"""
Created on Thu Oct  13 22:03:43 2015
Generate balanced Simlist


@author: Zengming Shen
"""

print "this is to generate image similarity list:" 
print "uages: MAPMain('valid_image_txt','valid_vgg.txt')"     
import csv,random
#%%

def Load(QueryFile):
  train=[]
  with open(QueryFile, 'r') as f:
      reader_true = csv.reader(f, quoting=csv.QUOTE_NONE)
      for row_true in reader_true:
        train.append(row_true)
  return train

def pop_random(lst):
  idx = random.randrange(0, len(lst))
  return lst.pop(idx)
 #%% 
def commutity(lst1,lst2):
  return (lst1[0]==lst2[1] and lst1[1]==lst2[0]) or lst1==lst2
lst1=[4,5,0]  
lst2=[5,4,0]
lst3=lst1
lst4=[4,4,0]
print commutity(lst1,lst2)
print commutity(lst1,lst3)
print commutity(lst1,lst4)
#%%
def commute(lst):
  temp=lst[0]
  lst[0]=lst[1]
  lst[1]=temp
  return lst
  return (lst1[0]==lst2[1] and lst1[1]==lst2[0]) or lst1==lst2
lst1=[4,5,0]  
lst2=[5,4,0]
lst3=lst1
lst4=[4,4,0]
print commute(lst1)
print commute(lst3)
print commute(lst4)

#%%
def compMetric1(rand1,rand2):
  if rand1[2] == rand2[2]:
    return 1
  else:
    return 0
  
def GenList(In,N):
  rand1 = pop_random(In,N)
  rand2 = pop_random(In,N)
  return [rand1[0],rand2[0],compMetric1(rand1,rand2)]

def GenListsFile(In,OutN):
  pos=0
  N=len(In)
  writer=csv.writer(open('ImgSimList.txt','wb'))
  while OutN > 0:
    data=GenList(In,N)
    if data[2] == 1:
      pos=pos+1
    writer.writerow(data)
    OutN=OutN-1
  writer.writerow(pos)
  return pos

def GenListsFileEven(In,OutN):
  N=len(In)
  Pos=OutN*0.5
  Neg=OutN-Pos
  writer=csv.writer(open('ImgSimListEvenTest.txt','wb'))
  while OutN > 0:
    data=GenList(In,N)
    if data[2] == 1:
      Pos=Pos-1
      writer.writerow(data)
      OutN=OutN-1
    elif Neg>0:
      writer.writerow(data)
      Neg=Neg-1
      OutN=OutN-1
      










  