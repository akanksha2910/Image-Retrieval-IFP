# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 22:03:43 2015

@author: Zengming Shen
"""

print "this is MAP@ 20 V2 metric: uages: MAPMain('valid_image_txt','valid_vgg.txt')"     
import csv
#%%
def AP2(GTRtn,TestRtn):
  MAPN=20 
  APvalue=0.0
  MatchNum=0.0;
  GTNum=len(GTRtn)
  if GTNum<MAPN:
    MAPN=GTNum
  for i in range(0,20):
    if TestRtn[i] in GTRtn:
      MatchNum=MatchNum+1
      APvalue=APvalue+MatchNum/(i+1.0)
  return APvalue/MAPN
  

 #%%
def MAP(GT,Test):
  MAPvalue=0.0
  for key in GT:
    MAPvalue=MAPvalue+AP2(GT[key],Test[key])
  return MAPvalue/len(GT)

def Load(QueryFile):
  GT={}
  Rtn=[]
  with open(QueryFile, 'rb') as f:
      reader_true = csv.reader(f,delimiter=';', quoting=csv.QUOTE_NONE)
      for row_true in reader_true:
        temp=row_true[0].split(",")
        row_true.remove(row_true[0])
        Rtn=row_true
        Rtn.insert(0,temp[1])
        GT[temp[0]]=Rtn
  return GT

def MAPMain2(GTFile,TestFile):
  GT=Load(GTFile)
  Test=Load(TestFile)
  return MAP(GT,Test)











