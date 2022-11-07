# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 22:02:59 2022

@author: Administrator
"""

import pandas as pd
import jieba
import collections
from zhon.hanzi import punctuation
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import math

# when given the name of a text file excluding the extension, this function reads the text and returns a string
def readText(filename):
    text = ""
    try:
        f = open(filename, "r", encoding='utf-8')
        text = f.read()
        f.close()
    
    except UnicodeDecodeError:
        f = open(filename, "r", encoding='utf-16')
        text = f.read()
        f.close()
    return text



if __name__=='__main__':
    
    #texts = ["太阳照在桑干河上.txt"]
    texts = ["祝福.txt"]
    for title in texts:
        sentence=readText(title)
        
        for i in punctuation:
            sentence=sentence.replace(i, "")   #清除中文标点符号
        sentence=sentence.replace('\n', "")   #清除换行符
        
        sentence_cut=jieba.lcut(sentence)      #分词
        
        
        
        a=500  #窗口的长度 
        thr=int(0.4*math.sqrt(len(sentence_cut)/a))  #阈值
        Lcounter = collections.Counter(sentence_cut) #根据阈值thr截断的tN，tM，tR矩阵
        words,_ = zip(*Lcounter.items())
        _,words_number = zip(*Lcounter.items())
        for i in range(len(Lcounter)):
            if words_number[i]<thr:
              del Lcounter[words[i]]
        twords,_ = zip(*Lcounter.items())   #按照阈值删除之后的词元祖
        _,twords_number = zip(*Lcounter.items())
        
        #tM矩阵
        tM=np.zeros((len(twords),len(words)))
        for i in range(len(twords)):
            for j in range(len(sentence_cut)):
                d=int(a/2)  #词i和词j之间的距离
                if twords[i]==sentence_cut[j]:
                    if(j-d>0 and j+d<len(sentence_cut)):
                        for m in range(d):
                            n=words.index(sentence_cut[j-m])  # 索引-a/2范围内的词在词列表中的位置
                            tM[i][n]=tM[i][n]+1
                            n=words.index(sentence_cut[j+m])  # 索引+a/2范围内的词在词列表中的位置
                            tM[i][n]=tM[i][n]+1
                    if(j-d<=0):
                        for m in range(d):
                            n=words.index(sentence_cut[j+m])  # 索引+a/2范围内的词在词列表中的位置
                            tM[i][n]=tM[i][n]+1
                    if(j+d>=len(sentence_cut)):
                        for m in range(d):
                            n=words.index(sentence_cut[j-m])  # 索引-a/2范围内的词在词列表中的位置
                            tM[i][n]=tM[i][n]+1
        
        
        #tN,tR矩阵  
        tR=np.zeros((len(twords),len(words)))
        tN=np.zeros((len(twords),len(words)))
        for i in range(len(twords)):
            for j in range(len(words)):      
                tR[i][j]=a*twords_number[i]*words_number[j]/len(sentence_cut)
                tN[i][j]=(tM[i][j]-tR[i][j])/math.sqrt(tR[i][j])
        #np.savetxt(r'tN.txt', tN, fmt='%d', delimiter=',') 
        
        #SVD tN矩阵
        U,sigma,VT=la.svd(tN)
        d=20
        Ur=U[:,0:d]
        sigmar=sigma[0:d]
        VTr=VT[0:d,:]
        tNr=Ur @ np.diag(sigmar) @ VTr
        #np.savetxt(r'tNr.txt', tNr, fmt='%f', delimiter=',') 
       
        
        #  v vector   V matrix
        windows_number=len(sentence_cut)-a+1
        V=np.zeros((windows_number,len(words)))
        for i in range(windows_number):
            subsentence_cut=[]
            v=np.zeros(len(words))   #词向量
            sum_m=0
            for j in range(a):
                #subsentence_cut.append(sentence_cut[i+j])   
                #print(i,j,len(sentence_cut))
                if((i+j)<len(sentence_cut)): 
                   # print(i,j)
                    subsentence_cut.append(sentence_cut[i+j])   
                    
            subcounter = collections.Counter(subsentence_cut) 
            subwords,_ = zip(*subcounter.items())
            _,subwords_number = zip(*subcounter.items())
    
            for q in range(len(subwords)):
                if(subwords_number[q]>=thr):
                    a=twords.index(subwords[q])   
                    m=subwords_number[q]  #词在窗口中出现的次数
                    v=v+tNr[a]*m   #词对应的SVD向量
                    sum_m=sum_m+m*m
            V[i]=v/math.sqrt(sum_m)
            
            
        #内积   
        C=[]  
        for i in range(windows_number):
            sequence=0
            c=0
            for j in range(windows_number):
                if(i+j<windows_number):
                    c=c+np.dot(V[j], V[i+j])
                    #print(V[i])
                    sequence=sequence+1
                    #print(c,sequence)
            C.append(c/sequence)
        np.savetxt(r'C.txt',C, fmt='%f', delimiter=',')          
        
            
            
         
            
            
       