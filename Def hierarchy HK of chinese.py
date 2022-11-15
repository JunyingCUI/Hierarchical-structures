# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 09:30:23 2022

@author: Administrator
"""

import jieba.posseg as psg
import pandas as pd
import jieba
import collections
from zhon.hanzi import punctuation
from sklearn import preprocessing
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import math
import tqdm
import random
from matplotlib import font_manager   #中文字体

fontP = font_manager.FontProperties()
fontP.set_family('SimHei')
fontP.set_size(14)

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

def cleaningText(a,thr_factor,sentence,stop_sentence):
    for i in punctuation:
         sentence=sentence.replace(i, "")   #清除中文标点符号
    sentence=sentence.replace('"', "")
    sentence=sentence.replace('\n', "")   #清除换行符
    sentence_have = psg.cut(sentence,HMM=True)
    sentence_cut = [x.word for x in sentence_have if True]      
    
    #a=200  #窗口的长度     
    thr=int(thr_factor*math.sqrt(len(sentence_cut)/a))  #阈值
    Lcounter = collections.Counter(sentence_cut) #根据阈值thr截断的tN，tM，tR矩阵
    words,words_number= zip(*Lcounter.items())
    for i in range(len(Lcounter)):
        if words_number[i]<thr:
          del Lcounter[words[i]]
    twords,twords_number = zip(*Lcounter.items())   #按照阈值删除之后的词元祖

    for i in range(len(Lcounter)):
        #如果小于阈值，就将其删除
        if twords[i] in stop_sentence:
          del Lcounter[twords[i]]
    swords,swords_number = zip(*Lcounter.items())   ##删除停止词
    print(np.sum(swords_number)/len(sentence_cut),thr,len(swords),len(sentence_cut))
    print(len(sentence_cut))
    return sentence_cut,words,twords,twords_number,words_number,swords

def N_matrix(a,sentence_cut,words,twords,twords_number,words_number):
    #print(Lcounter)
   
    #tM矩阵 
    tM=np.zeros((len(twords),len(words)))
    for i in range(len(twords)):
        for j in range(len(sentence_cut)):
            distance=int(a/2)  #词i和词j之间的距离
            if twords[i]==sentence_cut[j]:
                if(j-distance>0 and j+distance<len(sentence_cut)):
                    for m in range(distance):
                        n=words.index(sentence_cut[j-m])  # 索引-a/2范围内的词在词列表中的位置
                        tM[i][n]=tM[i][n]+1
                        n=words.index(sentence_cut[j+m])  # 索引+a/2范围内的词在词列表中的位置
                        tM[i][n]=tM[i][n]+1
                if(j-distance<=0):
                    for m in range(distance):
                        n=words.index(sentence_cut[j+m])  # 索引+a/2范围内的词在词列表中的位置
                        tM[i][n]=tM[i][n]+1
                if(j+distance>=len(sentence_cut)):
                    for m in range(distance):
                        n=words.index(sentence_cut[j-m])  # 索引-a/2范围内的词在词列表中的位置
                        tM[i][n]=tM[i][n]+1

    #tN,tR矩阵  
    tR=np.zeros((len(twords),len(words)))
    tN=np.zeros((len(twords),len(words)))
    #print((len(twords),len(words)))
    for i in range(len(twords)):
        for j in range(len(words)):      
            tR[i][j]=a*twords_number[i]*words_number[j]/len(sentence_cut)
            tN[i][j]=(tM[i][j]-tR[i][j])/math.sqrt(tR[i][j])
    return tN


def svd(dimension,tN):
    U,sigma,VT=la.svd(tN)
    Ur=U[:,0:dimension]
    sigmar=sigma[0:dimension]
    VTr=VT[0:dimension,:]
    return VTr


def new_tN(dimension,tN,VTr):
    tNr = np.zeros(shape=(tN.shape[0],dimension))
    for i in range(tN.shape[0]):
        for j in range(dimension):
            #tNr[i,j] = np.sum(E[w2id[twords[i]]]*VTr[j])#/np.sqrt(np.sum(tN[i]*tN[i]))
            tNr[i,j] = np.sum(tN[i]*VTr[j])#/np.sqrt(np.sum(tN[i]*tN[i]))
    return tNr


def origin_V_vectors(a,sentence_cut,dimension,swords,tNr):
    null = 0
    #  v vector   V matrix
    windows_number=len(sentence_cut)-a-1
    V=np.zeros((windows_number,dimension))
    for i in range(windows_number):
        subsentence_cut=[]
        v=np.zeros(dimension)   #词向量
        sum_m=0
        for j in range(a):
            if((i+j)<len(sentence_cut)): 
                subsentence_cut.append(sentence_cut[i+j])   
        
        subcounter = collections.Counter(subsentence_cut)
        subwords,subwords_number = zip(*subcounter.items())

        for q in range(len(subwords)):
            if subwords[q] in swords:
                pos=swords.index(subwords[q])   
                m=subwords_number[q]  #词在窗口中出现的次数
                v=v+tNr[pos]*m   #词对应的SVD向量
                sum_m=sum_m+m*m
        if sum_m==0:
            V[i]=0; null+=1
        else:
            V[i]=v/math.sqrt(sum_m)
    return V,windows_number

def random_V_vectors(a,sentence_cut,dimension,swords,tNr):
    null = 0
    #  random v vector   V matrix
    sequence=[]
    for i in range(len(sentence_cut)):
        sequence.append(i)
    random.shuffle(sequence)
    random_sentence_cut=[]
    for i in range(len(sentence_cut)):
        random_sentence_cut.append(sentence_cut[sequence[i]])   #原文本保持词频打乱顺序重新排列
    #print(random_sentence_cut)
    windows_number=len(random_sentence_cut)-a-1
    V=np.zeros((windows_number,dimension))
    for i in range(windows_number):
        subsentence_cut=[]
        v=np.zeros(dimension)   #词向量
        sum_m=0
        for j in range(a):
            if((i+j)<len(random_sentence_cut)): 
                subsentence_cut.append(random_sentence_cut[i+j])   
        
        subcounter = collections.Counter(subsentence_cut)
        subwords,subwords_number = zip(*subcounter.items())

        for q in range(len(subwords)):
            if subwords[q] in swords:
                pos=swords.index(subwords[q])   
                m=subwords_number[q]  #词在窗口中出现的次数
                v=v+tNr[pos]*m   #词对应的SVD向量
                sum_m=sum_m+m*m
        if sum_m==0:
            V[i]=0; null+=1
        else:
            V[i]=v/math.sqrt(sum_m)
    return V,windows_number
            
def hierarchy_V_vectors(K,H,J,E,l,a,sentence_cut,dimension,swords,tNr):
    null = 0
    #  random v vector   V matrix

# =============================================================================
#     K=4  #Level
#     H=3  #subdivision
#     J=1
#     E=5
# =============================================================================
    now_pro=[J for i in range(pow(H,K))]  
    for i in range(K):
        if(i==0):
            up_pro=[J for i in range(pow(H,i+1))]  #记录上一层级概率
        else:
            up_pro=[J for i in range(pow(H,i))]
            for j in range(pow(H,i)):
                up_pro[j]=now_pro[j]
        station=random.randrange(0,pow(H,i+1))
        for j in range(pow(H,i+1)):
            if(j!=station):   
                now_pro[j]=up_pro[j//H]                 
            else:
                now_pro[j]=(up_pro[j//H])*E  
    nomalize_pro=[]  #Hierarchy and the origin of Scaling,K levels and each level contains the same number of parts of H., to normalize the probabilities to 1
    for i in range(len(now_pro)):  #P normalize
        nomalize_pro.append(now_pro[i]/sum(now_pro))

    #l=82
    Hier_sentence_cut=[]
    for i in range(pow(H,K)*l):
        Hier_sentence_cut.append(sentence_cut[i])
    #print(Hier_sentence_cut)
        
    text_index=np.zeros((pow(H,K),l))
    for i in range(pow(H,K)):
        for j in range(l):
            text_index[i][j]=j
    text_words=['a' for i in range(pow(H,K)*l)]  


    ran_down=0
    ran_up=1        
    for i in range(len(Hier_sentence_cut)):
        m=0.0
        r=random.uniform(ran_down, ran_up)
        subdivision=0
        text_sequence=[]
        for j in range(len(nomalize_pro)):
            if(nomalize_pro[j]!=0):
                m=m+nomalize_pro[j]
                if(r<=m):
                    subdivision=j    #轮盘赌算法选第几个subdivision
        
        if(nomalize_pro[subdivision]!=0):
            for q in range(len(text_index[subdivision])):
                if(text_index[subdivision][q]!=2*l):
                    text_sequence.append(text_index[subdivision][q])
            if(len(text_sequence)==0):
                ran_up=ran_up-nomalize_pro[subdivision]
                nomalize_pro[subdivision]=0.0 
                r=random.uniform(ran_down, ran_up)
                subdivision=0
                text_sequence=[]
                for j in range(len(nomalize_pro)):
                    if(nomalize_pro[j]!=0):
                        m=m+nomalize_pro[j]
                        if(r<=m):
                            subdivision=j    #轮盘赌算法选第几个subdivision
                if(nomalize_pro[subdivision]!=0):
                    for q in range(len(text_index[subdivision])):
                        if(text_index[subdivision][q]!=2*l):
                            text_sequence.append(text_index[subdivision][q])
                part=random.randint(0, len(text_sequence)-1)  #产生一个选位置的随机数
                b=int(text_sequence[part])
                text_index[subdivision][b]=2*l
                text_words[subdivision*l+b]=Hier_sentence_cut[i]
            else:
                part=random.randint(0, len(text_sequence)-1)  #产生一个选位置的随机数
                b=int(text_sequence[part])
                text_index[subdivision][b]=2*l
                text_words[subdivision*l+b]=Hier_sentence_cut[i]

    hierarchy_sentence_cut=[]
    for i in range(len(text_words)):
        hierarchy_sentence_cut.append(text_words[i])     #hierchchy 文本
    #print(random_sentence_cut)
    windows_number=len(hierarchy_sentence_cut)-a-1
    V=np.zeros((windows_number,dimension))
    for i in range(windows_number):
        subsentence_cut=[]
        v=np.zeros(dimension)   #词向量
        sum_m=0
        for j in range(a):
            if((i+j)<len(hierarchy_sentence_cut)): 
                subsentence_cut.append(hierarchy_sentence_cut[i+j])   
        
        subcounter = collections.Counter(subsentence_cut)
        subwords,subwords_number = zip(*subcounter.items())

        for q in range(len(subwords)):
            if subwords[q] in swords:
                pos=swords.index(subwords[q])   
                m=subwords_number[q]  #词在窗口中出现的次数
                v=v+tNr[pos]*m   #词对应的SVD向量
                sum_m=sum_m+m*m
        if sum_m==0:
            V[i]=0; null+=1
        else:
            V[i]=v/math.sqrt(sum_m)
    return V,windows_number

def correction_function(V,windows_number):
    #内积   
    C=[]
    size = []
    V=preprocessing.normalize(V)
    for i in tqdm.tqdm(range(windows_number)):
        sequence=0
        c=0
        for j in range(windows_number):
            if(i+j<windows_number):
                c=c+np.dot(V[j], V[i+j])
                sequence=sequence+1
        C.append(c/sequence)
        size.append(sequence)
        
    return C
    #plt.loglog(range(1,len(C)+1),C)
    #np.savetxt(r'C.txt',C, fmt='%f', delimiter=',')   

if __name__ == '__main__':
    
    #texts = ["祝福.txt"]  #length=4944  祝福(K=3,H=3,l=190)
    #texts = ["孔乙己传.txt"]  #length=3677 (K=3,H=3,l=135)
    texts = ["古典爱情.txt"]   #length=11376 (K=4,H=3,l=140)
    #texts = ["命若琴弦.txt"]  #length=6454,(K=3,H=4,l=100)
    #texts = ["酒徒刘以鬯.txt"]
    stop_sentence=readText("all_stopwords.txt").replace('\n','')  #停止词词库
    a=200  #窗口大小
    thr_factor=0.6 #
    dimension=20  #dimension
    K=4  #Level  
    H=3 #subdivision
    J=1
    E=5
    l=140
    
    data = []
    for title in texts:
        sentence=readText(title)
        sentence_cut,words,twords,twords_number,words_number,swords=cleaningText(a, thr_factor,sentence, stop_sentence)
        tN=N_matrix(a, sentence_cut, words, twords, twords_number, words_number)
        VTr=svd(dimension,tN)
        tNr=new_tN(dimension, tN, VTr)
        origin_V,origin_windows_number=origin_V_vectors(a, sentence_cut, dimension, swords, tNr)
        random_V,random_windows_number=random_V_vectors(a, sentence_cut, dimension, swords, tNr)
        hierarchy_V,hierarchy_windows_number=hierarchy_V_vectors(K, H, J, E, l,a, sentence_cut, dimension, swords, tNr)
        origin_correction=correction_function(origin_V, origin_windows_number)
        random_correction=correction_function(random_V, random_windows_number)
        hierarchy_correction=correction_function(hierarchy_V, hierarchy_windows_number)
        
        
        titleString = title.split(".")[0] + " " 
        plt.title(titleString, fontproperties=fontP)
        plt.xlabel("Time(in word distance units)", fontproperties=fontP)
        plt.ylabel("AutoCorrection", fontproperties=fontP)
        plt.loglog(range(1,len(origin_correction)+1),origin_correction,linewidth =2.0, color='green',label='origin')
        plt.loglog(range(1,len(random_correction)+1),random_correction,linewidth =1.5, color='red', label='random')
        plt.loglog(range(1,len(hierarchy_correction)+1),hierarchy_correction,linewidth =1, color='blue', label='hierarchy')
        plt.legend()
        plt.show()
        
        
        
        
        
        
        
        



    
