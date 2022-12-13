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
import re
import copy
from matplotlib import font_manager   #中文字体

fontP = font_manager.FontProperties()
fontP.set_family('SimHei')
fontP.set_size(14)

#从其他文本随机插入
#最终版,层次结构随机插入其它文本时，除了保证框架，H，K一致，概率也要按照被插入的文本的概率。
#金庸-射雕英雄传缩减版 -V1.txt 删除句子中间的......

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

def cut_paragraph(para):  #每一段分句
    # 相关规则
    pattern = ['([。！？\?])([^”’])','(\.{6})([^”’])','(\…{2})([^”’])','([。！？\?][”’])([^，。！？\?])']
    for i in pattern:
        para = re.sub(i, r"\1\n\2", para)
    para = para.rstrip()
    return para.split("\n")

#每一句分词
def cut_sentence(sentence):
    for i in punctuation:
        sentence=sentence.replace(i, "")   #清除中文标点符号
    sentence=sentence.replace('"', "")
    sentence=sentence.replace('\n', "")   #清除换行符
    words_have = psg.cut(sentence,HMM=True)
    words_cut = [x.word for x in words_have if True] 
    return words_cut 

# 将整篇文章进行分段
def segments(url):
    raw = pd.read_csv(url,names=['txt'], sep='aaa', encoding="utf-8" ,engine='python')

   
    def m_head(tem_str):     #定位段落第一个字
        return tem_str[:1]
    
    def m_mid(tmp_str):   #定位段落是否包含章节名称关键字“回 ”
        return tmp_str.find("回 ")
    
    #接着将函数应用到每一个段落，找到这些定位处的内容
    raw['head'] = raw.txt.apply(m_head)  #每一行的开头，每一段的开头
    raw['mid'] = raw.txt.apply(m_mid)   
    raw['len'] = raw.txt.apply(len)  # 每一行的字长度
    #print(raw['mid'][31])
    #再加判断条件，添加章节编号
    chap_num = 0
    for i in range(len(raw)):
        if raw['head'][i] == "第" and raw['mid'][i] > 0 and raw['len'][i] < 30: #添加章节编号
            #print(i,raw['head'][i],raw['mid'][i],raw['len'][i])
            chap_num += 1
            #print(chap_num)
        raw.loc[i, 'chap'] = chap_num  #添加编号一列
    #添加编号后，可以删除临时变量
    del raw['head']
    del raw['mid']
    del raw['len']
    sentence=[]  #分词
    chap_len=[]  #记录每章的段数
    paragraph_len=[] #记录每段的句长
    words_len=[]  #记录每句的词长
    for i in range(1,chap_num+1):
        tmp_chap = raw[raw['chap'] ==i].copy()
        #print(tmp_chap)
        chap_len.append(len(raw[raw['chap'] == i]))
        tmp_chap.reset_index(drop=True, inplace=True)
        tmp_chap['paraidx'] = tmp_chap.index
        paragraph = tmp_chap['txt'].values.tolist()
        for j in range(1,len(paragraph)):
            paragraph_len.append(len(cut_paragraph(paragraph[j]))) 
            words_sentence=cut_paragraph(paragraph[j])
            for m in range(len(words_sentence)):
                #print(words_sentence[m],cut_sentence(words_sentence[m]))
                words_cut=cut_sentence(words_sentence[m])
                words_len.append(len(words_cut))
                for n in range(len(words_cut)):
                    sentence.append(words_cut[n])
    return chap_len,paragraph_len,words_len,sentence


def cleaningText(a,thr_factor,sentence_cut,stop_sentence):
# =============================================================================
#     for i in punctuation:
#          sentence=sentence.replace(i, "")   #清除中文标点符号
#     sentence=sentence.replace('"', "")
#     sentence=sentence.replace('\n', "")   #清除换行符
#     sentence_have = psg.cut(sentence,HMM=True)
#     sentence_cut = [x.word for x in sentence_have if True]      
# =============================================================================
    
    #a=200  #窗口的长度     
    thr=int(thr_factor*math.sqrt(len(sentence_cut)/a))  #阈值
    Lcounter = collections.Counter(sentence_cut) 
    words,words_number= zip(*Lcounter.items())
    for i in range(len(Lcounter)):
        if words_number[i]<thr:
          del Lcounter[words[i]]
    twords,twords_number = zip(*Lcounter.items())   #按照阈值删除之后的词元祖

    for i in range(len(Lcounter)):
        #如果小于阈值，就将其删除
        if twords[i] in stop_sentence:
          del Lcounter[twords[i]]
    swords,swords_number = zip(*Lcounter.items())   ##按照阈值删除之后再删除停止词 
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
    tR=np.zeros((len(twords),len(words)))        #twords按阈值删除的词
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


def origin_V_vectors(a,sentence_cut,dimension,swords,twords,tNr):
    null = 0
    #  v vector   V matrix
    #np.savetxt(r'祝福-origin.txt', sentence_cut,fmt = '%s',newline=',',delimiter=',')
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
                pos=twords.index(subwords[q])   
                m=subwords_number[q]  #词在窗口中出现的次数
                v=v+tNr[pos]*m   #词对应的SVD向量
                sum_m=sum_m+m*m
        if sum_m==0:
            V[i]=0; null+=1
        else:
            V[i]=v/math.sqrt(sum_m)
    return V,windows_number

def random_V_vectors(a,sentence_cut,dimension,swords,twords,tNr):
    null = 0
    #  random v vector   V matrix
    sequence=[]
    for i in range(len(sentence_cut)):
        sequence.append(i)
    random.shuffle(sequence)
    random_sentence_cut=[]
    for i in range(len(sentence_cut)):
        random_sentence_cut.append(sentence_cut[sequence[i]])   #原文本保持词频打乱顺序重新排列
    #np.savetxt(r'祝福-random.txt', random_sentence_cut,fmt = '%s',newline=',',delimiter=',') 
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
            
def hierarchy_V_vectors(K,H,J,E,l,a,sentence_cut,dimension,swords,twords,tNr):
    null = 0
    #  random v vector   V matrix
    now_pro=[J for i in range(pow(H,K))]  
    for i in range(K):
        if(i==0):
            up_pro=[J for i in range(pow(H,i+1))]  #记录上一层级概率
            station=random.randrange(0,pow(H,i+1))
        else:
            up_pro=[J for i in range(pow(H,i))]
            for j in range(pow(H,i)):
                up_pro[j]=now_pro[j]
            station=random.randrange(station*H,(station+1)*H)
        for j in range(pow(H,i+1)):
            if(j!=station):   
                now_pro[j]=up_pro[j//H]                 
            else:
                now_pro[j]=(up_pro[j//H])*E  
    nomalize_pro=[]  #Hierarchy and the origin of Scaling,K levels and each level contains the same number of parts of H., to normalize the probabilities to 1
    random_nomalize_pro=[]
    for i in range(len(now_pro)):  #P normalize
        nomalize_pro.append(now_pro[i]/sum(now_pro))
        random_nomalize_pro.append(now_pro[i]/sum(now_pro))
   
    #l=82
    Hier_sentence_cut=[]
    print(pow(H,K)*l,len(sentence_cut))
    for i in range(pow(H,K)*l):
        Hier_sentence_cut.append(sentence_cut[i])
        
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
    #np.savetxt(r'祝福-hierarchy.txt', hierarchy_sentence_cut,fmt = '%s',newline=',',delimiter=',') 
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
                pos=twords.index(subwords[q])   
                m=subwords_number[q]  #词在窗口中出现的次数
                v=v+tNr[pos]*m   #词对应的SVD向量
                sum_m=sum_m+m*m
        if sum_m==0:
            V[i]=0; null+=1
        else:
            V[i]=v/math.sqrt(sum_m)
    
    return V,windows_number,random_nomalize_pro

def random_slot_hierarchy_V_vectors(K,H,J,E,l,a,random_nomalize_pro,sentence_cut,dimension,swords,twords,tNr):
    
    random_nomalize_pro = copy.deepcopy(random_nomalize_pro)
    null = 0
    Hier_sentence_cut=[]
    for i in range(pow(H,K)*l):
        Hier_sentence_cut.append(sentence_cut[i])

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
        for j in range(len(random_nomalize_pro)):
            if(random_nomalize_pro[j]!=0):
                m=m+random_nomalize_pro[j]
                if(r<=m):
                    subdivision=j    #轮盘赌算法选第几个subdivision
        
        if(random_nomalize_pro[subdivision]!=0):
            for q in range(len(text_index[subdivision])):
                if(text_index[subdivision][q]!=2*l):
                    text_sequence.append(text_index[subdivision][q])
            if(len(text_sequence)==0):
                ran_up=ran_up-random_nomalize_pro[subdivision]
                random_nomalize_pro[subdivision]=0.0 
                r=random.uniform(ran_down, ran_up)
                subdivision=0
                text_sequence=[]
                for j in range(len(random_nomalize_pro)):
                    if(random_nomalize_pro[j]!=0):
                        m=m+random_nomalize_pro[j]
                        if(r<=m):
                            subdivision=j    #轮盘赌算法选第几个subdivision
                if(random_nomalize_pro[subdivision]!=0):
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
    #np.savetxt(r'祝福-hierarchy.txt', hierarchy_sentence_cut,fmt = '%s',newline=',',delimiter=',') 
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
                pos=twords.index(subwords[q])   
                m=subwords_number[q]  #词在窗口中出现的次数
                v=v+tNr[pos]*m   #词对应的SVD向量
                sum_m=sum_m+m*m
        if sum_m==0:
            V[i]=0; null+=1
        else:
            V[i]=v/math.sqrt(sum_m)
    return V,windows_number

def Real_hierarchy_V_vectors(J,E,l,a,chap_len,paragraph_len,words_len,sentence_cut,dimension,swords,twords,tNr):
    null = 0
    chap_pro=[J for i in range(len(chap_len))]  
    para_pro=[1 for i in range(len(paragraph_len))]
    words_pro=[1 for i in range(len(words_len))] 
    station_chap=random.randrange(0,len(chap_len))  #选择章
    sum_chap_len=0  #
    chose_chap_len=0
    for i in range(len(chap_len)):  
        for j in range(chap_len[i]-1):
            if(station_chap==i):
                para_pro[sum_chap_len+j]=E*chap_pro[i]
            else:
                para_pro[sum_chap_len+j]=chap_pro[i]   
        sum_chap_len=sum_chap_len+(chap_len[i]-1)
    if(station_chap!=0):
        for m in range(station_chap):
            chose_chap_len=chose_chap_len+(chap_len[m]-1)
            station_para=random.randrange(chose_chap_len,chose_chap_len+chap_len[station_chap])  #从被选的章里面选择段
    else:
        station_para=random.randrange(chose_chap_len,chose_chap_len+chap_len[station_chap])  #从被选的章里面选择段
    sum_para_len=0
    for p in range(len(paragraph_len)):  
        for q in range(paragraph_len[p]):
            if(sum_para_len<len(words_len)):
                if(station_para==p):
                    words_pro[sum_para_len+q]=E*para_pro[p]
                else:
                    words_pro[sum_para_len+q]=para_pro[p]
        sum_para_len=sum_para_len+(paragraph_len[p])
    nomalize_pro=[]  #Hierarchy and the origin of Scaling,K levels and each level contains the same number of parts of H., to normalize the probabilities to 1
    Real_random_nomalize_pro=[]
    for i in range(len(words_pro)):  #P normalize
        nomalize_pro.append(words_pro[i]/sum(words_pro))
        Real_random_nomalize_pro.append(words_pro[i]/sum(words_pro))
   
   
    #l=82
    Hier_sentence_cut=[]
    for i in range(sum(words_len)):
        Hier_sentence_cut.append(sentence_cut[i]) #将原文的文字内容复制进来
    #np.savetxt(r'祝福-hier.txt', Hier_sentence_cut,fmt = '%s',newline=',',delimiter=',') 
        
    text_index=np.zeros((len(words_len),max(words_len)))
    #print(len(words_len),max(words_len))
    for i in range(len(words_len)):
        for j in range(max(words_len)):
            if(j<words_len[i]):
                text_index[i][j]=j    
            else:
                text_index[i][j]=max(words_len)*l   
    text_words=['a' for i in range(sum(words_len))]   #填充词的内容为a
    #print(sum(words_len),len(Hier_sentence_cut))
    #print(len(nomalize_pro))
    #np.savetxt(r'祝福-pro.txt',nomalize_pro,fmt = '%f',newline=',',delimiter=',') 


    ran_down=0
    ran_up=1        
    for i in range(len(Hier_sentence_cut)):
        m=0.0
        r=random.uniform(ran_down, ran_up)
        subdivision=0
        text_sequence=[]
        words_index=0
        for j in range(len(nomalize_pro)):
            if(nomalize_pro[j]!=0):
                m=m+nomalize_pro[j]
                if(r<=m):
                    subdivision=j    #轮盘赌算法选第几个subdivision
        
        if(nomalize_pro[subdivision]!=0):
            for q in range(len(text_index[subdivision])):
                if(text_index[subdivision][q]!=max(words_len)*l):
                    text_sequence.append(text_index[subdivision][q])
            if(len(text_sequence)==0):
                ran_up=ran_up-nomalize_pro[subdivision]
                nomalize_pro[subdivision]=0.0 
                r=random.uniform(ran_down, ran_up)
                subdivision=0
                text_sequence=[]
                words_index=0
                for j in range(len(nomalize_pro)):
                    if(nomalize_pro[j]!=0):
                        m=m+nomalize_pro[j]
                        if(r<=m):
                            subdivision=j    #轮盘赌算法选第几个subdivision
                if(nomalize_pro[subdivision]!=0):
                    for q in range(len(text_index[subdivision])):
                        if(text_index[subdivision][q]!=max(words_len)*l):
                            text_sequence.append(text_index[subdivision][q])
                part=random.randint(0, len(text_sequence)-1)  #产生一个选位置的随机数
                b=int(text_sequence[part])
                text_index[subdivision][b]=max(words_len)*l
                for k in range(subdivision):
                    words_index=words_index+words_len[k]
                text_words[words_index+b]=Hier_sentence_cut[i]
            else:
                part=random.randint(0, len(text_sequence)-1)  #产生一个选位置的随机数
                b=int(text_sequence[part])
                text_index[subdivision][b]=max(words_len)*l
                for k in range(subdivision):
                    words_index=words_index+words_len[k]
                text_words[words_index+b]=Hier_sentence_cut[i]

    hierarchy_sentence_cut=[]
    for i in range(len(text_words)):
        hierarchy_sentence_cut.append(text_words[i])     #hierchchy 文本
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
                pos=twords.index(subwords[q])   
                m=subwords_number[q]  #词在窗口中出现的次数
                v=v+tNr[pos]*m   #词对应的SVD向量
                sum_m=sum_m+m*m
        if sum_m==0:
            V[i]=0; null+=1
        else:
            V[i]=v/math.sqrt(sum_m)
    
    return V,windows_number,Real_random_nomalize_pro

def Real_random_slot_hierarchy_V_vectors(J,E,l,a,chap_len,paragraph_len,words_len,Real_random_nomalize_pro,sentence_cut,dimension,swords,twords,tNr):
    
    Real_random_nomalize_pro = copy.deepcopy(Real_random_nomalize_pro)
    null = 0
    Hier_sentence_cut=[]
    for i in range(sum(words_len)):
        Hier_sentence_cut.append(sentence_cut[i]) #将原文的文字内容复制进来
   
        
    text_index=np.zeros((len(words_len),max(words_len)))
    #print(len(words_len),max(words_len))
    for i in range(len(words_len)):
        for j in range(max(words_len)):
            if(j<words_len[i]):
                text_index[i][j]=j    
            else:
                text_index[i][j]=max(words_len)*l   
    text_words=['a' for i in range(sum(words_len))]   #填充词的内容为a
    


    ran_down=0
    ran_up=1        
    for i in range(len(Hier_sentence_cut)):
        m=0.0
        r=random.uniform(ran_down, ran_up)
        subdivision=0
        text_sequence=[]
        words_index=0
        for j in range(len(Real_random_nomalize_pro)):
            if(Real_random_nomalize_pro[j]!=0):
                m=m+Real_random_nomalize_pro[j]
                if(r<=m):
                    subdivision=j    #轮盘赌算法选第几个subdivision
        
        if(Real_random_nomalize_pro[subdivision]!=0):
            for q in range(len(text_index[subdivision])):
                if(text_index[subdivision][q]!=max(words_len)*l):
                    text_sequence.append(text_index[subdivision][q])
            if(len(text_sequence)==0):
                ran_up=ran_up-Real_random_nomalize_pro[subdivision]
                Real_random_nomalize_pro[subdivision]=0.0 
                r=random.uniform(ran_down, ran_up)
                subdivision=0
                text_sequence=[]
                words_index=0
                for j in range(len(Real_random_nomalize_pro)):
                    if(Real_random_nomalize_pro[j]!=0):
                        m=m+Real_random_nomalize_pro[j]
                        if(r<=m):
                            subdivision=j    #轮盘赌算法选第几个subdivision
                if(Real_random_nomalize_pro[subdivision]!=0):
                    for q in range(len(text_index[subdivision])):
                        if(text_index[subdivision][q]!=max(words_len)*l):
                            text_sequence.append(text_index[subdivision][q])
                part=random.randint(0, len(text_sequence)-1)  #产生一个选位置的随机数
                b=int(text_sequence[part])
                text_index[subdivision][b]=max(words_len)*l
                for k in range(subdivision):
                    words_index=words_index+words_len[k]
                text_words[words_index+b]=Hier_sentence_cut[i]
            else:
                part=random.randint(0, len(text_sequence)-1)  #产生一个选位置的随机数
                b=int(text_sequence[part])
                text_index[subdivision][b]=max(words_len)*l
                for k in range(subdivision):
                    words_index=words_index+words_len[k]
                text_words[words_index+b]=Hier_sentence_cut[i]

    hierarchy_sentence_cut=[]
    for i in range(len(text_words)):
        hierarchy_sentence_cut.append(text_words[i])     #hierchchy 文本
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
                pos=twords.index(subwords[q])   
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

if __name__ == '__main__':
    
    texts = ["金庸-射雕英雄传缩减版 -V1.txt"]  #length=11902  句子(K=5,H=4,l=11)
    #texts = ["孔乙己传.txt"]  #length=3677 段落(K=3,H=3,l=135)  句子(K=4,H=4,l=14)
    #texts = ["古典爱情.txt"]   #length=11376 段落(K=4,H=3,l=140)  句子(K=5,H=4,l=11)
    #texts = ["命若琴弦.txt"]  #length=6454,段落(K=3,H=4,l=100) 句子(K=4,H=4,l=25)
    #texts = ["酒徒刘以鬯.txt"]  #length=56412 段落(K=4,H=5,l=90) 句子(K=5,H=5,l=18)
    #texts = ["家.txt"]   #length=140783 段落(K=4,H=6,l=108) 句子(K=5,H=6,l=18)
    #texts = ["春.txt"]   #length=149978 段落(K=4,H=6,l=115)  句子(K=5,H=6,l=19)
    #texts = ["秋.txt"]   #length=185719 (K=4,H=6,l=143)    
    stop_sentence=readText("all_stopwords.txt").replace('\n','')  #停止词词库
    a=120  #窗口大小
    thr_factor=0.6# #
    dimension=50  #dimension
    K=5 #Level  
    H=4 #subdivision
    J=1
    E=5
    l=11
    
    data = []
    for title in texts:
        chap_len,paragraph_len,words_len,sentence = segments(title)
        sentence_cut,words,twords,twords_number,words_number,swords=cleaningText(a, thr_factor,sentence, stop_sentence)
        tN=N_matrix(a, sentence_cut, words, twords, twords_number, words_number)
        VTr=svd(dimension,tN)
        tNr=new_tN(dimension, tN, VTr)
        origin_V,origin_windows_number=origin_V_vectors(a, sentence_cut, dimension, swords,twords, tNr)
        random_V,random_windows_number=random_V_vectors(a, sentence_cut, dimension, swords,twords, tNr)
        hierarchy_V,hierarchy_windows_number,hierarchy_random_nomalize_pro=hierarchy_V_vectors(K, H, J, E, l,a, sentence_cut, dimension, swords,twords, tNr)
        Real_hierarchy_V,Real_hierarchy_windows_number,Real_hierarchy_random_nomalize_pro=Real_hierarchy_V_vectors(J,E,l,a,chap_len,paragraph_len,words_len,sentence_cut,dimension,swords,twords,tNr)
        origin_correction=correction_function(origin_V, origin_windows_number)
        random_correction=correction_function(random_V, random_windows_number)
        hierarchy_correction=correction_function(hierarchy_V, hierarchy_windows_number)
        Real_hierarchy_correction=correction_function(Real_hierarchy_V, Real_hierarchy_windows_number)
        
        
      
        
        
        #从其他文本里面随机选择一部分放入层级文本里面,按照原文的顺序插入
        select_chap_len,select_paragraph_len,select_words_len,select_sentence = segments("E:/spyder/Fractality/ChineseHierchical-V8/金庸-射雕英雄传缩减版-V2.txt")
        select_sentence_cut,select_words,select_twords,select_twords_number,select_words_number,select_swords=cleaningText(a, thr_factor,select_sentence, stop_sentence)
        select_tN=N_matrix(a, select_sentence_cut, select_words, select_twords, select_twords_number, select_words_number)
        select_VTr=svd(dimension,select_tN)
        select_tNr=new_tN(dimension, select_tN, select_VTr)
        select_hierarchy_sentence_cut=[]
        select_sequence=[]
        add_sequence=[]
        for m in range(len(select_sentence_cut)):  #随机的选按照原文的顺序插入
            select_sequence.append(m)
        random.shuffle(select_sequence)
        #print(select_sequence)
        for n in range(len(sentence_cut)):  #随机的选按照原文的顺序插入
            add_sequence.append(select_sequence[n])
        add_sequence.sort()  #升序，得到被插入原文词出现的顺序       
        for s in range(len(sentence_cut)):
            select_hierarchy_sentence_cut.append(select_sentence_cut[add_sequence[s]])
        select_origin_V,select_origin_windows_number=origin_V_vectors(a, select_hierarchy_sentence_cut, dimension, select_swords, select_twords,select_tNr)
        select_origin_correction=correction_function(select_origin_V, select_origin_windows_number)
        #H,k的层级结构
        select_hierarchy_V,select_hierarchy_windows_number=random_slot_hierarchy_V_vectors(K, H, J, E, l, a, hierarchy_random_nomalize_pro, select_hierarchy_sentence_cut, dimension, select_swords, select_twords,select_tNr)
        select_hierarchy_correction=correction_function(select_hierarchy_V, select_hierarchy_windows_number)
        
        #真实的文本的层级结构，章，段，句，词
        Real_select_hierarchy_V,Real_select_hierarchy_windows_number=Real_random_slot_hierarchy_V_vectors(J,E,l,a,chap_len,paragraph_len,words_len,Real_hierarchy_random_nomalize_pro,select_hierarchy_sentence_cut, dimension, select_swords, select_twords,select_tNr)
        Real_select_hierarchy_correction=correction_function(Real_select_hierarchy_V, Real_select_hierarchy_windows_number)
        
        
        
        #从其他文本里面随机选择一部分放入层级文本里面,按照随机的顺序插入
        random_select_hierarchy_sentence_cut=[]
        random_select_sequence=[]
        for p in range(len(select_sentence_cut)):  #随机的选按照随机的顺序插入
            random_select_sequence.append(p)
        random.shuffle(random_select_sequence)
        for q in range(len(sentence_cut)):
            random_select_hierarchy_sentence_cut.append(select_sentence_cut[random_select_sequence[q]])
       
        random_select_origin_V,random_select_origin_windows_number=origin_V_vectors(a, random_select_hierarchy_sentence_cut, dimension, select_swords, select_twords,select_tNr)
        random_select_origin_correction=correction_function(random_select_origin_V, random_select_origin_windows_number)
        plt.loglog(range(1,len(random_select_origin_correction)+1),random_select_origin_correction,linewidth =1.5, label='random_select_origin')
        plt.legend()
        
        #H,k的层级结构
        random_select_hierarchy_V,random_select_hierarchy_windows_number=random_slot_hierarchy_V_vectors(K, H, J, E, l, a, hierarchy_random_nomalize_pro, random_select_hierarchy_sentence_cut, dimension, select_swords, select_twords,select_tNr)
        random_select_hierarchy_correction=correction_function(random_select_hierarchy_V, random_select_hierarchy_windows_number)
        plt.loglog(range(1,len(random_select_hierarchy_correction)+1),random_select_hierarchy_correction,linewidth =1, label='random_select_hierarchy')
        plt.legend()
        
        
        
        #二项式拟合句子长度
        #chap_len,paragraph_len,words_len,sentence = segments(title)
        def words_len_fit(x,y_fit):
            a,b,c=y_fit.tolist()
            return a*x**2+b*x+c
        words_len.sort()   #升序，得到被插入原文词出现的顺序    
        words_sequence=[]
        for i in range(len(words_len)):
            words_sequence.append(i)
        y_fit=np.polyfit(words_sequence,words_len,2)#二次多项式拟合
        fit_words_len=words_len_fit(np.array(words_sequence),y_fit)
        ifit_words_len=[]
        for i in range(len(fit_words_len)):
            ifit_words_len.append(int(fit_words_len[i]))
        print(sum(ifit_words_len))
        print(sum(words_len))
        fit_Real_hierarchy_V,fit_Real_hierarchy_windows_number,fit_Real_random_nomalize_pro=Real_hierarchy_V_vectors(J,E,l,a,chap_len,paragraph_len,ifit_words_len,sentence_cut,dimension,swords,twords,tNr)
        fit_Real_hierarchy_correction=correction_function(fit_Real_hierarchy_V, fit_Real_hierarchy_windows_number)
        
        #随机打乱二项式拟合得到的句子长度
        random.shuffle(ifit_words_len)
        fit_Real_hierarchy_V,fit_Real_hierarchy_windows_number,fit_Real_random_nomalize_pro=Real_hierarchy_V_vectors(J,E,l,a,chap_len,paragraph_len,ifit_words_len,sentence_cut,dimension,swords,twords,tNr)
        fit_Real_hierarchy_correction=correction_function(fit_Real_hierarchy_V, fit_Real_hierarchy_windows_number)
        plt.loglog(range(1,len(fit_Real_hierarchy_correction)+1),fit_Real_hierarchy_correction,linewidth =1.5,label='random_fit_Real_hierarchy')
        plt.legend()
        
        
        
        
        
        

        titleString = title.split(".")[0] + " " 
        plt.title(titleString, fontproperties=fontP)
        plt.xlabel("Time(in word distance units)", fontproperties=fontP)
        plt.ylabel("AutoCorrection", fontproperties=fontP)
        plt.loglog(range(1,len(origin_correction)+1),origin_correction,linewidth =2.0, color='green',label='origin')
        #np.savetxt('d='+str(dimension)+'a='+str(a)+'thr='+str(thr_factor)+str(title), origin_correction,fmt = '%f')        
        plt.loglog(range(1,len(random_correction)+1),random_correction,linewidth =1.5, color='red', label='random')
        plt.loglog(range(1,len(hierarchy_correction)+1),hierarchy_correction,linewidth =1, color='blue', label='hierarchy')
        plt.loglog(range(1,len(Real_hierarchy_correction)+1),Real_hierarchy_correction,linewidth =1.5, color='black', label='Real_hierarchy')
        #np.savetxt('d='+str(dimension)+'a='+str(a)+'thr='+str(thr_factor)+str(title), Real_hierarchy_correction,fmt = '%f')
        plt.loglog(range(1,len(Real_select_hierarchy_correction)+1), Real_select_hierarchy_correction,linewidth =1.5, color='deepskyblue', label='Real_select_hierarchy')
        plt.loglog(range(1,len(select_origin_correction)+1),select_origin_correction,linewidth =1.5, color='purple', label='select_origin')       
        plt.loglog(range(1,len(fit_Real_hierarchy_correction)+1),fit_Real_hierarchy_correction,linewidth =1.5, color='yellow', label='fit_Real_hierarchy')
        plt.legend()
        plt.show()
        
# =============================================================================
#         titleString = title.split(".")[0] + " " 
#         plt.title(titleString, fontproperties=fontP)
#         plt.xlabel("Time(in word distance units)", fontproperties=fontP)
#         plt.ylabel("AutoCorrection", fontproperties=fontP)
#         plt.loglog(range(1,len(random_correction)+1),random_correction,linewidth =1.5,label='random,d=50')
#         for i in range(1,5):
#             dimension=20*i
#             Ur=svd(dimension,tN)
#             #tNr=new_tN(dimension, tN, VTr)
#             origin_V,origin_windows_number=origin_V_vectors(a, sentence_cut, dimension, swords, Ur)
#             random_V,random_windows_number=random_V_vectors(a, sentence_cut, dimension, swords, Ur)
#             hierarchy_V,hierarchy_windows_number=hierarchy_V_vectors(K, H, J, E, l,a, sentence_cut, dimension, swords, Ur)
#             origin_correction=correction_function(origin_V, origin_windows_number)
#             plt.loglog(range(1,len(origin_correction)+1),origin_correction,linewidth =2.0,label='d='+str(dimension))
#         plt.legend()
#         plt.show()
# =============================================================================
        
        
        
        
        
        
        
        



    


