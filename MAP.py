import torch
import pdb
import os
import numpy as np
import sklearn.metrics.pairwise as skp
import gc
import time
def comp_Ap(list_retrieval):
    m=0;
    Ap=0.;
    for i in range(len(list_retrieval)):
        if list_retrieval[i]:
            m+=1
            Ap+=m/(i+1)
    return Ap/m

def comp_rc(binary,top):
    r = 0;
    for i in range(1,top+1):
        if binary[i]:
	        r+=1
	        break
    return r 

def comp_MAp(ranks,clusters):
    MAp=0;
    recall = [0]*3;
    top = [1,5,10]
    #f = open('101_circle_long.txt','w')

    for i in range(ranks.shape[0]):
        binary=[clusters[i]==clusters[j] for j in ranks[i]]
        MAp+=comp_Ap(binary)
        for j in range(3):
            r = comp_rc(binary,top[j])
            #if j == 0:
            #   f.write(str(i)+';'+str(ranks[i][:11])+ str(list(similarity[i][ranks[i][:11]]))+'\n')
            recall[j] += r
    #f.close()
    recall=[r/ranks.shape[0] for r in recall] 
    return MAp/ranks.shape[0],recall


def Test(ranks,clusters):
    
    st = time.time()
    MAp,recall = comp_MAp(ranks,clusters);
    print('map_time:',time.time() - st)
    return MAp,recall 

    
    

    



