import numpy as np
from numpy import unravel_index
import matplotlib.pyplot as plt
import scipy.stats


def createM(noS,prob1,prob2,rewardvector):
    M = np.zeros((noS,noS,6,6,noS,noS,2,2)) #s_t^1,s_t^2,r_t^1,r_t^2,s_t-1^1,s_t-1^2,a_t^1,a_t^2
    for s1 in range(noS-2):
        for s2 in range(noS-2):
            M[s1+1,s2+1,:,:,s1,s2,1,1] = prob1*prob1
            M[s1,s2+1,:,:,s1,s2,1,1] = prob2*prob1
            M[s1+1,s2,:,:,s1,s2,1,1] = prob1*prob2
            M[s1,s2,:,:,s1,s2,1,1] = prob2*prob2
            M[s1+2,s2+1,:,:,s1,s2,1,1] = prob1*prob2
            M[s1+1,s2+2,:,:,s1,s2,1,1] = prob1*prob2
            M[s1+2,s2+2,:,:,s1,s2,1,1] = prob2*prob2
            M[s1+2,s2,:,:,s1,s2,1,1] = prob2*prob2
            M[s1,s2+2,:,:,s1,s2,1,1] = prob2*prob2
            
    M[noS-1,noS-1,:,:,noS-2,noS-2,1,1] = prob1*prob1+prob2*prob2 + 2*prob1*prob2 
    M[noS-1,noS-2,:,:,noS-2,noS-2,1,1] = prob1*prob2 +prob2*prob2 
    M[noS-2,noS-1,:,:,noS-2,noS-2,1,1] = prob1*prob2+prob2*prob2 
    M[noS-2,noS-2,:,:,noS-2,noS-2,1,1] = prob2*prob2
    M[noS-1,noS-1,:,:,noS-1,noS-1,1,1] = 1
    
    M[noS-1,noS-1,:,:,noS-2,noS-1,1,1] = prob1*prob1 + 3*prob2*prob1 + 2*prob2*prob2
    M[noS-2,noS-1,:,:,noS-2,noS-1,1,1] = prob2*prob1 +2*prob2*prob2
    M[noS-1,noS-1,:,:,noS-1,noS-2,1,1] = prob1*prob1 + 3*prob2*prob1 + 2*prob2*prob2
    M[noS-1,noS-2,:,:,noS-1,noS-2,1,1] = prob2*prob1 +2*prob2*prob2
    
    
    
    for s2 in range(noS-2):
        M[noS-1,s2+1,:,:,noS-2,s2,1,1] = (prob1+prob2)*prob1 
        M[noS-1,s2,:,:,noS-2,s2,1,1] = (prob1+prob2)*prob2
        M[noS-1,s2+2,:,:,noS-2,s2,1,1] = (prob1+prob2)*prob2
        M[noS-2,s2+1,:,:,noS-2,s2,1,1] = prob1*prob2
        M[noS-2,s2,:,:,noS-2,s2,1,1] = prob2*prob2
        M[noS-2,s2+2,:,:,noS-2,s2,1,1] = prob2*prob2
        M[noS-1,s2,:,:,noS-1,s2,1,1] = 1*prob2
        M[noS-1,s2+2,:,:,noS-1,s2,1,1] = 1*prob2
        M[noS-1,s2+1,:,:,noS-1,s2,1,1] = 1*prob1
    
    for s1 in range(noS-2):
        M[s1+1,noS-1,:,:,s1,noS-2,1,1] = (prob1+prob2)*prob1 
        M[s1,noS-1,:,:,s1,noS-2,1,1] = (prob1+prob2)*prob2
        M[s1+2,noS-1,:,:,s1,noS-2,1,1] = (prob1+prob2)*prob2
        M[s1+1,noS-2,:,:,s1,noS-2,1,1] = prob1*prob2
        M[s1,noS-2,:,:,s1,noS-2,1,1] = prob2*prob2
        M[s1+2,noS-2,:,:,s1,noS-2,1,1] = prob2*prob2
        M[s1,noS-1,:,:,s1,noS-1,1,1] = 1*prob2
        M[s1+2,noS-1,:,:,s1,noS-1,1,1] = 1*prob2
        M[s1+1,noS-1,:,:,s1,noS-1,1,1] = 1*prob1
    
     
            
    for s1 in range(2,noS):
        for s2 in range(2,noS):
            M[s1-1,s2-1,:,:,s1,s2,0,0] = prob1*prob1
            M[s1,s2-1,:,:,s1,s2,0,0] = prob2*prob1
            M[s1-1,s2,:,:,s1,s2,0,0] = prob1*prob2
            M[s1,s2,:,:,s1,s2,0,0] = prob2*prob2
            M[s1-2,s2-1,:,:,s1,s2,0,0] = prob1*prob2
            M[s1-1,s2-2,:,:,s1,s2,0,0] = prob1*prob2
            M[s1-2,s2-2,:,:,s1,s2,0,0] = prob2*prob2
            M[s1-2,s2,:,:,s1,s2,0,0] = prob2*prob2
            M[s1,s2-2,:,:,s1,s2,0,0] = prob2*prob2
        
    M[0,0,:,:,1,1,0,0] = prob1*prob1+prob2*prob2 + 2*prob1*prob2 
    M[0,1,:,:,1,1,0,0] = prob1*prob2
    M[1,0,:,:,1,1,0,0] = prob1*prob2
    M[1,1,:,:,1,1,0,0] = prob2*prob2
    M[0,0,:,:,0,0,0,0] = 1
    
    
    M[0,0,:,:,1,0,0,0] = prob1*prob1 + 3*prob2*prob1 + 2*prob2*prob2
    M[1,0,:,:,1,0,0,0] = prob2*prob1 +2*prob2*prob2
    M[0,0,:,:,0,1,0,0] = prob1*prob1 + 3*prob2*prob1 + 2*prob2*prob2
    M[0,1,:,:,0,1,0,0] = prob2*prob1 +2*prob2*prob2
    
    

    for s2 in range(2,noS):
        M[0,s2-1,:,:,1,s2,0,0] = (prob1+prob2)*prob1 
        M[0,s2,:,:,1,s2,0,0] = (prob1+prob2)*prob2
        M[0,s2-2,:,:,1,s2,0,0] = (prob1+prob2)*prob2
        M[1,s2-1,:,:,1,s2,0,0] = prob1*prob2
        M[1,s2,:,:,1,s2,0,0] = prob2*prob2
        M[1,s2-2,:,:,1,s2,0,0] = prob2*prob2
        M[0,s2,:,:,0,s2,0,0] = 1*prob2
        M[0,s2-2,:,:,0,s2,0,0] = 1*prob2
        M[0,s2-1,:,:,0,s2,0,0] = 1*prob1
    
    for s1 in range(2,noS):
        M[s1-1,0,:,:,s1,1,0,0] = (prob1+prob2)*prob1 
        M[s1,0,:,:,s1,1,0,0] = (prob1+prob2)*prob2
        M[s1-2,0,:,:,s1,1,0,0] = (prob1+prob2)*prob2
        M[s1-1,1,:,:,s1,1,0,0] = prob1*prob2
        M[s1,1,:,:,s1,1,0,0] = prob2*prob2
        M[s1-2,1,:,:,s1,1,0,0] = prob2*prob2
        M[s1,0,:,:,s1,0,0,0] = 1*prob2
        M[s1-2,0,:,:,s1,0,0,0] = 1*prob2
        M[s1-1,0,:,:,s1,0,0,0] = 1*prob1
    
            
    for s1 in range(noS-2):
        for s2 in range(2,noS):
            M[s1+1,s2-1,:,:,s1,s2,1,0] = prob1*prob1
            M[s1,s2-1,:,:,s1,s2,1,0] = prob2*prob1
            M[s1+1,s2,:,:,s1,s2,1,0] = prob1*prob2
            M[s1,s2,:,:,s1,s2,1,0] = prob2*prob2
            M[s1+2,s2-1,:,:,s1,s2,1,0] = prob1*prob2
            M[s1+1,s2-2,:,:,s1,s2,1,0] = prob1*prob2
            M[s1+2,s2-2,:,:,s1,s2,1,0] = prob2*prob2
            M[s1+2,s2,:,:,s1,s2,1,0] = prob2*prob2
            M[s1,s2-2,:,:,s1,s2,1,0] = prob2*prob2
            
    M[noS-1,0,:,:,noS-2,1,1,0] = prob1*prob1+prob2*prob2 + 2*prob1*prob2 
    M[noS-1,1,:,:,noS-2,1,1,0] = prob1*prob2
    M[noS-2,0,:,:,noS-2,1,1,0] = prob1*prob2
    M[noS-2,1,:,:,noS-2,1,1,0] = prob2*prob2
    M[noS-1,0,:,:,noS-1,0,1,0] = 1
            
    
    M[noS-1,0,:,:,noS-2,0,1,0] = prob1*prob1 + 3*prob2*prob1 + 2*prob2*prob2
    M[noS-2,0,:,:,noS-2,0,1,0] = prob2*prob1 +2*prob2*prob2
    M[noS-1,0,:,:,noS-1,1,1,0] = prob1*prob1 + 3*prob2*prob1 + 2*prob2*prob2
    M[noS-1,1,:,:,noS-1,1,1,0] = prob2*prob1 +2*prob2*prob2
    
    
    
    for s2 in range(2,noS):
        M[noS-1,s2-1,:,:,noS-2,s2,1,0] = (prob1+prob2)*prob1 
        M[noS-1,s2,:,:,noS-2,s2,1,0] = (prob1+prob2)*prob2
        M[noS-1,s2-2,:,:,noS-2,s2,1,0] = (prob1+prob2)*prob2
        M[noS-2,s2-1,:,:,noS-2,s2,1,0] = prob1*prob2
        M[noS-2,s2,:,:,noS-2,s2,1,0] = prob2*prob2
        M[noS-2,s2-2,:,:,noS-2,s2,1,0] = prob2*prob2
        M[noS-1,s2,:,:,noS-1,s2,1,0] = 1*prob2
        M[noS-1,s2-2,:,:,noS-1,s2,1,0] = 1*prob2
        M[noS-1,s2-1,:,:,noS-1,s2,1,0] = 1*prob1
    
    for s1 in range(noS-2):
        M[s1+1,0,:,:,s1,1,1,0] = (prob1+prob2)*prob1 
        M[s1,0,:,:,s1,1,1,0] = (prob1+prob2)*prob2
        M[s1+2,0,:,:,s1,1,1,0] = (prob1+prob2)*prob2
        M[s1+1,1,:,:,s1,1,1,0] = prob1*prob2
        M[s1,1,:,:,s1,1,1,0] = prob2*prob2
        M[s1+2,1,:,:,s1,1,1,0] = prob2*prob2
        M[s1,0,:,:,s1,0,1,0] = 1*prob2
        M[s1+2,0,:,:,s1,0,1,0] = 1*prob2
        M[s1+1,0,:,:,s1,0,1,0] = 1*prob1
    
    
    for s1 in range(2,noS):
        for s2 in range(noS-2):
            M[s1-1,s2+1,:,:,s1,s2,0,1] = prob1*prob1
            M[s1,s2+1,:,:,s1,s2,0,1] = prob2*prob1
            M[s1-1,s2,:,:,s1,s2,0,1] = prob1*prob2
            M[s1,s2,:,:,s1,s2,0,1] = prob2*prob2
            M[s1-2,s2+1,:,:,s1,s2,0,1] = prob1*prob2
            M[s1-1,s2+2,:,:,s1,s2,0,1] = prob1*prob2
            M[s1-2,s2+2,:,:,s1,s2,0,1] = prob2*prob2
            M[s1-2,s2,:,:,s1,s2,0,1] = prob2*prob2
            M[s1,s2+2,:,:,s1,s2,0,1] = prob2*prob2
    
    M[0,noS-1,:,:,1,noS-2,0,1] = prob1*prob1+prob2*prob2 + 2*prob1*prob2 
    M[1,noS-1,:,:,1,noS-2,0,1] = prob1*prob2
    M[0,noS-2,:,:,1,noS-2,0,1] = prob1*prob2
    M[1,noS-2,:,:,1,noS-2,0,1] = prob2*prob2
    M[0,noS-1,:,:,0,noS-1,0,1] = 1
    
    M[0,noS-1,:,:,1,noS-1,0,1] = prob1*prob1 + 3*prob2*prob1 + 2*prob2*prob2
    M[1,noS-1,:,:,1,noS-1,0,1] = prob2*prob1 +2*prob2*prob2
    M[0,noS-1,:,:,0,noS-2,0,1] = prob1*prob1 + 3*prob2*prob1 + 2*prob2*prob2
    M[0,noS-2,:,:,0,noS-2,0,1] = prob2*prob1 +2*prob2*prob2
    
    
    for s2 in range(noS-2):
        M[0,s2+1,:,:,1,s2,0,1] = (prob1+prob2)*prob1 
        M[0,s2,:,:,1,s2,0,1] = (prob1+prob2)*prob2
        M[0,s2+2,:,:,1,s2,0,1] = (prob1+prob2)*prob2
        M[1,s2+1,:,:,1,s2,0,1] = prob1*prob2
        M[1,s2,:,:,1,s2,0,1] = prob2*prob2
        M[1,s2+2,:,:,1,s2,0,1] = prob2*prob2
        M[0,s2,:,:,0,s2,0,1] = 1*prob2
        M[0,s2+2,:,:,0,s2,0,1] = 1*prob2
        M[0,s2+1,:,:,0,s2,0,1] = 1*prob1
    
    for s1 in range(2,noS):
        M[s1-1,noS-1,:,:,s1,noS-2,0,1] = (prob1+prob2)*prob1 
        M[s1,noS-1,:,:,s1,noS-2,0,1] = (prob1+prob2)*prob2
        M[s1-2,noS-1,:,:,s1,noS-2,0,1] = (prob1+prob2)*prob2
        M[s1-1,noS-2,:,:,s1,noS-2,0,1] = prob1*prob2
        M[s1,noS-2,:,:,s1,noS-2,0,1] = prob2*prob2
        M[s1-2,noS-2,:,:,s1,noS-2,0,1] = prob2*prob2
        M[s1,noS-1,:,:,s1,noS-1,0,1] = 1*prob2
        M[s1-2,noS-1,:,:,s1,noS-1,0,1] = 1*prob2
        M[s1-1,noS-1,:,:,s1,noS-1,0,1] = 1*prob1
    
    
    #now create reward matrix thingy
            
    for s1 in range(noS):
        for s2 in range(noS):
            r1=0
            r2 = 0
            if s2>=s1:
                r1 = -2
                r2 = -2
            if s1<((noS+1)/2 - 1):
                r1+=-1
            if s2<((noS+1)/2 - 1):
                r2+=-1
            for t1 in range(len(rewardvector)):
                for t2 in range(len(rewardvector)):
                    if rewardvector[t1] != r1 or rewardvector[t2] != r2:
                        M[s1,s2,t1,t2,:,:,:,:] = 0
    
    
    return M
    
import numpy as np
def createMT(noS,prob1,prob2,rewardvector):
    M = np.zeros((noS,noS,6,6,noS,noS,2,2)) #s_t^1,s_t^2,r_t^1,r_t^2,s_t-1^1,s_t-1^2,a_t^1,a_t^2
    for s1 in range(noS-2):
        for s2 in range(noS-2):
            M[s1+1,s2+1,:,:,s1,s2,1,1] = prob1*prob1
            M[s1,s2+1,:,:,s1,s2,1,1] = prob2*prob1
            M[s1+1,s2,:,:,s1,s2,1,1] = prob1*prob2
            M[s1,s2,:,:,s1,s2,1,1] = prob2*prob2
            M[s1+2,s2+1,:,:,s1,s2,1,1] = prob1*prob2
            M[s1+1,s2+2,:,:,s1,s2,1,1] = prob1*prob2
            M[s1+2,s2+2,:,:,s1,s2,1,1] = prob2*prob2
            M[s1+2,s2,:,:,s1,s2,1,1] = prob2*prob2
            M[s1,s2+2,:,:,s1,s2,1,1] = prob2*prob2
            
    M[noS-1,noS-1,:,:,noS-2,noS-2,1,1] = prob1*prob1+prob2*prob2 + 2*prob1*prob2 
    M[noS-1,noS-2,:,:,noS-2,noS-2,1,1] = prob1*prob2 +prob2*prob2 
    M[noS-2,noS-1,:,:,noS-2,noS-2,1,1] = prob1*prob2+prob2*prob2 
    M[noS-2,noS-2,:,:,noS-2,noS-2,1,1] = prob2*prob2
    M[noS-1,noS-1,:,:,noS-1,noS-1,1,1] = 1
    
    M[noS-1,noS-1,:,:,noS-2,noS-1,1,1] = prob1*prob1 + 3*prob2*prob1 + 2*prob2*prob2
    M[noS-2,noS-1,:,:,noS-2,noS-1,1,1] = prob2*prob1 +2*prob2*prob2
    M[noS-1,noS-1,:,:,noS-1,noS-2,1,1] = prob1*prob1 + 3*prob2*prob1 + 2*prob2*prob2
    M[noS-1,noS-2,:,:,noS-1,noS-2,1,1] = prob2*prob1 +2*prob2*prob2
    
    
    
    for s2 in range(noS-2):
        M[noS-1,s2+1,:,:,noS-2,s2,1,1] = (prob1+prob2)*prob1 
        M[noS-1,s2,:,:,noS-2,s2,1,1] = (prob1+prob2)*prob2
        M[noS-1,s2+2,:,:,noS-2,s2,1,1] = (prob1+prob2)*prob2
        M[noS-2,s2+1,:,:,noS-2,s2,1,1] = prob1*prob2
        M[noS-2,s2,:,:,noS-2,s2,1,1] = prob2*prob2
        M[noS-2,s2+2,:,:,noS-2,s2,1,1] = prob2*prob2
        M[noS-1,s2,:,:,noS-1,s2,1,1] = 1*prob2
        M[noS-1,s2+2,:,:,noS-1,s2,1,1] = 1*prob2
        M[noS-1,s2+1,:,:,noS-1,s2,1,1] = 1*prob1
    
    for s1 in range(noS-2):
        M[s1+1,noS-1,:,:,s1,noS-2,1,1] = (prob1+prob2)*prob1 
        M[s1,noS-1,:,:,s1,noS-2,1,1] = (prob1+prob2)*prob2
        M[s1+2,noS-1,:,:,s1,noS-2,1,1] = (prob1+prob2)*prob2
        M[s1+1,noS-2,:,:,s1,noS-2,1,1] = prob1*prob2
        M[s1,noS-2,:,:,s1,noS-2,1,1] = prob2*prob2
        M[s1+2,noS-2,:,:,s1,noS-2,1,1] = prob2*prob2
        M[s1,noS-1,:,:,s1,noS-1,1,1] = 1*prob2
        M[s1+2,noS-1,:,:,s1,noS-1,1,1] = 1*prob2
        M[s1+1,noS-1,:,:,s1,noS-1,1,1] = 1*prob1
    
     
            
    for s1 in range(2,noS):
        for s2 in range(2,noS):
            M[s1-1,s2-1,:,:,s1,s2,0,0] = prob1*prob1
            M[s1,s2-1,:,:,s1,s2,0,0] = prob2*prob1
            M[s1-1,s2,:,:,s1,s2,0,0] = prob1*prob2
            M[s1,s2,:,:,s1,s2,0,0] = prob2*prob2
            M[s1-2,s2-1,:,:,s1,s2,0,0] = prob1*prob2
            M[s1-1,s2-2,:,:,s1,s2,0,0] = prob1*prob2
            M[s1-2,s2-2,:,:,s1,s2,0,0] = prob2*prob2
            M[s1-2,s2,:,:,s1,s2,0,0] = prob2*prob2
            M[s1,s2-2,:,:,s1,s2,0,0] = prob2*prob2
        
    M[0,0,:,:,1,1,0,0] = prob1*prob1+prob2*prob2 + 2*prob1*prob2 
    M[0,1,:,:,1,1,0,0] = prob1*prob2
    M[1,0,:,:,1,1,0,0] = prob1*prob2
    M[1,1,:,:,1,1,0,0] = prob2*prob2
    M[0,0,:,:,0,0,0,0] = 1
    
    
    M[0,0,:,:,1,0,0,0] = prob1*prob1 + 3*prob2*prob1 + 2*prob2*prob2
    M[1,0,:,:,1,0,0,0] = prob2*prob1 +2*prob2*prob2
    M[0,0,:,:,0,1,0,0] = prob1*prob1 + 3*prob2*prob1 + 2*prob2*prob2
    M[0,1,:,:,0,1,0,0] = prob2*prob1 +2*prob2*prob2
    
    

    for s2 in range(2,noS):
        M[0,s2-1,:,:,1,s2,0,0] = (prob1+prob2)*prob1 
        M[0,s2,:,:,1,s2,0,0] = (prob1+prob2)*prob2
        M[0,s2-2,:,:,1,s2,0,0] = (prob1+prob2)*prob2
        M[1,s2-1,:,:,1,s2,0,0] = prob1*prob2
        M[1,s2,:,:,1,s2,0,0] = prob2*prob2
        M[1,s2-2,:,:,1,s2,0,0] = prob2*prob2
        M[0,s2,:,:,0,s2,0,0] = 1*prob2
        M[0,s2-2,:,:,0,s2,0,0] = 1*prob2
        M[0,s2-1,:,:,0,s2,0,0] = 1*prob1
    
    for s1 in range(2,noS):
        M[s1-1,0,:,:,s1,1,0,0] = (prob1+prob2)*prob1 
        M[s1,0,:,:,s1,1,0,0] = (prob1+prob2)*prob2
        M[s1-2,0,:,:,s1,1,0,0] = (prob1+prob2)*prob2
        M[s1-1,1,:,:,s1,1,0,0] = prob1*prob2
        M[s1,1,:,:,s1,1,0,0] = prob2*prob2
        M[s1-2,1,:,:,s1,1,0,0] = prob2*prob2
        M[s1,0,:,:,s1,0,0,0] = 1*prob2
        M[s1-2,0,:,:,s1,0,0,0] = 1*prob2
        M[s1-1,0,:,:,s1,0,0,0] = 1*prob1
    
            
    for s1 in range(noS-2):
        for s2 in range(2,noS):
            M[s1+1,s2-1,:,:,s1,s2,1,0] = prob1*prob1
            M[s1,s2-1,:,:,s1,s2,1,0] = prob2*prob1
            M[s1+1,s2,:,:,s1,s2,1,0] = prob1*prob2
            M[s1,s2,:,:,s1,s2,1,0] = prob2*prob2
            M[s1+2,s2-1,:,:,s1,s2,1,0] = prob1*prob2
            M[s1+1,s2-2,:,:,s1,s2,1,0] = prob1*prob2
            M[s1+2,s2-2,:,:,s1,s2,1,0] = prob2*prob2
            M[s1+2,s2,:,:,s1,s2,1,0] = prob2*prob2
            M[s1,s2-2,:,:,s1,s2,1,0] = prob2*prob2
            
    M[noS-1,0,:,:,noS-2,1,1,0] = prob1*prob1+prob2*prob2 + 2*prob1*prob2 
    M[noS-1,1,:,:,noS-2,1,1,0] = prob1*prob2
    M[noS-2,0,:,:,noS-2,1,1,0] = prob1*prob2
    M[noS-2,1,:,:,noS-2,1,1,0] = prob2*prob2
    M[noS-1,0,:,:,noS-1,0,1,0] = 1
            
    
    M[noS-1,0,:,:,noS-2,0,1,0] = prob1*prob1 + 3*prob2*prob1 + 2*prob2*prob2
    M[noS-2,0,:,:,noS-2,0,1,0] = prob2*prob1 +2*prob2*prob2
    M[noS-1,0,:,:,noS-1,1,1,0] = prob1*prob1 + 3*prob2*prob1 + 2*prob2*prob2
    M[noS-1,1,:,:,noS-1,1,1,0] = prob2*prob1 +2*prob2*prob2
    
    
    
    for s2 in range(2,noS):
        M[noS-1,s2-1,:,:,noS-2,s2,1,0] = (prob1+prob2)*prob1 
        M[noS-1,s2,:,:,noS-2,s2,1,0] = (prob1+prob2)*prob2
        M[noS-1,s2-2,:,:,noS-2,s2,1,0] = (prob1+prob2)*prob2
        M[noS-2,s2-1,:,:,noS-2,s2,1,0] = prob1*prob2
        M[noS-2,s2,:,:,noS-2,s2,1,0] = prob2*prob2
        M[noS-2,s2-2,:,:,noS-2,s2,1,0] = prob2*prob2
        M[noS-1,s2,:,:,noS-1,s2,1,0] = 1*prob2
        M[noS-1,s2-2,:,:,noS-1,s2,1,0] = 1*prob2
        M[noS-1,s2-1,:,:,noS-1,s2,1,0] = 1*prob1
    
    for s1 in range(noS-2):
        M[s1+1,0,:,:,s1,1,1,0] = (prob1+prob2)*prob1 
        M[s1,0,:,:,s1,1,1,0] = (prob1+prob2)*prob2
        M[s1+2,0,:,:,s1,1,1,0] = (prob1+prob2)*prob2
        M[s1+1,1,:,:,s1,1,1,0] = prob1*prob2
        M[s1,1,:,:,s1,1,1,0] = prob2*prob2
        M[s1+2,1,:,:,s1,1,1,0] = prob2*prob2
        M[s1,0,:,:,s1,0,1,0] = 1*prob2
        M[s1+2,0,:,:,s1,0,1,0] = 1*prob2
        M[s1+1,0,:,:,s1,0,1,0] = 1*prob1
    
    
    for s1 in range(2,noS):
        for s2 in range(noS-2):
            M[s1-1,s2+1,:,:,s1,s2,0,1] = prob1*prob1
            M[s1,s2+1,:,:,s1,s2,0,1] = prob2*prob1
            M[s1-1,s2,:,:,s1,s2,0,1] = prob1*prob2
            M[s1,s2,:,:,s1,s2,0,1] = prob2*prob2
            M[s1-2,s2+1,:,:,s1,s2,0,1] = prob1*prob2
            M[s1-1,s2+2,:,:,s1,s2,0,1] = prob1*prob2
            M[s1-2,s2+2,:,:,s1,s2,0,1] = prob2*prob2
            M[s1-2,s2,:,:,s1,s2,0,1] = prob2*prob2
            M[s1,s2+2,:,:,s1,s2,0,1] = prob2*prob2
    
    M[0,noS-1,:,:,1,noS-2,0,1] = prob1*prob1+prob2*prob2 + 2*prob1*prob2 
    M[1,noS-1,:,:,1,noS-2,0,1] = prob1*prob2
    M[0,noS-2,:,:,1,noS-2,0,1] = prob1*prob2
    M[1,noS-2,:,:,1,noS-2,0,1] = prob2*prob2
    M[0,noS-1,:,:,0,noS-1,0,1] = 1
    
    M[0,noS-1,:,:,1,noS-1,0,1] = prob1*prob1 + 3*prob2*prob1 + 2*prob2*prob2
    M[1,noS-1,:,:,1,noS-1,0,1] = prob2*prob1 +2*prob2*prob2
    M[0,noS-1,:,:,0,noS-2,0,1] = prob1*prob1 + 3*prob2*prob1 + 2*prob2*prob2
    M[0,noS-2,:,:,0,noS-2,0,1] = prob2*prob1 +2*prob2*prob2
    
    
    for s2 in range(noS-2):
        M[0,s2+1,:,:,1,s2,0,1] = (prob1+prob2)*prob1 
        M[0,s2,:,:,1,s2,0,1] = (prob1+prob2)*prob2
        M[0,s2+2,:,:,1,s2,0,1] = (prob1+prob2)*prob2
        M[1,s2+1,:,:,1,s2,0,1] = prob1*prob2
        M[1,s2,:,:,1,s2,0,1] = prob2*prob2
        M[1,s2+2,:,:,1,s2,0,1] = prob2*prob2
        M[0,s2,:,:,0,s2,0,1] = 1*prob2
        M[0,s2+2,:,:,0,s2,0,1] = 1*prob2
        M[0,s2+1,:,:,0,s2,0,1] = 1*prob1
    
    for s1 in range(2,noS):
        M[s1-1,noS-1,:,:,s1,noS-2,0,1] = (prob1+prob2)*prob1 
        M[s1,noS-1,:,:,s1,noS-2,0,1] = (prob1+prob2)*prob2
        M[s1-2,noS-1,:,:,s1,noS-2,0,1] = (prob1+prob2)*prob2
        M[s1-1,noS-2,:,:,s1,noS-2,0,1] = prob1*prob2
        M[s1,noS-2,:,:,s1,noS-2,0,1] = prob2*prob2
        M[s1-2,noS-2,:,:,s1,noS-2,0,1] = prob2*prob2
        M[s1,noS-1,:,:,s1,noS-1,0,1] = 1*prob2
        M[s1-2,noS-1,:,:,s1,noS-1,0,1] = 1*prob2
        M[s1-1,noS-1,:,:,s1,noS-1,0,1] = 1*prob1
    
    
    #now create reward matrix thingy
            
    for s1 in range(noS):
        for s2 in range(noS):
            r1=0
            r2 = 0
            if s2==((noS+1)/2 - 1):
                r2 = 1
            else: 
                r2 = -10
            if s1==((noS+1)/2 - 1):
                r1 = 1
            else:
                r1 = -10

            for t1 in range(len(rewardvector)):
                for t2 in range(len(rewardvector)):
                    if rewardvector[t1] != r1 or rewardvector[t2] != r2:
                        M[s1,s2,t1,t2,:,:,:,:] = 0
    
    
    return M
    
#this code creates the flattensor for states |-_{s}>
def createflatstate(T):
    # flatstate = np.asarray([])
    # for n in range(2*T+1):
    #     flatstate = np.append(flatstate,n-T)
    flatstate = np.ones(2*T+1)
    return flatstate

def createW(rewardmatrix):
    W = np.zeros((2,2,6,6))
    #create identity operator matrices and reward operator matrix.
   # for i in range(6):
    W[0,0,:,:] = 1
    W[1,1,:,:] = 1
    W[0,1,:,:] = rewardmatrix
    return W

def createW1(rewardmatrix):
    W1 = np.zeros((2,6,6))
    #create identity operator matrices and reward operator matrix.
   # for i in range(6):
    W1[0,:,:] = 1
    W1[1,:,:] = rewardmatrix
    return W1

def createWT(rewardmatrix):
    WT = np.zeros((2,6,6))
    #create identity operator matrices and reward operator matrix.
    #for i in range(6):
    WT[0,:,:] = rewardmatrix
    WT[1,:,:] = 1
    return WT

#create copytensor that makes 2 copies
def createcopytensor2(noS):
    copy = np.zeros((noS,noS,noS))
    for i in range(noS):
        copy[i,i,i] = 1
    return copy

def createcopytensor3(noS):
    copy = np.zeros((noS,noS,noS,noS))
    for i in range(noS):
        copy[i,i,i,i] = 1
    return copy
    
def createrewardmatrix(rewardvector):
    rewardmatrix = np.zeros((6,6))

    for i in range(6):
        for j in range(6):
            rewardmatrix[i,j] = rewardvector[i]+rewardvector[j]
    
    return rewardmatrix

def initializepolicy(T, noS):
    policy = np.ones((T,noS,noS,2,2))*(1/4)
    return policy

def choosedistribution(dist,sd):
    if dist =='normal':
        prob1 = scipy.stats.norm.cdf(1,0,sd) - scipy.stats.norm.cdf(-1,0,sd)
        prob2 = (1-prob1)/2
    elif dist == 'uniform':
        prob1 = 1/3
        prob2 = 1/3
    elif dist=='deterministic':
        prob1 = 1
        prob2 = 0 
        
    else:
        print('error')
    return prob1,prob2

def contractlayer1(M,p0,policy,W1):
    #(st1,st2,r1,r2,a1,a2)
    temp1 = np.tensordot(p0,M,((0,1),(4,5)))
    
    action = np.tensordot(policy[0],p0,((0,1),(0,1)))
    
    #st1,st2,r1,r2
    temp2 = np.tensordot(temp1,action,((4,5),(0,1)))
    
    #st1,st2,viout
    temp3 = np.tensordot(temp2,W1,((2,3),(1,2)))
    
    return temp3

def contractlayer(M,prevlayeroutput,policy,W,copy2,t):
    
    temp1 = np.tensordot(copy2,prevlayeroutput,((0),(1)))
    #st-11,st-11,st-12,st-12,viin
    temp1 = np.tensordot(copy2,temp1,((0),(2)))
    
    #st-11,st-12,viin,a1,a2
    temp1 = np.tensordot(temp1,policy[t],((0,2),(0,1)))
    
    
    #(st1,st2,st-11,st-12,a1,a2,viin,viout)
    layer = np.tensordot(M,W,((2,3),(2,3)))
    #(st1,st2,viout)
    temp2 = np.tensordot(temp1,layer,((0,1,2,3,4),(2,3,6,4,5)))
    
    return temp2

def contractlayerT(MT,prevlayeroutput,policy,WT,copy2,T,flatstate):
    
    temp1 = np.tensordot(copy2,prevlayeroutput,((0),(1)))
    #st-11,st-11,st-12,st-12,viin
    temp1 = np.tensordot(copy2,temp1,((0),(2)))
    
    #st-11,st-12,viin,a1,a2
    temp1 = np.tensordot(temp1,policy[T-1],((0,2),(0,1)))
    
    
    #(st1,st2,st-11,st-12,a1,a2,viin)
    layer = np.tensordot(MT,WT,((2,3),(1,2)))
    #(st1,st2)
    temp2 = np.tensordot(temp1,layer,((0,1,2,3,4),(2,3,6,4,5)))
    
    final = np.tensordot(temp2,flatstate,((0),(0)))
    final = np.tensordot(final,flatstate,((0),(0)))

    return final

def contracteverything(M,p0,policy,W1,W,copy2,MT,WT,flatstate,T):
    output = contractlayer1(M,p0,policy,W1)
    
    for t in range(1,T-1):
        output = contractlayer(M,output,policy,W,copy2,t)
        
    expectedreturn = contractlayerT(MT,output,policy,WT,copy2,T,flatstate)
    
    return expectedreturn
    
def DRMG(M,p0,policy,W1,W,copy2,MT,WT,flatstate,T,flatreward):
    #first optimise final layer.
    output = contractlayer1(M,p0,policy,W1)
    
    for t in range(1,T-1):
        output = contractlayer(M,output,policy,W,copy2,t)
        #(st1,st2,viout)
    
    temp1 = np.tensordot(copy2,output,((0),(1)))
    #st-11,st-11,st-12,st-12,viin
    temp1 = np.tensordot(copy2,temp1,((0),(2)))
    
    #(st1,st2,st-11,st-12,a1,a2,viin)
    layer = np.tensordot(MT,WT,((2,3),(1,2)))
    
    layer = np.tensordot(layer,flatstate,((0),(0)))
    #(st-11,st-12,a1,a2,viin)
    layer = np.tensordot(layer,flatstate,((0),(0)))
    
    #(st-11,st-12,a1,a2)
    temp2 = np.tensordot(temp1,layer,((0,2,4),(0,1,4)))

    policy[T-1]= changepolicy(policy[T-1], temp2, T)
    print('layer'+str(T-1))
    
    #now do second last layer

    output = contractlayer1(M,p0,policy,W1)
    
    for t in range(1,T-2):
        output = contractlayer(M,output,policy,W,copy2,t)
        #(st1,st2,viout)
    
    temp1 = np.tensordot(copy2,output,((0),(1)))
   
    #st-11,st-11,st-12,st-12,viin
    temp1 = np.tensordot(copy2,temp1,((0),(2)))
    
    #now make last layer
    #(st1,st2,st-11,st-12,a1,a2,viin)
    layer = np.tensordot(MT,WT,((2,3),(1,2)))
    
    layer = np.tensordot(layer,flatstate,((0),(0)))
    #(st-11,st-12,a1,a2,viin)
    layer = np.tensordot(layer,flatstate,((0),(0)))
    #(st-11,st-12,viin,st-11,st-12)
    layer = np.tensordot(layer,policy[T-1],((2,3),(2,3)))
    
    #(st-12,st-11,viin,st-11,)
    layer =np.tensordot(copy2,layer,((0,1),(1,4)))
    #(st-11,st-12,viin)
    layer =np.tensordot(copy2,layer,((0,1),(1,3)))
    
    #(st1,st2,st-11,st-12,a1,a2,viin,viout)
    optimiselayer = np.tensordot(M,W,((2,3),(2,3)))
    #(st-11,st-12,a1,a2,viin)
    optimiselayer = np.tensordot(optimiselayer,layer,((0,1,7),(0,1,2)))
    
    #(st-11,st-12,a1,a2)
    optimiselayer = np.tensordot(temp1,optimiselayer,((0,2,4),(0,1,4)))
    
    policy[T-2]= changepolicy(policy[T-2], optimiselayer, T)
    
    for x in range(2,T-1):
        #t = layer being optimised. 
        t = T-x
        print('layer'+str(t))
        output = contractlayer1(M,p0,policy,W1)
    
        for i in range(1,t-1):
            output = contractlayer(M,output,policy,W,copy2,i)
            #(st1,st2,viout)
        
        output2 = createlayer(flatreward,policy,M,W,copy2,MT,WT,flatstate,T,t)
        #(st-11,st-12,st1,st2,viin,viout)
        for i in range(t+1,T-1):
            output2 = contractafteroptimise(M,output2,policy,W,copy2,i)
        
        #skeep1,skeep2,vikeep
        output2 = contractToptimise(flatreward,policy,output2,copy2,MT,WT,flatstate,T)
        
        #### now contract the 2 outputs with the layer being optimised #####
        #(st1,st2,st-11,st-12,a1,a2,viin,viout)
        layer = np.tensordot(M,W,((2,3),(2,3)))
        
        #(st-11,st-12,a1,a2,viin)
        temp1 = np.tensordot(layer,output2,((0,1,7),(0,1,2)))
        
        temp2 = np.tensordot(copy2,output,((0),(1)))
        temp2 = np.tensordot(copy2,temp2,((0),(2)))
        #(st1,st1,st2,st2,viout)
        
        #(st1,st2,a1,a2)
        temp3 = np.tensordot(temp2,temp1,((0,2,4),(0,1,4)))
        policy[t-1]= changepolicy(policy[t-1], temp3, T)
        
    #finally optimise the first layer.
           
    output2 = createlayer(flatreward,policy,M,W,copy2,MT,WT,flatstate,T,t=1)
        #(st-11,st-12,st1,st2,viin,viout)
    for i in range(2,T-1):
            output2 = contractafteroptimise(M,output2,policy,W,copy2,i)
        
    #skeep1,skeep2,vikeep
    output2 = contractToptimise(flatreward,policy,output2,copy2,MT,WT,flatstate,T)
    
    #st1,st2,st-11,st-12,a1,a2,vie
    temp1 = np.tensordot(M,W1,((2,3),(1,2)))
    
    #st-11,st-12,a1,a2
    temp1 = np.tensordot(temp1,output2,((0,1,6),(0,1,2))) 
    
    initial = np.tensordot(copy2,p0,((0),(1)))
    initial = np.tensordot(initial,copy2,((2),(0))) 
    #st1,st1,st2,st2
    
    #st-11,st-12,a1,a2
    final = np.tensordot(initial,temp1,((0,2),(0,1)))

    policy[0]= changepolicy(policy[0], final, T)
    
    return policy

def contractafteroptimise(M,prevlayeroutput,policy,W,copy2,t):
    #output coming in with form (skeep1,skeep2,st1,st2,vikeep,viout)
    
    #(st1,st2,st-11,st-12,a1,a2,viin,viout)
    layer = np.tensordot(M,W,((2,3),(2,3)))

    #(st-12,st-12,st-11,a1,a2)
    temp1 = np.tensordot(copy2,policy[t],((0),(1)))
    #(st-11,st-11,st-12,st-12,a1,a2)
    temp1 = np.tensordot(copy2,temp1,((0),(1)))
    
    #(st-11,st-12,st1,st2,viin,viout)
    temp2 = np.tensordot(temp1,layer,((0,2,4,5),(2,3,4,5)))
    
    #skeep1,skeep2,vikeep,st1,st2,viout
    temp3 = np.tensordot(prevlayeroutput,temp2,((2,3,5),(0,1,4)))
    #skeep1,skeep2,st1,vikeep,st2,viout
    temp3 = np.swapaxes(temp3,2,3)
    #(skeep1,skeep2,st1,st2,vikeep,viout)
    temp3 = np.swapaxes(temp3,3,4)
    return temp3

def contractToptimise(flatreward,policy,prevlayeroutput,copy2,MT,WT,flatstate,T):
   #output coming in with form (skeep1,skeep2,st1,st2,vikeep,viout)
    
    #(st1,st2,st-11,st-12,a1,a2,viin)
    layer = np.tensordot(MT,WT,((2,3),(1,2)))
    
    layer = np.tensordot(layer,flatstate,((0),(0)))
    #(st-11,st-12,a1,a2,viin)
    layer = np.tensordot(layer,flatstate,((0),(0)))
    #(st-11,st-12,viin,st-11,st-12)
    layer = np.tensordot(layer,policy[T-1],((2,3),(2,3)))
    
    #(st-12,st-11,viin,st-11,)
    layer =np.tensordot(copy2,layer,((0,1),(1,4)))
    #(st-11,st-12,viin)
    layer =np.tensordot(copy2,layer,((0,1),(1,3)))
    
    temp1 = np.tensordot(layer,prevlayeroutput,((0,1,2),(2,3,5)))
    #skeep1,skeep2,vikeep
    return temp1    
    
    


def createlayer(flatreward,policy,M,W,copy2,MT,WT,flatstate,T,t):
    #(st1,st2,st-11,st-12,a1,a2,viin,viout)
    layer = np.tensordot(M,W,((2,3),(2,3)))

    #(st-12,st-12,st-11,a1,a2)
    temp1 = np.tensordot(copy2,policy[t],((0),(1)))
    #(st-11,st-11,st-12,st-12,a1,a2)
    temp1 = np.tensordot(copy2,temp1,((0),(2)))
    
    #(st-11,st-12,st1,st2,viin,viout)
    temp2 = np.tensordot(temp1,layer,((0,2,4,5),(2,3,4,5)))
    
    return temp2
    


def changepolicy(policy,contraction,T):
    for i in range(2*T+1):#
        for j in range(2*T+1):
             if len(np.unique(contraction[i,j])) !=1:
                 value = np.argmax(contraction[i,j])
                
                 if value == 0:
                            policy[i,j,0,0] = 1
                            policy[i,j,0,1] = 0
                            policy[i,j,1,0] = 0
                            policy[i,j,1,1] = 0
                            
                 elif value == 1:
                            policy[i,j,0,0] = 0
                            policy[i,j,0,1] = 1
                            policy[i,j,1,0] = 0
                            policy[i,j,1,1] = 0
                            
                 elif value == 2:
                            policy[i,j,0,0] = 0
                            policy[i,j,0,1] = 0
                            policy[i,j,1,0] = 1
                            policy[i,j,1,1] = 0
                            
                 elif value ==3: 
                            policy[i,j,0,0] = 0
                            policy[i,j,0,1] = 0
                            policy[i,j,1,0] = 0
                            policy[i,j,1,1] = 1
                        
    return policy


def plotgreedy(policy,T,p0,M,MT,copy2,noS):
   state = p0
   s1,s2 = np.where(state==1)
   shist1=np.zeros(T+1)
   shist1[0] = s1
   shist2=np.zeros(T+1)
   shist2[0] = s2
   
   for t in range(T-1):
       #get action
       actions = np.tensordot(state,policy[t],((0,1),(0,1)))

        #st1,st2,r1,r2,a1,a2
       temp1 = np.tensordot(state,M,((0,1),(4,5)))
       #st1,st2,r1,r2
       temp1 = np.tensordot(actions,temp1,((0,1),(4,5)))
       
       #st1,st2
       temp1 = np.tensordot(np.ones(6),temp1,((0),(2)))
       temp1 = np.tensordot(np.ones(6),temp1,((0),(2)))
              
       states = np.unravel_index(temp1.argmax(), temp1.shape)
       s1 = states[0]
       s2 = states[1]



       state = np.zeros((noS,noS))
       state[s1,s2] = 1
       
           
       shist1[t+1] = s1
       shist2[t+1] = s2

   actions = np.tensordot(state,policy[T-1],((0,1),(0,1)))

     #st1,st2,r1,r2,a1,a2
   temp1 = np.tensordot(state,MT,((0,1),(4,5)))
    #st1,st2,r1,r2
   temp1 = np.tensordot(actions,temp1,((0,1),(4,5)))
    
    #st1,st2
   temp1 = np.tensordot(np.ones(6),temp1,((0),(2)))
   temp1 = np.tensordot(np.ones(6),temp1,((0),(2)))
           
   states = np.unravel_index(temp1.argmax(), temp1.shape)
   s1 = states[0]
   s2 = states[1]

    
        
   shist1[T] = s1
   shist2[T] = s2
    
   plt.plot(range(T+1),shist1-T)
   plt.plot(range(T+1),shist2-T)
   plt.plot(range(-1,T+2),np.zeros(T+3),'r:')
   plt.xlim(0,T)
   plt.ylim(-T,T)
   plt.xlabel('Time')
   plt.ylabel('Position')
   plt.xticks(np.arange(0,T+1))
   plt.yticks(np.arange(-T,T))
   plt.legend(['Agent 1', 'Agent 2'])
   plt.grid()
   


if __name__ == '__main__': 
    #can choose value of T
    T = 6
    #number of states.
    noS = 2*T+1
    
    #create the flatstates, needed for contractions later. 
    flatreward = np.ones(6)
    flatstate = createflatstate(T)
    
    #initial probability vector has all zeros apart from at s=0. 
    p0 = np.zeros((noS,noS))
    p0[int((noS+1)/2-1),int((noS+1)/2-1)] = 1
    
    
    #initialize an random policy
    policy = initializepolicy(T, noS)

    copy2 = createcopytensor2(noS)

    
    rewardvector = np.asarray([1,0,-1,-2,-3,-10])
    
    prob1,prob2 = choosedistribution('normal', 1)
    M = createM(noS,prob1,prob2,rewardvector)
    MT = createMT(noS,prob1,prob2,rewardvector)
    rewardmatrix = createrewardmatrix(rewardvector)
    #W tensors are same for all timesteps apart from timesteps 1 and T
    W = createW(rewardmatrix)
    W1 = createW1(rewardmatrix)
    WT = createWT(rewardmatrix)    
    
    
    er = contracteverything(M, p0, policy, W1, W, copy2, MT, WT, flatstate,T)
    policy = DRMG(M,p0,policy,W1,W,copy2,MT,WT,flatstate,T,flatreward)
    er2 = contracteverything(M, p0, policy, W1, W, copy2, MT, WT, flatstate,T)
    plotgreedy(policy,T,p0,M,MT,copy2,noS)