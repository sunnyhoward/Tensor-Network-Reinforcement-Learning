# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 10:10:41 2020

@author: Sunny
"""



import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

#This implementation includes the noise when going up. When going down nothing
# has changed, but when going up 68% chance of going up 1, 16% of 2 or staying same

#function to create the initial policy
def initializepolicy(T,noS):
    #policy represents P(a_t|s^1_{t-1},s^2_{t-1})
    policy = np.ones((T,noS,noS,2)) * 0.5
    return policy

#plot path chosen by policy. 
def plotgreedy(policy,T,p0):
   state = p0
   s = np.where(state==1)[0]
   shist=np.zeros(T+1)
   shist[0] = s
   for t in range(T):
       #get action
       #action = np.argmax(np.tensordot(state,policy[t],((0),(0))))
       if np.tensordot(state,policy[t],((0),(0)))[1] > np.random.rand(1):
           action = 1
       else:
           action = 0
       state = np.zeros(2*T+1)
       
       if action == 1:
           state[s+1] = 1
       else:
           state[s-1] = 1
           
       s = np.where(state==1)[0]
       shist[t+1] = s
   plt.plot(range(T+1),shist-T)
   plt.plot(range(-1,T+2),np.zeros(T+3),'r:')
   plt.xlim(0,T)
   plt.ylim(-T,T)
   plt.xlabel('Time')
   plt.ylabel('Position')
   plt.xticks(np.arange(0,T+1))
   plt.yticks(np.arange(-T,T))
   plt.grid()
   
def generatetrajectory(policy,T,p0):
   state = p0
   s = np.where(state==1)[0]
   shist=np.zeros(T+1)
   shist[0] = s
   R = 0
   for t in range(T):
       #get action
       #action = np.argmax(np.tensordot(state,policy[t],((0),(0))))
       if np.tensordot(state,policy[t],((0),(0)))[1] > np.random.rand(1):
           action = 1
       else:
           action = 0
       state = np.zeros(2*T+1)
       
       if action == 1:
           rand = np.random.rand()
           if rand < 0.68:
               state[s+1] = 1
           elif rand>0.68 and rand<0.84:
               state[s+2] = 1
           elif rand > 0.84:
               state[s] = 1
       else:
           state[s-1] = 1
           
       s = np.where(state==1)[0]
       shist[t+1] = s
       if t!=T-1:
           if s[0]<(len(p0)+1)/2-1:
               R+=-1
       elif t ==T-1:
           if s[0]==(len(p0)+1)/2-1:
               R+=1
           else:
               R+=-10
           
   plt.plot(range(T+1),shist-T)
   plt.plot(range(-1,T+2),np.zeros(T+3),'r:')
   plt.xlim(0,T)
   plt.ylim(-T,T)
   plt.xlabel('Time')
   plt.ylabel('Position')
   plt.xticks(np.arange(0,T+1))
   plt.yticks(np.arange(-T,T))
   plt.grid()
   print('The Return was ' + str(R))
    

#function to create the rank 4 tensor. 
def createM1(noS,prob1,prob2):
    M = np.zeros((noS,noS,noS,2,6)) 
    
    # M[S_{t-1}^1,S_{t}^1,S_{t}^2,a^{1},r]
    #indices for start state, end state, actions and rewards.
    #possible values for action are down (0) and up (1), 
    #possible values for reward are 1, 0, -1, -2, -3, -10
    
    #note that this M tensor will not be the one at t = T, so that there is a 0 chance
    #of recieving a reward of -10 or 1 for all indices according to rules in section
    #5.1 of Ed's paper.
    
    #the tensor is filled using the update rules described in the paper.
    
    #the ((noS+1)/2)-1 th indicy signifys a position of s = 0.
    
    #first will update the M tensor so that if the initial state is 0 or below, and we take
    #an action to go down then there is a prob of 1 of getting reward  = -1 
    for x in range(2,int((noS+1)/2)):
        for y in range(noS):
            if y>=x:
                M[x,x,y,0,4] = prob2 #if agent 2 is bigger then penalty
            else:
                M[x,x,y,0,2] = prob2
            
            if y>=x-1:
                M[x,x-1,y,0,4] = prob1
            else:
                M[x,x-1,y,0,2] = prob1

            if y>=x-2:
                M[x,x-2,y,0,4] = prob2
            else:
                M[x,x-2,y,0,2] = prob2
                
    
    M[int((noS+1)/2)-1,int((noS+1)/2)-1,:,0,2] = 0 #correct 0
    M[int((noS+1)/2)-1,int((noS+1)/2)-1,:,0,4] = 0 #correct 0
    for y in range(noS):
        if y>=int((noS+1)/2)-1:
            M[int((noS+1)/2)-1,int((noS+1)/2)-1,y,0,3] = prob2  
        else:
            M[int((noS+1)/2)-1,int((noS+1)/2)-1,y,0,1] = prob2 
    
    M[1,0,:,0,4]=prob1+prob2 #if jump down from second last

    M[1,1,0,0,2]=prob2
    M[1,1,1:,0,4]=prob2

    M[0,0,:,0,4] = 1 
        
    #similarly, if we go up from anything below -2, we also will recieve a negative reward.
    for x in range(0,int((noS+1)/2-3)):
        for y in range(noS):
            if y>=x+2:
                M[x,x+2,y,1,4] = prob2
            else:
                M[x,x+2,y,1,2] = prob2
                
            if y>=x+1:
                M[x,x+1,y,1,4] = prob1
            else:
                M[x,x+1,y,1,2] = prob1
                
            if y>=x:
                M[x,x,y,1,4] = prob2
            else:
                M[x,x,y,1,2] = prob2

        
    #if in state -2, if we go up 2 we get reward = 0
    M[int((noS+1)/2 -3),int((noS+1)/2 -2),0:int((noS+1)/2 -2) ,1,2]=prob1
    M[int((noS+1)/2 -3),int((noS+1)/2 -2),int((noS+1)/2 -2):,1,4]=prob1

    M[int((noS+1)/2 -3),int((noS+1)/2 -3),0:int((noS+1)/2 -3) ,1,2]=prob2
    M[int((noS+1)/2 -3),int((noS+1)/2 -3),int((noS+1)/2 -3): ,1,4]=prob2

    
    M[int((noS+1)/2 -3),int((noS+1)/2 -1),0:int((noS+1)/2-1),1,1]=prob2
    M[int((noS+1)/2 -3),int((noS+1)/2 -1),int((noS+1)/2-1):,1,3]=prob2
    


    #now update that when 0 or above, going up will give reward of 0 
    for x in range(int((noS+1)/2-1),noS-2):
        for y in range(noS):
            if y>=x+2:
                M[x,x+2,y,1,3] = prob2
            else:
                M[x,x+2,y,1,1] = prob2
            if y>=x+1:  
                M[x,x+1,y,1,3] = prob1
            else:
                M[x,x+1,y,1,1] = prob1
            if y>=x:
                M[x,x,y,1,3] = prob2
            else:
                M[x,x,y,1,1] = prob2
    
    
    
    #if instate -1, going up 1 or 2 gives 0 reward, staying same gives -1 reward
    
    M[int((noS+1)/2 -2),int((noS+1)/2-1),0:int((noS+1)/2-1),1,1]=prob1
    M[int((noS+1)/2 -2),int((noS+1)/2-1),int((noS+1)/2-1):,1,3]=prob1

    M[int((noS+1)/2 -2),int((noS+1)/2),0:int((noS+1)/2) ,1,1]=prob2
    M[int((noS+1)/2 -2),int((noS+1)/2),int((noS+1)/2): ,1,3]=prob2

    M[int((noS+1)/2 -2),int((noS+1)/2 -2),0:int((noS+1)/2 -2),1,2]=prob2  
    M[int((noS+1)/2 -2),int((noS+1)/2 -2),int((noS+1)/2 -2):,1,4]=prob2  


    
    #now update that when above 0, going down will give reward of 0 
    for x in range(int((noS+1)/2),noS):
        for y in range(noS):
            if y>=x:
                M[x,x,y,0,3] = prob2
            else:
                M[x,x,y,0,1] = prob2
                
            if y>=x-1:
                M[x,x-1,y,0,3] = prob1
            else:
                M[x,x-1,y,0,1] = prob1
                
            if y>=x-2:
                M[x,x-2,y,0,3] = prob2
            else:
                M[x,x-2,y,0,1] = prob2

            
        
    M[int((noS+1)/2),int((noS+1)/2)-2,:,0,1] = 0
    M[int((noS+1)/2),int((noS+1)/2)-2,:,0,3] = 0

    
    M[int((noS+1)/2),int((noS+1)/2)-2,0:(int((noS+1)/2)-2),0,2] = prob2
    M[int((noS+1)/2),int((noS+1)/2)-2,(int((noS+1)/2)-2):,0,4] = prob2

    
    #Finally lets ammend the top 2 states so that we cant jump out of cell.
    M[int(noS-2),:,:,:,:] = 0
    M[int(noS-1),:,:,:,:] = 0
    
    
    M[int(noS-2),int(noS-2),0:int(noS-2),1,1] = prob2 #if jump up from 2nd top cell then stay same
    M[int(noS-2),int(noS-2),int(noS-2):,1,3] = prob2 #if jump up from 2nd top cell then stay same

    
    M[int(noS-2),int(noS-1),0:int(noS-1),1,1] = prob1+prob2 #if jump up from 2nd top cell then go up 1
    M[int(noS-2),int(noS-1),int(noS-1),1,3] = prob1+prob2 #if jump up from 2nd top cell then go up 1

    
    M[int(noS-2),int(noS-3),0:int(noS-3),0,1] = prob1 #if go down from 2nd top cell then go down 
    M[int(noS-2),int(noS-3),int(noS-3):,0,3] = prob1 #if go down from 2nd top cell then go down 

    
    M[int(noS-2),int(noS-2),0:int(noS-2),0,1] = prob2 #if go down from 2nd top cell then go down 
    M[int(noS-2),int(noS-2),int(noS-2):,0,3] = prob2 #if go down from 2nd top cell then go down 

    
    M[int(noS-2),int(noS-4),0:int(noS-4),0,1] = prob2 #if go down from 2nd top cell then go down 
    M[int(noS-2),int(noS-4),int(noS-4):,0,3] = prob2 #if go down from 2nd top cell then go down 


    M[int(noS-1),int(noS-1),0:int(noS-1),1,1] = 1 #if jump up from top cell then stay same
    M[int(noS-1),int(noS-1),int(noS-1),1,3] = 1 #if jump up from top cell then stay same


    M[int(noS-1),int(noS-2),0:int(noS-2),0,1] = prob1 #if jump down from  top cell then go down
    M[int(noS-1),int(noS-2),int(noS-2):,0,3] = prob1 #if jump down from  top cell then go down

    
    M[int(noS-1),int(noS-3),0:int(noS-3),0,1] = prob2 #if jump down from  top cell then go 
    M[int(noS-1),int(noS-3),int(noS-3):,0,3] = prob2 #if jump down from  top cell then go 

    
    M[int(noS-1),int(noS-1),0:int(noS-1),0,1] = prob2 #if jump down from  top cell then go down
    M[int(noS-1),int(noS-1),int(noS-1),0,3] = prob2 #if jump down from  top cell then go down

    return M


#function to create the rank 4 tensor. 
def createM2(noS,prob1,prob2):
    M = np.zeros((noS,noS,noS,2,6)) 
    
    # M[S_{t-1}^2,S_{t}^2,S_{t}^1,a^{2},r_{2}]
    #indices for start state, end state, actions and rewards.
    #possible values for action are down (0) and up (1), 
    #possible values for reward are 1, 0, -1, -2, -3, -10
    
    #note that this M tensor will not be the one at t = T, so that there is a 0 chance
    #of recieving a reward of -10 or 1 for all indices according to rules in section
    #5.1 of Ed's paper.
    
    #the tensor is filled using the update rules described in the paper.
    
    #the ((noS+1)/2)-1 th indicy signifys a position of s = 0.
    
    #first will update the M tensor so that if the initial state is 0 or below, and we take
    #an action to go down then there is a prob of 1 of getting reward  = -1 
    for x in range(2,int((noS+1)/2)):
        for y in range(noS):
            if x>=y:
                M[x,x,y,0,4] = prob2 #if agent 2 is bigger then penalty
            else:
                M[x,x,y,0,2] = prob2
            
            if x-1>=y:
                M[x,x-1,y,0,4] = prob1
            else:
                M[x,x-1,y,0,2] = prob1

            if x-2>=y:
                M[x,x-2,y,0,4] = prob2
            else:
                M[x,x-2,y,0,2] = prob2
                
    
    M[int((noS+1)/2)-1,int((noS+1)/2)-1,:,0,2] = 0 #correct 0
    M[int((noS+1)/2)-1,int((noS+1)/2)-1,:,0,4] = 0 #correct 0
    
    for y in range(noS):
        if int((noS+1)/2)-1>=y:
            M[int((noS+1)/2)-1,int((noS+1)/2)-1,y,0,3] = prob2  
        else:
            M[int((noS+1)/2)-1,int((noS+1)/2)-1,y,0,1] = prob2 
    
    
    M[1,0,1:,0,2]=prob1+prob2 #if jump down from second last
    M[1,0,0,0,4]=prob1+prob2 #if jump down from second last

    M[1,1,0:2,0,4]=prob2
    M[1,1,2:,0,2]=prob2

    M[0,0,1:,0,2] = 1 
    M[0,0,0,0,4] = 1 

        
    #similarly, if we go up from anything below -2, we also will recieve a negative reward.
    for x in range(0,int((noS+1)/2-3)):
        for y in range(noS):
            if x+2>=y:
                M[x,x+2,y,1,4] = prob2
            else:
                M[x,x+2,y,1,2] = prob2
                
            if x+1>=y:
                M[x,x+1,y,1,4] = prob1
            else:
                M[x,x+1,y,1,2] = prob1
                
            if x>=y:
                M[x,x,y,1,4] = prob2
            else:
                M[x,x,y,1,2] = prob2

        
    #if in state -2, if we go up 2 we get reward = 0
    M[int((noS+1)/2 -3),int((noS+1)/2 -2),0:int((noS+1)/2 -1) ,1,4]=prob1
    M[int((noS+1)/2 -3),int((noS+1)/2 -2),int((noS+1)/2 -1):,1,2]=prob1

    M[int((noS+1)/2 -3),int((noS+1)/2 -3),0:int((noS+1)/2 -2) ,1,4]=prob2
    M[int((noS+1)/2 -3),int((noS+1)/2 -3),int((noS+1)/2 -2): ,1,2]=prob2

    
    M[int((noS+1)/2 -3),int((noS+1)/2 -1),0:int((noS+1)/2),1,3]=prob2
    M[int((noS+1)/2 -3),int((noS+1)/2 -1),int((noS+1)/2):,1,1]=prob2
    


    #now update that when 0 or above, going up will give reward of 0 
    for x in range(int((noS+1)/2-1),noS-2):
        for y in range(noS):
            if x+2>=y:
                M[x,x+2,y,1,3] = prob2
            else:
                M[x,x+2,y,1,1] = prob2
            if x+1>=y:  
                M[x,x+1,y,1,3] = prob1
            else:
                M[x,x+1,y,1,1] = prob1
            if x>=y:
                M[x,x,y,1,3] = prob2
            else:
                M[x,x,y,1,1] = prob2
    
    
    
    #if instate -1, going up 1 or 2 gives 0 reward, staying same gives -1 reward
    
    M[int((noS+1)/2 -2),int((noS+1)/2-1),0:int((noS+1)/2),1,3]=prob1
    M[int((noS+1)/2 -2),int((noS+1)/2-1),int((noS+1)/2):,1,1]=prob1

    M[int((noS+1)/2 -2),int((noS+1)/2),0:(int((noS+1)/2)+1) ,1,3]=prob2
    M[int((noS+1)/2 -2),int((noS+1)/2),(int((noS+1)/2)+1): ,1,1]=prob2

    M[int((noS+1)/2 -2),int((noS+1)/2 -2),0:int((noS+1)/2 -1),1,4]=prob2  
    M[int((noS+1)/2 -2),int((noS+1)/2 -2),int((noS+1)/2 -1):,1,2]=prob2  


    
    #now update that when above 0, going down will give reward of 0 
    for x in range(int((noS+1)/2),noS):
        for y in range(noS):
            if x>=y:
                M[x,x,y,0,3] = prob2
            else:
                M[x,x,y,0,1] = prob2
                
            if x-1>=y:
                M[x,x-1,y,0,3] = prob1
            else:
                M[x,x-1,y,0,1] = prob1
                
            if x-2>=y:
                M[x,x-2,y,0,3] = prob2
            else:
                M[x,x-2,y,0,1] = prob2

            
        
    M[int((noS+1)/2),int((noS+1)/2)-2,:,0,1] = 0
    M[int((noS+1)/2),int((noS+1)/2)-2,:,0,3] = 0

    
    M[int((noS+1)/2),int((noS+1)/2)-2,0:(int((noS+1)/2)-1),0,4] = prob2
    M[int((noS+1)/2),int((noS+1)/2)-2,(int((noS+1)/2)-1):,0,2] = prob2

    
    #Finally lets ammend the top 2 states so that we cant jump out of cell.
    M[int(noS-2),:,:,:,:] = 0
    M[int(noS-1),:,:,:,:] = 0
    
    
    M[int(noS-2),int(noS-2),0:int(noS-1),1,3] = prob2 #if jump up from 2nd top cell then stay same
    M[int(noS-2),int(noS-2),int(noS-1),1,1] = prob2 #if jump up from 2nd top cell then stay same

    
    M[int(noS-2),int(noS-1),:,1,3] = prob1+prob2 #if jump up from 2nd top cell then go up 1
    
    M[int(noS-2),int(noS-3),0:int(noS-2),0,3] = prob1 #if go down from 2nd top cell then go down 
    M[int(noS-2),int(noS-3),int(noS-2):,0,1] = prob1 #if go down from 2nd top cell then go down 

    
    M[int(noS-2),int(noS-2),0:int(noS-1),0,3] = prob2 #if go down from 2nd top cell then go down 
    M[int(noS-2),int(noS-2),int(noS-1),0,1] = prob2 #if go down from 2nd top cell then go down 

    
    M[int(noS-2),int(noS-4),0:int(noS-3),0,3] = prob2 #if go down from 2nd top cell then go down 
    M[int(noS-2),int(noS-4),int(noS-3):,0,1] = prob2 #if go down from 2nd top cell then go down 


    M[int(noS-1),int(noS-1),:,1,3] = 1 #if jump up from top cell then stay same

    M[int(noS-1),int(noS-2),0:int(noS-1),0,3] = prob1 #if jump down from  top cell then go down
    M[int(noS-1),int(noS-2),int(noS-1),0,1] = prob1 #if jump down from  top cell then go down

    
    M[int(noS-1),int(noS-3),0:int(noS-2),0,3] = prob2 #if jump down from  top cell then go 
    M[int(noS-1),int(noS-3),int(noS-2):,0,1] = prob2 #if jump down from  top cell then go 

    
    M[int(noS-1),int(noS-1),:,0,3] = prob2 #if jump down from  top cell then go down

    return M
    
    
#function to create the end of M tensor for the last time step. 
def createMT(noS,prob1,prob2):
    MT = np.zeros((noS,noS,2,6)) 
    
    #in this case, we simply get a -10 reward for not being at position 0, at T
    #and a 1 reward for being there. 
    
    # M[St-1,St,a,r]
    #possible values for reward are 1, 0, -1,, -2 ,-3 -10
    
    for x in range(int((noS+1)/2)+1,noS):
        MT[x,x-1,0,5] = prob1
        MT[x,x,0,5] = prob2
        MT[x,x-2,0,5] = prob2
    
    MT[int((noS+1)/2)+1,int((noS+1)/2)-1,0,5] = 0
    MT[int((noS+1)/2)+1,int((noS+1)/2)-1,0,0] = prob2
    MT[int((noS+1)/2),int((noS+1)/2)-1,0,0] = prob1
    MT[int((noS+1)/2),int((noS+1)/2)-2,0,5] = prob2
    MT[int((noS+1)/2),int((noS+1)/2),0,5] = prob2
     
    for x in range(2,int((noS+1)/2)):
        MT[x,x-1,0,5] = prob1
        MT[x,x,0,5] = prob2
        MT[x,x-2,0,5] = prob2
    
    MT[int((noS+1)/2)-1,int((noS+1)/2)-1,0,5] = 0
    MT[int((noS+1)/2)-1,int((noS+1)/2)-1,0,0] = prob2
    
    MT[1,0,0,5] = prob1+prob2
    MT[1,1,0,5] = prob2
    MT[0,0,0,5] = 1 
        
    #similarly, if we go up from 1 or above we will recieve -10 reward.
    for x in range(int((noS+1)/2),noS-2):
        MT[x,x+1,1,5] = prob1
        MT[x,x+2,1,5] = prob2
        MT[x,x,1,5] = prob2
    
    #going up from zero can score you either +1 or -10
    MT[int((noS+1)/2)-1,int((noS+1)/2)-1,1,0] = prob2
    MT[int((noS+1)/2)-1,int((noS+1)/2),1,5] = prob1
    MT[int((noS+1)/2)-1,int((noS+1)/2)+1,1,5] = prob2
    
    #If we go up or down from highest state we score -10
    MT[int(noS-1),int(noS-1),1,5] = 1
    
    #if we go up from second highest state we score -10
    MT[int(noS-2),int(noS-2),1,5] = prob2
    MT[int(noS-2),int(noS-1),1,5] = prob1+prob2

    
    #if we go up from a state between the min and -3 we will recieve -10 reward.
    for x in range(0,int((noS+1)/2-3)):
        MT[x,x+1,1,5] = prob1
        MT[x,x+2,1,5] = prob2
        MT[x,x,1,5] = prob2
        
    #going up from -2 can yield 2 possibiliies
    MT[int((noS+1)/2-3),int((noS+1)/2-1),1,0] = prob2
    MT[int((noS+1)/2-3),int((noS+1)/2-2),1,5] = prob1
    MT[int((noS+1)/2-3),int((noS+1)/2-3),1,5] = prob2

    
    

    
    #if we go up from -1 then a few things can happen
    MT[int((noS+1)/2-2),int((noS+1)/2-1),1,0] = prob1
    MT[int((noS+1)/2-2),int((noS+1)/2-2),1,5] = prob2
    MT[int((noS+1)/2-2),int((noS+1)/2),1,5] = prob2

    
    return MT
        
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

def createcopytensor(noS):
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

def contractlayer1only(flatreward,policy1,policy2,M1,M2,W1,p10,p20):
    #contracting initial prob dist with M1
    #gives something of (endstate1,endstate2,action1,reward1)
    contraction1 = np.tensordot(p10,M1,((0),(0)))
    
    #gives something of (endstate2,endstate1,action2,reward2)
    contraction2=np.tensordot(p20,M2,((0),(0)))
    
    #find the actions chosen by contracting initial prob with policy pi0
    actionstemp1 = np.tensordot(p10,policy1[0],((0),(0)))
    actions1 = np.tensordot(p20,actionstemp1,((0),(0)))
    
    actionstemp2 = np.tensordot(p20,policy2[0],((0),(0)))
    actions2 = np.tensordot(p10,actionstemp2,((0),(0)))
   
    #shape: endstate1,endstate2,reward1
    contraction11 = np.tensordot(contraction1,actions1,((2),(0)))
    contraction22 = np.tensordot(contraction2,actions2,((2),(0)))
    
    
    #need to make a copy of end state.
    copytensor = np.zeros((noS,noS,noS))
    for i in range(noS):
        copytensor[i,i,i]=1
    
    #gives shape(endstate1,endstate1,endstate2,reward1)
    contraction111 = np.tensordot(copytensor,contraction11,((0),(0)))
    contraction222 = np.tensordot(copytensor,contraction22, ((0),(0)))
    
    #final shape = (endstate1,virutalindex1,endstate2,virtualindex2)
    temp1 = np.tensordot(contraction111,contraction222,((0,2),(2,0)))
    temp2 =np.tensordot(temp1,flatstate,((0),(0)))
    rewardprobabilities =np.tensordot(temp2,flatstate,((1),(0)))
    
    
    rewardmatrix = np.zeros((6,6))

    for i in range(6):
        for j in range(6):
            rewardmatrix[i,j] = rewardvector[i]+rewardvector[j]
    
    expectedreturn = np.tensordot(rewardprobabilities,rewardmatrix,((0,1),(0,1)))
    return expectedreturn


def contractlayer123only(flatreward,policy1,policy2,M1,M2,W1,p10,p20):
    #contracting initial prob dist with M1
    #gives something of (endstate1,endstate2,action1,reward1)
    contraction1 = np.tensordot(p10,M1,((0),(0)))
    
    #gives something of (endstate2,endstate1,action2,reward2)
    contraction2=np.tensordot(p20,M2,((0),(0)))
    
    #find the actions chosen by contracting initial prob with policy pi0
    actionstemp1 = np.tensordot(p10,policy1[0],((0),(0)))
    actions1 = np.tensordot(p20,actionstemp1,((0),(0)))
    
    actionstemp2 = np.tensordot(p20,policy2[0],((0),(0)))
    actions2 = np.tensordot(p10,actionstemp2,((0),(0)))
   
    #shape: endstate1,endstate2,reward1
    contraction11 = np.tensordot(contraction1,actions1,((2),(0)))
    contraction22 = np.tensordot(contraction2,actions2,((2),(0)))
    
    
    #need to make a copy of end state.
    copytensor = np.zeros((noS,noS,noS))
    for i in range(noS):
        copytensor[i,i,i]=1
    
    #gives shape(endstate1,endstate1,endstate2,reward1)
    contraction111 = np.tensordot(copytensor,contraction11,((0),(0)))
    contraction222 = np.tensordot(copytensor,contraction22, ((0),(0)))
    
    #final shape = (endstate1,reward1,endstate2,reward2)
    temp1 = np.tensordot(contraction111,contraction222,((0,2),(2,0)))
    
    #endstate1,endstate2,vie
    layer1 = np.tensordot(temp1,W1,((1,3),(1,2)))
    
    #layer1  done
    
    #bs1,bs1,bs1,bs2,vie
    temp3 = np.tensordot(copy,layer1,((0),(0)))
    
    #bs1,bs1,bs1,vib,bs2,bs2,bs2
    temp4 = np.tensordot(temp3,copy,((3),(0)))
    
    #bstate1,bs1,vie,bs2,bs2,action1
    temp5 = np.tensordot(temp4,policy1[1],((0,4),(0,1)))
    #endstate1,vie,es2,action1,action2
    temp6 = np.tensordot(temp5,policy2[1],((3,0),(0,1)))
    
    
    #es1,es1,bs1,es2,a1,r1
    tempo = np.tensordot(copytensor,M1,((1),(0)))
    #es2,es2,bs2,es1,a2,r2    
    tempo2 = np.tensordot(copytensor,M2,((1),(0)))
    #es1,bs1,a1,r1,es2,bs2,a2,r2
    harvest = np.tensordot(tempo,tempo2,((0,3),(3,0)))
    
    #vie,es1,r1,es2,r2
    combine = np.tensordot(temp6,harvest,((0,2,3,4),(1,5,2,6)))
    #es1,es2,vib
    layer2 = np.tensordot(combine,W,((0,2,4),(0,2,3)))
    
    
    
    #endstate1,es1,es1,es2,vie
    new3 = np.tensordot(copy,layer2,((0),(0)))
    
    #endstate1,es1,es1,vie,es2,es2,es2
    new4 = np.tensordot(new3,copy,((3),(0)))
    
    #endstate1,es1,vie,es2,es2,action1
    new5 = np.tensordot(new4,policy1[2],((0,4),(0,1)))
    #endstate1,vie,es2,action1,action2
    new6 = np.tensordot(new5,policy2[2],((3,0),(0,1)))
    
    
    #es1,es1,bs1,es2,a1,r1
    ntempo = np.tensordot(copytensor,M1,((1),(0)))
    #es1,bs1,es2,a1,r1
    ntempo1 = np.tensordot(ntempo,flatstate,((0),(0)))
    
    ntempo2 = np.tensordot(copytensor,M2,((1),(0)))
    #es2,bs2,es1,a2,r2
    ntempo22 = np.tensordot(ntempo2,flatstate,((0),(0)))
    #bs1,a1,r1,bs2,a2,r2
    nharvest = np.tensordot(ntempo1,ntempo22,((0,2),(2,0)))
    
    #vie,r1,r2
    ncombine = np.tensordot(new6,nharvest,((0,2,3,4),(0,3,1,4)))
    
    expectedreturn = np.tensordot(ncombine,WT,((0,1,2),(0,1,2)))
    
    test1 = np.tensordot(ntempo,ntempo2,((0,3),(3,0)))

    test2 = np.tensordot(new6,test1,((0,2,3,4),(1,5,2,6)))

    test3 = np.tensordot(test2,WT,((0,2,4),(0,1,2)))
    
    return expectedreturn

def contractlayer1(flatreward,policy1,policy2,M1,M2,W1,p10,p20):
    #contracting initial prob dist with M1
    #gives something of (endstate1,endstate2,action1,reward1)
    contraction1 = np.tensordot(p10,M1,((0),(0)))
    
    #gives something of (endstate2,endstate1,action2,reward2)
    contraction2=np.tensordot(p20,M2,((0),(0)))
    
    #find the actions chosen by contracting initial prob with policy pi0
    actionstemp1 = np.tensordot(p10,policy1[0],((0),(0)))
    actions1 = np.tensordot(p20,actionstemp1,((0),(0)))
    
    actionstemp2 = np.tensordot(p20,policy2[0],((0),(0)))
    actions2 = np.tensordot(p20,actionstemp2,((0),(0)))
   
    #shape: endstate1,endstate2,reward1
    contraction11 = np.tensordot(contraction1,actions1,((2),(0)))
    contraction22 = np.tensordot(contraction2,actions2,((2),(0)))
    
    # #contracting W1 
    # #gives shape (next state,virtualindex,flatreward)
    # contraction111 = np.tensordot(contraction11,W1,((2),(1)))
    
    # contraction222 = np.tensordot(contraction22,W1,((2),(1)))
    
    # #contract with flat reward, gives shape (endstate1,endstate2,imagindex)

    # contraction1111 = np.tensordot(contraction111, flatreward,((3),(0)))
    # contraction2222 = np.tensordot(contraction222, flatreward,((3),(0)))
    #need to make a copy of end state.
    copytensor = np.zeros((noS,noS,noS))
    for i in range(noS):
        copytensor[i,i,i]=1
    
    #gives shape(endstate1,endstate1,endstate2,reward1)
    contraction11111 = np.tensordot(copytensor,contraction11,((0),(0)))
    #gives shape(endstate2,endstate2,endstate1,reward2)
    contraction22222 = np.tensordot(copytensor,contraction22, ((0),(0)))
    
    #final shape = (endstate1,reward1,endstate2,reward2)
    joint = np.tensordot(contraction11111,contraction22222,((0,2),(2,0)))

    #shape = (endstate1,endstate2,vi)
    finalcontraction = np.tensordot(joint,W1,((1,3),(1,2)))

    return finalcontraction
    #now the first layer is fully contracted. repeat this process for every layer but the last.
    
    

#function to manage the contracts betweenlayers, 
#such as policy and virtual index. This wont work if at layers 1 or end. 
def contractlayer(previouslayeroutput,policy1,policy2,copy,flatreward,W,M1,M2):
    
    #first contract M with W to give object of type
    #(statet-1{1},statet{1},statet{2},at-1{1},virtualbegin1,virtualend1,reward2)
    contraction1 = np.tensordot(M1,W,((4),(2)))

    copytensor = np.zeros((noS,noS,noS))
    for i in range(noS):
        copytensor[i,i,i]=1 
        
#(statet{1},statet{1},statet-1{1},statet{2},at-1{1},virtualbegin1,virtualend1,reward2)
    layer1 = np.tensordot(copytensor,contraction1,((0),(1)))
    #statet-1{2},statet{1},at-1{2},reward{2},statet{2},statet{2}
    layer2 = np.tensordot(M2,copytensor,((1),(0)))

    #(statet{1},statet-1{1},at-1{1},vib,vie,statet-1{2},at-1{2},statet{2})
    connect = np.tensordot(layer1,layer2,((0,3,7),(1,4,3)))
    

    #previouslayer output is shape (endstate1,endstate2,vib)
    #use copytensor to get(endstate1,endstate1,endstate1,vib,endstate2,es2,es2)
    temp1 = np.tensordot(copy,previouslayeroutput,((0),(0)))
    temp2 = np.tensordot(temp1,copy,((3),(0)))

    
    
    #contract with policy to get new action.
    #resulting object is (endstate1,endstate1,vib,es2,es2,action1)
    temp3 = np.tensordot(temp2,policy1,((0,4),(0,1)))
    
    #resulting object is (endstate1,vib,es2,action1,action2)
    temp4 = np.tensordot(temp3,policy2,((0,3),(1,0)))
 
    #now contract total layer
    #finalshape (statet{1},vie,statet{2})
    newoutput = np.tensordot(temp4,connect,((0,1,2,3,4),(1,3,5,2,6)))
    
    #(statet{1},statet{2},vie)
    newoutput = np.moveaxis(newoutput,2,1)

    
    return newoutput

#contracting last layer with the rest.
def contractlayerT(MT,WT,flatstate,copy,previouslayeroutput,policy1,policy2,flatreward):
    
    #gives shape(input state,action,reward)
    contraction1 = np.tensordot(MT,flatstate,((1),(0)))
    

    
    #gives shape(input state,action,reward)
    contraction2 = np.tensordot(MT,flatstate,((1),(0)))
    
    
    #makeshape (statet-1_{1},statet-1_{1},statet-1{1},statet-1{2}
    #vie)    
    contraction4 = np.tensordot(copy,previouslayeroutput,((0),(0)))
    
    #makeshape (statet-1_{1},statet-1_{1},statet-1{1},
    #vie,statet-1_{2},statet-1_{2},statet-1{2})    
    contraction5 = np.tensordot(contraction4,copy,((3),(0)))

    #makeshape (statet-1_{1},statet-1{1},vie,statet-1_{2},statet-1{2},action1)    
    contraction6 = np.tensordot(contraction5,policy1,((0,4),(0,1)))

    #makeshape (statet-1_{1},vie,statet-1_{2},action1,action2)    
    contraction7 = np.tensordot(contraction6,policy2,((3,0),(0,1)))

    #(reward1, vie,state2,action2)
    temp1 = np.tensordot(contraction1,contraction7,((0,1),(0,3)))
    
    #(reward1,vie,reward2)
    temp2 = np.tensordot(temp1,contraction2,((2,3),(0,1))) 
    #do the final contraction to obtain expected reward.
    
    
    expectedreturn = np.tensordot(temp2,WT,((1,0,2),(0,1,2)))
    return expectedreturn

#function to contract all layers to give the expected return. 
def contracteverything(M1,M2,MT,W,W1,WT,flatreward,policy1,policy2,p10,p20,flatstate,T,copy,plot):
    
    #first contract layer 1. 
    output = contractlayer1(flatreward,policy1,policy2,M1,M2,W1,p10,p20)

    #now contract all middle layers. 
    for i in range(1,T-1):
        output = contractlayer(output,policy1[i],policy2[i],copy,flatreward,W,M1,M2)
    
    #now contract final layer
    
    expectedreturn = contractlayerT(MT,WT,flatstate,copy,output,policy1[T-1],policy2[T-1],flatreward)
    
    #print(expectedreturn)
    #if plot==True:
    #    plotgreedy(policy,T,p0)
    return expectedreturn


#function to change the values in policy array, used in function below.
def changepolicy(policy,contraction,T):
    #contraction is of form state1,state2,action1
    for i in range(2*T+1):
        for j in range(2*T+1):
            if float(contraction[i,j,0] - contraction[i,j,1]) > 0:
                policy[i,j,0] = 1
                policy[i,j,1] = 0
            elif float(contraction[i,j,0] - contraction[i,j,1]) < 0:
                policy[i,j,0] = 0
                policy[i,j,1] = 1
    return policy
        
#it is easier to also have a function to contract all stuff after the layer being
#optimised.
def contractafteroptimiselayer(output,M1,M2,W,flatreward,copy,policy1,policy2):
    
    #output = (st-1{1},st-1{2},vib,policya{1},policya{2},policystate{1},policystate{1},policystate{2},policystate{2})
 
    #Now create the layer that we wish to optimise.
    #(st-1{1},st{1},s_{2},a{1},vib,vie,reward2)
    contraction1 = np.tensordot(M1,W,((4),(2)))
    copytensor = np.zeros((noS,noS,noS))
    for i in range(noS):
        copytensor[i,i,i] = 1
    
    #(st{1},st{1},st-1{1},s_{2},a{1},vib,vie,reward2)
    contraction2 = np.tensordot(copytensor,contraction1,((0),(1)))
    
    #(st-1{2},st{1},a{2},r{2},st{2},st{2})
    contraction3 = np.tensordot(M2,copytensor,((1),(0)))
    
    #(st{1},st-1{1},a{1},vib,vie,st-1{2},a{2},st{2})
    contraction4 = np.tensordot(contraction2,contraction3,((0,3,7),(1,4,3)))
    
    #now can contract the output from chosen layer with copy. this 
    #gives shape (st-1{2},st-1{2},st-1{2},st-1{1},vib,policyshit)
    temp1 = np.tensordot(copy,output,((0),(1)))
    
    #gives shape (st-1{1},st-1{1},st-1{1},st-1{2},st-1{2},st-1{2},vib,policyshit)
    temp2 = np.tensordot(copy,temp1,((0),(3)))
    

    #gives shape (a{1},st-1{1},st-1{1},st-1{2},st-1{2},vib,policyshit)
    temp3 = np.tensordot(policy1,temp2,((0,1),(0,3)))
    
    
    #gives shape (a{2},a{1},st-1{1},st-1{2},vib,policyshit)
    temp4 = np.tensordot(policy2,temp3,((0,1),(3,1)))
    
    #finally contract this with the new layer. Need to contract the old
    #output virtual with new input virtual, and new input state with old copy, and
    #new action.
    #gives (st{1},vie,st{2},policyshit)
    output = np.tensordot(contraction4,temp4,((1,2,3,5),(2,1,4,3)))
    
    output = np.swapaxis(output,1,2)
    return output


#function to optimise policy. Start from T-1 layer and contract everything else

def DRMGnew(flatreward,policy1,policy2,M1,M2,W,W1,p10,p20,copy,MT,WT,flatstate,T):
    ##we first optimize final layer.
    print('finallayer')
    #first contract all layers but last.     
    #first contract layer 1. 
    output = contractlayer1(flatreward,policy1,policy2,M1,M2,W1,p10,p20)

    #now contract all middle layers. 
    for i in range(1,T-1):
        output = contractlayer(output,policy1[i],policy2[i],copy,flatreward,W,M1,M2)
    #outputshape = (statet{1},statet{2},vie)
    
    #now must contract all final layer, except the policy. 
    
    #gives shape(statet{1},a1,r1)
    contraction1 = np.tensordot(MT,flatstate,((1),(0)))
    #gives shape(st{2},a2,r2)
    contraction2 = np.tensordot(MT,flatstate,((1),(0)))
    
    
    #contract with WT (st{1},a1,vi,r2)
    contraction3 = np.tensordot(contraction1,WT,((2),(1)))
    #(st{1},a1,vi,st{2},a2)
    contraction4 = np.tensordot(contraction3,contraction2,((3),(2)))
    
    
    
        
    #(st{1},st{1},st{1},st{2},vie)
    temp1 = np.tensordot(copy,output,((0),(0)))
    #(st{1},st{1},st1,vie,st2,st2,st2)
    temp2 = np.tensordot(temp1,copy,((2),(0)))
    
    #st{1},st1,st{2},st2,a1,a2
    temp3 = np.tensordot(temp2,contraction4,((0,4,3),(0,3,2)))
    
    # #st{1},st1,st{2},st2,a1,a2
    # temp3 = np.moveaxis(temp3,[2,1],[1,2])
    
    #now we need to optimise policy for one agent, before optimising for another agent. 
    N = 2 #2 times
    for n in range(N):
        #first optimise policy1
        
        #object of type (state1,state2,action1)
        temp4 = np.tensordot(temp3,policy2[T-1],((2,0,5),(0,1,2)))
        policy1[T-1] = changepolicy(policy1[T-1],temp4,T)
        
        #now optimise policy2
        temp5 = np.tensordot(temp3,policy1[T-1],((0,2,4),(0,1,2)))
        policy2[T-1] = changepolicy(policy2[T-1],temp5,T)

    #now optimize other layers
    for x in range(1,T-1):
        #t = layer being optimised. 
        t = T-x
        print('layer'+str(t))
        
        #first contract layer 1. 
        output = contractlayer1(flatreward,policy1,policy2,M1,M2,W1,p10,p20)
        #now contract all middle layers until chosen layer. 
        for i in range(1,t-1):
            output = contractlayer(output,policy1[i],policy2[i],copy,flatreward,W,M1,M2)
            #outputshape = (statet{1},statet{2},vie)
        output2 = createlayer(flatreward,policy1[t],policy2[t],M1,M2,W,W1,p10,p20,copy,MT,WT,flatstate,T)
        
        for i in range(t+1,T-1):
            output2 = contractafteroptimise(flatreward, policy1[i], policy2[i], M1, M2, W, W1, p10, p20, copy, MT, WT, flatstate, T, output2)
            #shape = soptimise{1},st{1},stoptimise{2},st{2},vioptimise,vie,
            
        output2 = contractlayerToptimise(flatreward,policy1[T-1],policy2[T-1],M1,M2,W,W1,p10,p20,copy,MT,WT,flatstate,T,output2)
            #(soptimise1,stoptimse{2},vioptimise)
                
        #now contract output and output2 with optimised layer.
        
        copytensor = np.zeros((noS,noS,noS))
        for i in range(noS):
            copytensor[i,i,i]=1
        
        #st-11,st2,a1,r1,st1,st1
        optimiselayer1 = np.tensordot(M1,copytensor,((1),(0)))
        optimiselayer2 = np.tensordot(M2,copytensor,((1),(0)))
        
        #st-11,a1,r1,st1,st-12,a2,r2,st2
        optimiselayer3 = np.tensordot(optimiselayer1,optimiselayer2,((4,1),(1,4)))
        #st-11,a1,st1,st-12,a2,st2,vib,vie
        optimiselayer4= np.tensordot(optimiselayer3,W,((2,6),(2,3)))
        
        #st-11,a1,st-12,a2,vib
        optimiselayer5 = np.tensordot(optimiselayer4,output2,((2,5,7),(0,1,2)))
        
        #(statet{2},state{2,st{2},statet{1},vie)
        temp1 = np.tensordot(copy,output,((0),(1)))
        #st1,st1,st1,st2,st2,st2,vie
        temp2 = np.tensordot(copy,temp1,((0),(3)))
        
        #st1,st1,st2,st2,a1,a2
        finalcontraction = np.tensordot(temp2,optimiselayer5,((0,3,6),(0,2,4)))
        
        N = 2 #2 times
        for n in range(N):
            #first optimise policy1
            
            #object of type (state1,state2,action1)
            temp4 = np.tensordot(finalcontraction,policy2[t-1],((2,0,5),(0,1,2)))
            policy1[T-1] = changepolicy(policy1[t-1],temp4,T)
            
            #now optimise policy2
            temp5 = np.tensordot(finalcontraction,policy1[t-1],((0,2,4),(0,1,2)))
            policy2[t-1] = changepolicy(policy2[t-1],temp5,T)
        
    #Now lets contract the first layer
    print('First layer')
    #st1,st2,a1,r1
    temp1 = np.tensordot(M1,p10,((0),(0)))
    temp2 = np.tensordot(M1,p20,((0),(0)))
    #st1,st1,st2,a1,r1
    temp11 = np.tensordot(copytensor,temp1,((0),(0)))
    temp22 = np.tensordot(copytensor,temp2,((0),(0)))
    #st1,a1,r1,st2,a2,r2
    temp3 = np.tensordot(temp11,temp22,((0,2),(2,0)))
    #st1,a1,st2,a2,vie
    temp4 = np.tensordot(temp3,W1,((2,5),(1,2)))
    
    output2 = createlayer(flatreward,policy1[1],policy2[1],M1,M2,W,W1,p10,p20,copy,MT,WT,flatstate,T)
    for i in range(2,T-1):
        output2 = contractafteroptimise(flatreward, policy1[i], policy2[i], M1, M2, W, W1, p10, p20, copy, MT, WT, flatstate, T, output2)
    
    output2 = contractlayerToptimise(flatreward,policy1[T-1],policy2[T-1],M1,M2,W,W1,p10,p20,copy,MT,WT,flatstate,T,output2)
    #(soptimise1,stoptimse{2},vioptimise)
    
    final = np.tensordot(temp4,output2,((0,2,4),(0,1,2)))
    #a1,a2
    
    N = 2 #2 times
    for n in range(N):
        #first optimise policy1

        temp1 = np.tensordot(p20,policy2[0],((0),(0)))
        temp2 = np.tensordot(temp1,p10,((0),(0)))
        temp3 = np.tensordot(temp2,final,((0),(1))) #now have an object of just a1.
        temp4 = np.tensordot(p10,p20,axes=0)
        temp5 = np.tensordot(temp4,temp3,axes=0)
        
        policy1[0] = changepolicy(policy1[0],temp5,T)
        
        #now optimise policy2
        temp1 = np.tensordot(p10,policy1[0],((0),(0)))
        temp2 = np.tensordot(temp1,p20,((0),(0)))
        temp3 = np.tensordot(temp2,final,((0),(0))) #now have an object of just a1.
        temp4 = np.tensordot(p10,p20,axes=0)
        temp5 = np.tensordot(temp4,temp3,axes=0)
        policy2[0] = changepolicy(policy2[0],temp5,T)
    
    return policy1,policy2
    
def createlayer(flatreward,policy1,policy2,M1,M2,W,W1,p10,p20,copy,MT,WT,flatstate,T):
    #st-1{1},st-1{1},st-1{1},st-1{2},a{1}
    contraction1 = np.tensordot(copy,policy1,((0),(0)))
    
    #st-1{2},st-1{2},st-1{2},st-1{1},a{2}
    contraction2 = np.tensordot(copy,policy2,((0),(0)))
    
    #st-1{1},st-1{1},st-1{2},st{1},st{2},r{1}
    contraction11 = np.tensordot(contraction1,M1,((0,4),(0,3)))
    
    #st-1{2},st-1{2},st-1{1},st{2},st{1},r{2}
    contraction22 = np.tensordot(contraction2,M2,((0,4),(0,3)))
    
    copytensor = np.zeros((noS,noS,noS))
    for i in range(noS):
        copytensor[i,i,i]=1
        
    #st-1{1},st-1{1},st-1{2},st{2},r{1},st{1},st{1},
    contraction111 = np.tensordot(contraction11,copytensor,((3),(0)))
    contraction222 = np.tensordot(contraction22,copytensor,((3),(0)))
    
    #st-1{1},r{1},st{1},st-1{2},r{2},st{2}
    contraction3 = np.tensordot(contraction111,contraction222,((0,2,5,3),(2,0,3,5)))

    #st-1{1},st{1},st-1{2},st{2},vib,vie
    contraction4 = np.tensordot(contraction3,W,((1,4),(2,3)))
    
    return contraction4

def contractafteroptimise(flatreward,policy1,policy2,M1,M2,W,W1,p10,p20,copy,MT,WT,flatstate,T,output):
    #st-1{1},st{1},st-1{2},st{2},vib,vie
    mainlayer = createlayer(flatreward,policy1,policy2,M1,M2,W,W1,p10,p20,copy,MT,WT,flatstate,T)
    
    #output dimensions(st-2{1},st-1{1},st-2{2},st-1{2},vib-1,vib)
    
    #st{1},st{2},vie,st-2{1},st-2{2},vib-1
    newoutput = np.tensordot(mainlayer,output,((0,2,4),(1,3,5)))

    #st-2{1},st{2},vie,st{1},st-2{2},vib-1
    newoutput = np.swapaxes(newoutput,3,0)
    #st-2{1},st{1},vie,st{2},st-2{2},vib-1
    newoutput = np.swapaxes(newoutput,3,1)
    #st-2{1},st{1},st-2{2},st{2},vie,vib-1
    newoutput = np.swapaxes(newoutput,2,4)
    #st-2{1},st{1},st-2{2},st{2},vibt-1,vie,
    newoutput = np.swapaxes(newoutput,4,5)
    
    return newoutput
    
def contractlayerToptimise(flatreward,policy1,policy2,M1,M2,W,W1,p10,p20,copy,MT,WT,flatstate,T,output):
    #shape = soptimise{1},st{1},stoptimise{2},st{2},vioptimise,vie,
    
    
    
    #first create the final layer.
    #st-1{1},st-1{1},st-1{1},st-1{2},a{1}
    contraction1 = np.tensordot(copy,policy1,((0),(0)))
    
    #st-1{2},st-1{2},st-1{2},st-1{1},a{2}
    contraction2 = np.tensordot(copy,policy2,((0),(0)))
    
    #st-1{1},st-1{1},st-1{2},st{1},r{1}
    contraction11 = np.tensordot(contraction1,MT,((0,4),(0,2)))
    #st-1{1},st-1{1},st-1{2},r{1}
    contraction111 = np.tensordot(contraction11, flatstate,((3),(0)))
    
    #st-1{2},st-1{2},st-1{1},st{2},r{2}
    contraction22 = np.tensordot(contraction2,MT,((0,4),(0,2)))
    #st-1{2},st-1{2},st-1{1},r{2}
    contraction222 = np.tensordot(contraction22, flatstate,((3),(0)))

    #st-1{1},r{1},st-1{2},r_{2}
    contraction3 = np.tensordot(contraction111,contraction222,((0,2),(2,0)))
    #st-1{1},st-1{2},vib
    contraction4 = np.tensordot(contraction3,WT,((1,3),(1,2)))


    #now contract with output. outputshape:#soptimise{1},st{1},stoptimise{2},st{2},vioptimise,vie,
    
    #shape (#soptimise1,stoptimse{2},vioptimise)
    finalcontraction = np.tensordot(contraction4,output,((0,1,2),(1,3,5)))
    
    return finalcontraction

def choosedistribution(dist,sd):
    if dist =='normal':
        prob1 = scipy.stats.norm.cdf(1,0,sd) - scipy.stats.norm.cdf(-1,0,sd)
        prob2 = (1-prob1)/2
    elif dist == 'uniform':
        prob1 = 1/3
        prob2 = 1/3
    else:
        print('error')
    return prob1,prob2

def debugM(M):
    for i in range(noS):
        for j in range(noS):
            for k in range(2):
                if np.sum(M[i,:,j,k,:]) !=1:
                    print(i,j,k)

#here is practice setting up the model. 
if __name__ == '__main__':
    
    #can choose value of T
    T = 4
    #number of states.
    noS = 2*T+1
    
    #create the flatstates, needed for contractions later. 
    flatreward = np.ones(6)
    flatstate = createflatstate(T)
    
    #initial probability vector has all zeros apart from at s=0. 
    p10 = np.zeros(noS)
    p10[int((noS+1)/2-1)] = 1
    
    p20 = np.zeros(noS)
    p20[int((noS+1)/2-1)] = 1
    
    #initialize an random policy
    policy1 = initializepolicy(T, noS)
    policy2 = initializepolicy(T, noS)
    # policy1 = np.ones((T,noS,noS,2))
    # policy1[:,:,:,0] = policy1[:,:,:,0]-1
    # policy2 = np.ones((T,noS,noS,2))
    # policy2[:,:,:,1] = policy2[:,:,:,1]-1


    
    rewardvector = np.asarray([1,0,-1,-2,-3,-10])
    
    #prob1,prob2 = choosedistribution('normal', 1)
    prob1 = 1
    prob2 = 0
    #M tensors are same for all time steps apart from last (MT)
    M1=createM1(noS,prob1,prob2)
    M2=createM2(noS,prob1,prob2)

    MT=createMT(noS,prob1,prob2)
    
    rewardmatrix = createrewardmatrix(rewardvector)
    #W tensors are same for all timesteps apart from timesteps 1 and T
    W = createW(rewardmatrix)
    W1 = createW1(rewardmatrix)
    WT = createWT(rewardmatrix)    
    
    #copy tensor is essentially a 3D identity. 
    copy = createcopytensor(noS)

    er = contracteverything(M1,M2,MT,W,W1,WT,flatreward,policy1,policy2,p10,p20,flatstate,T,copy,False)
    print(er)
    policy1,policy2 = DRMGnew(flatreward,policy1,policy2,M1,M2,W,W1,p10,p20,copy,MT,WT,flatstate,T)
    plt.figure()
    contracteverything(M1,M2,MT,W,W1,WT,flatreward,policy1,policy2,p10,p20,flatstate,T,copy,False)