import numpy as np
from numpy import unravel_index
import matplotlib.pyplot as plt
import scipy.stats

#function to create the initial policy
def initializepolicy(T,noS):
    #policy represents P(a_t|s^1_{t-1},s^2_{t-1})
    policy = np.ones((T,noS,noS,2)) * 0.5
    return policy

#plot path chosen by policy. 
def plotgreedy(policy1,policy2,T,p10,p20,M1,M2,MT,copy2):
   state1 = p10
   state2 = p20
   s1 = np.where(state1==1)[0]
   s2 = np.where(state2==1)[0]
   shist1=np.zeros(T+1)
   shist1[0] = s1
   shist2=np.zeros(T+1)
   shist2[0] = s2
   
   for t in range(T-1):
       #get action
       temp1 = np.tensordot(state1,policy1[t],((0),(0)))
       temp2 = np.tensordot(state2,policy2[t],((0),(0)))

       action1 = np.tensordot(state2,temp1,((0),(0)))
       action2 = np.tensordot(state1,temp2,((0),(0)))

       temp1 = np.tensordot(state1,M1,((0),(0)))
       temp2 = np.tensordot(state2,M2,((0),(0)))

       temp1 = np.tensordot(action1,temp1,((0),(2)))
       temp2 = np.tensordot(action2,temp2,((0),(2)))
       
       #st+1,s2
       temp1 = np.tensordot(np.ones(6),temp1,((0),(2)))
       temp2 = np.tensordot(np.ones(6),temp2,((0),(2)))
       
       temp1 = np.tensordot(temp1,copy2,((0),(0)))
       temp2 = np.tensordot(temp2,copy2,((0),(0)))
       
       #st+1(1),st+1(2)
       temp3 = np.tensordot(temp1,temp2,((0,2),(2,0)))
       
       states = np.unravel_index(temp3.argmax(), temp3.shape)
       s1 = states[0]
       s2 = states[1]



       state1 = np.zeros(2*T+1)
       state2 = np.zeros(2*T+1)
       state1[s1] = 1
       state2[s2] = 1
       
           
       shist1[t+1] = s1
       shist2[t+1] = s2

   temp1 = np.tensordot(state1,policy1[T-1],((0),(0)))
   temp2 = np.tensordot(state2,policy2[T-1],((0),(0)))

   action1 = np.tensordot(state2,temp1,((0),(0)))
   action2 = np.tensordot(state1,temp2,((0),(0)))

   temp1 = np.tensordot(state1,MT,((0),(0)))
   temp2 = np.tensordot(state2,MT,((0),(0)))
   
   #st+1,r
   temp1 = np.tensordot(action1,temp1,((0),(1)))
   temp2 = np.tensordot(action2,temp2,((0),(1)))
    
    #st+1
   temp1 = np.tensordot(np.ones(6),temp1,((0),(1)))
   temp2 = np.tensordot(np.ones(6),temp2,((0),(1)))
    
   
    
   
   s1 = np.argmax(temp1)
   s2 = np.argmax(temp2)  
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


# #plot path chosen by policy. 
# def plotgreedy(policy1,policy2,T,p10,p20):
#    state1 = p10
#    state2 = p20
#    s1 = np.where(state1==1)[0]
#    s2 = np.where(state2==1)[0]
#    shist1=np.zeros(T+1)
#    shist1[0] = s1
#    shist2=np.zeros(T+1)
#    shist2[0] = s2
   
#    for t in range(T):
#        #get action
#        temp1 = np.tensordot(state1,policy1[t],((0),(0)))
#        temp2 = np.tensordot(state2,policy2[t],((0),(0)))

#        action1 = np.argmax(np.tensordot(state2,temp1,((0),(0))))
#        action2 = np.argmax(np.tensordot(state1,temp2,((0),(0))))

#        state1 = np.zeros(2*T+1)
#        state2 = np.zeros(2*T+1)
       
#        if action1 == 1:
#            state1[s1+1] = 1
#        else:
#            state1[s1-1] = 1
           
#        if action2 == 1:
#            state2[s2+1] = 1
#        else:
#            state2[s2-1] = 1
           
#        s1 = np.where(state1==1)[0]
#        shist1[t+1] = s1
#        s2 = np.where(state2==1)[0]
#        shist2[t+1] = s2
       
#    plt.plot(range(T+1),shist1-T)
#    plt.plot(range(T+1),shist2-T)
#    plt.plot(range(-1,T+2),np.zeros(T+3),'r:')
#    plt.xlim(0,T)
#    plt.ylim(-T,T)
#    plt.xlabel('Time')
#    plt.ylabel('Position')
#    plt.xticks(np.arange(0,T+1))
#    plt.yticks(np.arange(-T,T))
#    plt.legend(['Agent 1', 'Agent 2'])
#    plt.grid()
   
# this will only be relevent for planning
def generatetrajectoryarray(policy1,policy2,T,p10,p20,exploration,M1,M2,MT,copy2,rewardvector):
    state1 = p10
    state2 = p20
    s1 = np.where(state1==1)[0]
    s2 = np.where(state2==1)[0]
    shist1=np.zeros(T+1)
    shist2=np.zeros(T+1)
    shist1[0] = s1
    shist2[0] = s2
    ahist1 = np.zeros(T)
    rhist1 = np.zeros(T)
    ahist2 = np.zeros(T)
    rhist2 = np.zeros(T)
   
    for t in range(T-1):
        #get action
        action1 = (np.tensordot(state1,policy1[t],((0),(0))))
        action1 = (np.tensordot(state2,action1,((0),(0))))
        action2 = (np.tensordot(state2,policy2[t],((0),(0))))
        action2 = (np.tensordot(state1,action2,((0),(0))))

           
        if np.random.rand(1)<exploration:
            action1 = 1-action1
           
        if np.random.rand(1)<exploration:
            action2 = 1-action2
          
        ahist1[t] = np.random.choice([0,1],p = action1)
        ahist2[t] = np.random.choice([0,1],p = action2)
        
        action1 = np.asarray([1-ahist1[t],ahist1[t]])
        action2 = np.asarray([1-ahist2[t],ahist2[t]])

       
        c1 = np.tensordot(state1,M1,((0),(0)))
        c2 = np.tensordot(state2,M2,((0),(0)))
        #(st{1}),st{2},rt{1}
        c11 = np.tensordot(c1,action1,((2),(0)))
        c22 = np.tensordot(c2,action2,((2),(0)))
        #(st{1},st{1},st{2},rt{1})
        c111 = np.tensordot(copy2,c11,((0),(0)))
        #(st{2},st{2},st{1},rt{2})
        c222 = np.tensordot(copy2,c22,((0),(0)))
       
        #(st{1},r{1},st{2},rt{2})
        c3 = np.tensordot(c111,c222,((0,2),(2,0)))
       
        possible = np.asarray(np.where(c3>0))
       
       
    
        probs = c3[c3>0]/np.sum(c3)
        s = np.random.choice(np.arange(len(possible[0,:])),p = probs)
        chosen = possible[:,s]
        state1 = np.zeros(2*T+1)
        state1[chosen[0]] = 1
        state2 = np.zeros(2*T+1)
        state2[chosen[2]] = 1
        shist1[t+1] = chosen[0]
        shist2[t+1] = chosen[2]
        rhist1[t] = rewardvector[chosen[1]]
        rhist2[t] = rewardvector[chosen[3]]
       
       
    #now do final timestep
    action1 = (np.tensordot(state1,policy1[T-1],((0),(0))))   
    action1 = (np.tensordot(state2,action1,((0),(0))))
    action2 = (np.tensordot(state2,policy2[T-1],((0),(0))))
    action2 = (np.tensordot(state1,action2,((0),(0))))
    
    
    if np.random.rand(1)<exploration:
            action1 = 1-action1
           
    if np.random.rand(1)<exploration:
            action2 = 1-action2
          
    ahist1[T-1] = np.random.choice([0,1],p = action1)
    ahist2[T-1] = np.random.choice([0,1],p = action2)
    action1 = np.asarray([1-ahist1[T-1],ahist1[T-1]])
    action2 = np.asarray([1-ahist2[T-1],ahist2[T-1]])
    
   
    c1 = np.tensordot(state1,MT,((0),(0)))
    c2 = np.tensordot(state2,MT,((0),(0)))
    c11 = np.tensordot(c1,action1,((1),(0)))
    c22 = np.tensordot(c2,action2,((1),(0)))
    
    possible = np.asarray(np.where(c11>0))
    possible2 = np.asarray(np.where(c22>0))
    probs1 = c11[c11>0]
    probs2 = c22[c22>0]
    
    choose1 = np.random.choice(np.arange(len(probs1)),p = probs1)
    choose2 = np.random.choice(np.arange(len(probs2)),p = probs2)

    s1,r1 = possible[:,choose1]
    s2,r2 = possible2[:,choose2]


    shist1[T] = s1
    shist2[T] = s2

    rhist1[T-1] = rewardvector[r1]
    rhist2[T-1] = rewardvector[r2]

    return shist1,shist2,ahist1,ahist2,rhist1,rhist2


#function to create the rank 5 tensor. 
def createM1(noS,prob1,prob2):
    M = np.zeros((noS,noS,noS,2,6)) 
    
    # M[S_{t-1}^1,S_{t}^1,S_{t}^2,a^{1},r{1}]

    #possible values for reward are 1, 0, -1, -2, -3, -10
    

    
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


#function to create the rank 5 tensor for agent 2. 
def createM2(noS,prob1,prob2):
    M = np.zeros((noS,noS,noS,2,6)) 
    
    # M[S_{t-1}^2,S_{t}^2,S_{t}^1,a^{2},r_{2}]

    #possible values for reward are 1, 0, -1, -2, -3, -10
    

    
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


def contractlayer1(flatreward,policy1,policy2,M1,M2,W1,p10,p20,copy2):
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
    
    
    #gives shape(endstate1,endstate1,endstate2,reward1)
    contraction11111 = np.tensordot(copy2,contraction11,((0),(0)))
    #gives shape(endstate2,endstate2,endstate1,reward2)
    contraction22222 = np.tensordot(copy2,contraction22, ((0),(0)))
    
    #final shape = (endstate1,reward1,endstate2,reward2)
    joint = np.tensordot(contraction11111,contraction22222,((0,2),(2,0)))

    #shape = (endstate1,endstate2,vi)
    finalcontraction = np.tensordot(joint,W1,((1,3),(1,2)))

    return finalcontraction
    #now the first layer is fully contracted. repeat this process for every layer but the last.
    
    

#function to manage the contracts betweenlayers, 
#such as policy and virtual index. This wont work if at layers 1 or end. 
def contractlayer(previouslayeroutput,policy1,policy2,copy3,copy2,flatreward,W,M1,M2):
    
    #first contract M with W to give object of type
    #(statet-1{1},statet{1},statet{2},at-1{1},virtualbegin1,virtualend1,reward2)
    contraction1 = np.tensordot(M1,W,((4),(2)))

        
#(statet{1},statet{1},statet-1{1},statet{2},at-1{1},virtualbegin1,virtualend1,reward2)
    layer1 = np.tensordot(copy2,contraction1,((0),(1)))
    #statet-1{2},statet{1},at-1{2},reward{2},statet{2},statet{2}
    layer2 = np.tensordot(M2,copy2,((1),(0)))

    #(statet{1},statet-1{1},at-1{1},vib,vie,statet-1{2},at-1{2},statet{2})
    connect = np.tensordot(layer1,layer2,((0,3,7),(1,4,3)))
    

    #previouslayer output is shape (endstate1,endstate2,vib)
    #use copytensor to get(endstate1,endstate1,endstate1,vib,endstate2,es2,es2)
    temp1 = np.tensordot(copy3,previouslayeroutput,((0),(0)))
    temp2 = np.tensordot(temp1,copy3,((3),(0)))

    
    
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
def contractlayerT(MT,WT,flatstate,copy3,previouslayeroutput,policy1,policy2,flatreward):
    
    #gives shape(input state,action,reward)
    contraction1 = np.tensordot(MT,flatstate,((1),(0)))
    

    
    #gives shape(input state,action,reward)
    contraction2 = np.tensordot(MT,flatstate,((1),(0)))
    
    
    #makeshape (statet-1_{1},statet-1_{1},statet-1{1},statet-1{2}
    #vie)    
    contraction4 = np.tensordot(copy3,previouslayeroutput,((0),(0)))
    
    #makeshape (statet-1_{1},statet-1_{1},statet-1{1},
    #vie,statet-1_{2},statet-1_{2},statet-1{2})    
    contraction5 = np.tensordot(contraction4,copy3,((3),(0)))

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
def contracteverything(M1,M2,MT,W,W1,WT,flatreward,policy1,policy2,p10,p20,flatstate,T,copy3,copy2,plot):
    
    #first contract layer 1. 
    output = contractlayer1(flatreward,policy1,policy2,M1,M2,W1,p10,p20,copy2)

    #now contract all middle layers. 
    for i in range(1,T-1):
        output = contractlayer(output,policy1[i],policy2[i],copy3,copy2,flatreward,W,M1,M2)
    
    #now contract final layer
    
    expectedreturn = contractlayerT(MT,WT,flatstate,copy3,output,policy1[T-1],policy2[T-1],flatreward)
    
    #print(expectedreturn)
    if plot==True:
        plotgreedy(policy1,policy2,T,p10,p20,M1,M2,MT,copy2)
    return expectedreturn


#function to change the values in policy array, used in function below.
def changepolicy(policy1,policy2,contraction,T):
    #contraction is of form state1,state1,state2,state2,action1,action2
    for i in range(2*T+1): # 
        for j in range(2*T+1):
            if len(np.unique(contraction[i,i,j,j])) !=1:
                value = np.argmax(contraction[i,i,j,j])
                    
                if value == 0:
                        policy1[i,j,0] = 1
                        policy2[j,i,0] = 1
                        policy1[i,j,1] = 0
                        policy2[j,i,1] = 0
                elif value == 1:
                        policy1[i,j,0] = 1
                        policy2[j,i,0] = 0
                        policy1[i,j,1] = 0
                        policy2[j,i,1] = 1
                elif value == 2:
                        policy1[i,j,0] = 0
                        policy2[j,i,0] = 1
                        policy1[i,j,1] = 1
                        policy2[j,i,1] = 0
                elif value ==3: 
                        policy1[i,j,0] = 0
                        policy2[j,i,0] = 0 
                        policy1[i,j,1] = 1
                        policy2[j,i,1] = 1
                            
                            
    return policy1,policy2
        
#it is easier to also have a function to contract all stuff after the layer being
#optimised.

#function to optimise policy. Start from T-1 layer and contract everything else

def DRMGnew(flatreward,policy1,policy2,M1,M2,W,W1,p10,p20,copy3,copy2,MT,WT,flatstate,T,N):
    ##we first optimize final layer.
    print('finallayer')
    #first contract all layers but last.     
    #first contract layer 1. 
    output = contractlayer1(flatreward,policy1,policy2,M1,M2,W1,p10,p20,copy2)

    #now contract all middle layers. 
    for i in range(1,T-1):
        output = contractlayer(output,policy1[i],policy2[i],copy3,copy2,flatreward,W,M1,M2)
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
    temp1 = np.tensordot(copy3,output,((0),(0)))
    #(st{1},st{1},st1,vie,st2,st2,st2)
    temp2 = np.tensordot(temp1,copy3,((3),(0)))
    
    #st{1},st1,st{2},st2,a1,a2
    temp3 = np.tensordot(temp2,contraction4,((0,4,3),(0,3,2)))
    
    # #st{1},st1,st{2},st2,a1,a2
    
    #now we need to optimise policy for both agents at once 
    
    policy1[T-1],policy2[T-1] = changepolicy(policy1[T-1],policy2[T-1], temp3, T)
    
    
    
    #We also must contract the second last layer seperately.
    print('layer'+str(T-1))
        
    #first contract layer 1. 
    output = contractlayer1(flatreward,policy1,policy2,M1,M2,W1,p10,p20,copy2)
    #now contract all middle layers until chosen layer. 
    for i in range(1,T-2):
        output = contractlayer(output,policy1[i],policy2[i],copy3,copy2,flatreward,W,M1,M2)
         #outputshape = (statet{1},statet{2},vie)
        
    #now create the final layer.
    #st-1{1},st-1{1},st-1{1},st-1{2},a{1}
    contraction1 = np.tensordot(copy3,policy1[T-1],((0),(0)))
    
    #st-1{2},st-1{2},st-1{2},st-1{1},a{2}
    contraction2 = np.tensordot(copy3,policy2[T-1],((0),(0)))
    
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
    output2 = np.tensordot(contraction3,WT,((1,3),(1,2)))
    #output2shape = (statet-1{1},statet-1{2},vib)
       

    
    #st-11,st2,a1,r1,st1,st1
    optimiselayer1 = np.tensordot(M1,copy2,((1),(0)))
    optimiselayer2 = np.tensordot(M2,copy2,((1),(0)))
    
    #st-11,a1,r1,st1,st-12,a2,r2,st2
    optimiselayer3 = np.tensordot(optimiselayer1,optimiselayer2,((4,1),(1,4)))
    #st-11,a1,st1,st-12,a2,st2,vib,vie
    optimiselayer4= np.tensordot(optimiselayer3,W,((2,6),(2,3)))
    
    #st-11,a1,st-12,a2,vib
    optimiselayer5 = np.tensordot(optimiselayer4,output2,((2,5,7),(0,1,2)))
    
    #(statet{2},state{2,st{2},statet{1},vie)
    temp1 = np.tensordot(copy3,output,((0),(1)))
    #st1,st1,st1,st2,st2,st2,vie
    temp2 = np.tensordot(copy3,temp1,((0),(3)))
    
    #st1,st1,st2,st2,a1,a2
    finalcontraction = np.tensordot(temp2,optimiselayer5,((0,3,6),(0,2,4)))
    
    policy1[T-2],policy2[T-2] = changepolicy(policy1[T-2],policy2[T-2], finalcontraction, T)

    
        

    #now optimize all other layers
    for x in range(2,T-1):
        #t = layer being optimised. 
        t = T-x
        print('layer'+str(t))
        
        #first contract layer 1. 
        output = contractlayer1(flatreward,policy1,policy2,M1,M2,W1,p10,p20,copy2)
        #now contract all middle layers until chosen layer. 
        for i in range(1,t-1):
            output = contractlayer(output,policy1[i],policy2[i],copy3,copy2,flatreward,W,M1,M2)
            #outputshape = (statet{1},statet{2},vie)
        
        #create layer after chosen layer
        output2 = createlayer(flatreward,policy1[t],policy2[t],M1,M2,W,W1,p10,p20,copy3,copy2,MT,WT,flatstate,T)
        
        #contract all layers after that
        for i in range(t+1,T-1):
            output2 = contractafteroptimise(flatreward, policy1[i], policy2[i], M1, M2, W, W1, p10, p20, copy3,copy2, MT, WT, flatstate, T, output2)
            #shape = soptimise{1},st{1},stoptimise{2},st{2},vioptimise,vie,
        
        #contract final layer to it.
        output2 = contractlayerToptimise(flatreward,policy1[T-1],policy2[T-1],M1,M2,W,W1,p10,p20,copy3,MT,WT,flatstate,T,output2)
            #(soptimise1,stoptimse{2},vioptimise)
                
        #now contract output and output2 with optimised layer.
        #st-11,st2,a1,r1,st1,st1
        optimiselayer1 = np.tensordot(M1,copy2,((1),(0)))
        optimiselayer2 = np.tensordot(M2,copy2,((1),(0)))
        
        #st-11,a1,r1,st1,st-12,a2,r2,st2
        optimiselayer3 = np.tensordot(optimiselayer1,optimiselayer2,((4,1),(1,4)))
        #st-11,a1,st1,st-12,a2,st2,vib,vie
        optimiselayer4= np.tensordot(optimiselayer3,W,((2,6),(2,3)))
        
        #st-11,a1,st-12,a2,vib
        optimiselayer5 = np.tensordot(optimiselayer4,output2,((2,5,7),(0,1,2)))
        
        #(statet{2},state{2,st{2},statet{1},vie)
        temp1 = np.tensordot(copy3,output,((0),(1)))
        #st1,st1,st1,st2,st2,st2,vie
        temp2 = np.tensordot(copy3,temp1,((0),(3)))
        
        #st1,st1,st2,st2,a1,a2
        finalcontraction = np.tensordot(temp2,optimiselayer5,((0,3,6),(0,2,4)))
        
        policy1[t-1],policy2[t-1] = changepolicy(policy1[t-1],policy2[t-1], finalcontraction, T)

        # if policytooptimise == 1:
        #     temp4 = np.tensordot(finalcontraction,policy2[t-1],((2,0,5),(0,1,2)))
        #     policy1[t-1] = changepolicy(policy1[t-1],temp4,T)
        # elif policytooptimise == 2:
        #     temp5 = np.tensordot(finalcontraction,policy1[t-1],((0,2,4),(0,1,2)))
        #     temp5 = np.swapaxes(temp5,0,1)
        #     policy2[t-1] = changepolicy(policy2[t-1],temp5,T)
        
       
        
    #Now lets contract the first layer
    print('First layer')
    #st1,st2,a1,r1
    temp1 = np.tensordot(M1,p10,((0),(0)))
    temp2 = np.tensordot(M1,p20,((0),(0)))
    #st1,st1,st2,a1,r1
    temp11 = np.tensordot(copy2,temp1,((0),(0)))
    temp22 = np.tensordot(copy2,temp2,((0),(0)))
    #st1,a1,r1,st2,a2,r2
    temp3 = np.tensordot(temp11,temp22,((0,2),(2,0)))
    #st1,a1,st2,a2,vie
    temp4 = np.tensordot(temp3,W1,((2,5),(1,2)))
    
    output2 = createlayer(flatreward,policy1[1],policy2[1],M1,M2,W,W1,p10,p20,copy3,copy2,MT,WT,flatstate,T)
    for i in range(2,T-1):
        output2 = contractafteroptimise(flatreward, policy1[i], policy2[i], M1, M2, W, W1, p10, p20, copy3,copy2, MT, WT, flatstate, T, output2)
    
    output2 = contractlayerToptimise(flatreward,policy1[T-1],policy2[T-1],M1,M2,W,W1,p10,p20,copy3,MT,WT,flatstate,T,output2)
    #(soptimise1,stoptimse{2},vioptimise)
    
    final = np.tensordot(temp4,output2,((0,2,4),(0,1,2)))
    #a1,a2
    
    temp1 = np.tensordot(p10,p10,axes=0)
    temp2 = np.tensordot(temp1,p20,axes=0)
    temp3 = np.tensordot(temp2,p20,axes=0)
    temp4 = np.tensordot(temp3,final,axes=0)
    policy1[0],policy2[0] = changepolicy(policy1[0],policy2[0],temp4,T)
    
    
    return policy1,policy2
    
    
def createlayer(flatreward,policy1,policy2,M1,M2,W,W1,p10,p20,copy3,copy2,MT,WT,flatstate,T):
    #st-1{1},st-1{1},st-1{1},st-1{2},a{1}
    contraction1 = np.tensordot(copy3,policy1,((0),(0)))
    
    #st-1{2},st-1{2},st-1{2},st-1{1},a{2}
    contraction2 = np.tensordot(copy3,policy2,((0),(0)))
    
    #st-1{1},st-1{1},st-1{2},st{1},st{2},r{1}
    contraction11 = np.tensordot(contraction1,M1,((0,4),(0,3)))
    
    #st-1{2},st-1{2},st-1{1},st{2},st{1},r{2}
    contraction22 = np.tensordot(contraction2,M2,((0,4),(0,3)))
    
        
    #st-1{1},st-1{1},st-1{2},st{2},r{1},st{1},st{1},
    contraction111 = np.tensordot(contraction11,copy2,((3),(0)))
    contraction222 = np.tensordot(contraction22,copy2,((3),(0)))
    
    #st-1{1},r{1},st{1},st-1{2},r{2},st{2}
    contraction3 = np.tensordot(contraction111,contraction222,((0,2,5,3),(2,0,3,5)))

    #st-1{1},st{1},st-1{2},st{2},vib,vie
    contraction4 = np.tensordot(contraction3,W,((1,4),(2,3)))
    
    return contraction4

def contractafteroptimise(flatreward,policy1,policy2,M1,M2,W,W1,p10,p20,copy3,copy2,MT,WT,flatstate,T,output):
    #st-1{1},st{1},st-1{2},st{2},vib,vie
    mainlayer = createlayer(flatreward,policy1,policy2,M1,M2,W,W1,p10,p20,copy3,copy2,MT,WT,flatstate,T)
    
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
    
def contractlayerToptimise(flatreward,policy1,policy2,M1,M2,W,W1,p10,p20,copy3,MT,WT,flatstate,T,output):
    #shape = soptimise{1},st{1},stoptimise{2},st{2},vioptimise,vie,
    
    
    
    #first create the final layer.
    #st-1{1},st-1{1},st-1{1},st-1{2},a{1}
    contraction1 = np.tensordot(copy3,policy1,((0),(0)))
    
    #st-1{2},st-1{2},st-1{2},st-1{1},a{2}
    contraction2 = np.tensordot(copy3,policy2,((0),(0)))
    
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
    elif dist=='deterministic':
        prob1 = 1
        prob2 = 0 
        
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
    T = 6
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


    
    rewardvector = np.asarray([1,0,-1,-2,-3,-10])
    
    prob1,prob2 = choosedistribution('deterministic', 1)
    
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
    copy3 = createcopytensor3(noS)
    copy2 = createcopytensor2(noS)
    
    er = contracteverything(M1,M2,MT,W,W1,WT,flatreward,policy1,policy2,p10,p20,flatstate,T,copy3,copy2,True)
    print(er)
    
    policy1,policy2 = DRMGnew(flatreward,policy1,policy2,M1,M2,W,W1,p10,p20,copy3,copy2,MT,WT,flatstate,T,N=1)
    
    plt.figure()
    er = contracteverything(M1,M2,MT,W,W1,WT,flatreward,policy1,policy2,p10,p20,flatstate,T,copy3,copy2,True)
    # print(er)
    
    # exploration = 0
    # sarray1 = np.zeros((100,7))
    # sarray2 = np.zeros((100,7))
    # for i in range(100):
    #     shist1,shist2,ahist1,ahist2,rhist1,rhist2 = generatetrajectoryarray(policy1,policy2, T, p10, p20, exploration, M1, M2, MT, copy2, rewardvector)
    #     sarray1[i] = shist1
    #     sarray2[i] = shist2

    # plt.figure()
    # for i in range(100):
    #     plt.plot(np.arange(T+1),sarray1[i]-T,'b',alpha=0.2)
    #     plt.plot(np.arange(T+1),sarray2[i]-T,'#ff7f0e',alpha=0.2)
    # plt.plot(range(-1,T+2),np.zeros(T+3),'r:')
    # plt.xlim(0,T)
    # plt.ylim(-T,T)
    # plt.xlabel('Time')
    # plt.ylabel('Position')
    # plt.xticks(np.arange(0,T+1))
    # plt.yticks(np.arange(-T,T))
    # plt.legend(['Agent 1', 'Agent 2'])
    # plt.grid()