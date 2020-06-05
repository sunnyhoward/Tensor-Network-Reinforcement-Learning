from randomwalkertensor import (createflatstate, initializepolicy, createM,createMT,
                                createW,createW1,createWT,createcopytensor,
                                contracteverything,DRMGpolicyoptimisation)
import numpy as np 
import matplotlib.pyplot as plt


def generatetrajectory(policy,T,p0):
   state = p0
   s = np.where(state==1)[0]
   shist=np.zeros(T+1)
   shist[0] = s
   ahist = np.zeros(T)
   rhist = np.zeros(T)
   
   for t in range(T):
       #get action
       #action = np.argmax(np.tensordot(state,policy[t],((0),(0))))
       if np.tensordot(state,policy[t],((0),(0)))[1] > np.random.rand(1):
           action = 1
       else:
           action = 0
       ahist[t] = action
       state = np.zeros(2*T+1)
       
       if action == 1:
          state[s+1] = 1
           
       else:
           state[s-1] = 1
           
       s = np.where(state==1)[0]
       shist[t+1] = s
       if t!=T-1:
           if s[0]<(len(p0)+1)/2-1:
               rhist[t] = -1
       elif t ==T-1:
           if s[0]==(len(p0)+1)/2-1:
               rhist[t] = 1
           else:
               rhist[t] = -10
               
   return shist,ahist,rhist

if __name__ == '__main__':
    
    #can choose value of T
    T = 6
    #number of states.
    noS = 2*T+1
    
    #create the flatstates, needed for contractions later. 
    flatreward = np.ones(4)
    flatstate = createflatstate(T)
    
    #initial probability vector has all zeros apart from at s=0. 
    p0 = np.zeros(noS)
    p0[int((noS+1)/2-1)] = 1
    
    #initialize an random policy
    policy = initializepolicy(T, noS)
    
    rewardvector = np.asarray([1,0,-1,-10])
    
    #M tensors are same for all time steps apart from last (MT)
    M=createM(noS)
    MT=createMT(noS)
    
    #W tensors are same for all timesteps apart from timesteps 1 and T
    W = createW(rewardvector)
    W1 = createW1(rewardvector)
    WT = createWT(rewardvector)    
    
    #copy tensor is essentially a 3D identity. 
    copy = createcopytensor(noS)

    contracteverything(M,MT,W,W1,WT,flatreward,policy,p0,flatstate,T,copy)
    
    
    Ntraj =1000
    storetraj = np.zeros((Ntraj,T+1))
    storeaction = np.zeros((Ntraj,T))
    storerewards = np.zeros((Ntraj,T))
    for i in range(Ntraj):
        storetraj[i], storeaction[i], storerewards[i] = generatetrajectory(policy, T, p0)
    
    
    
    
    newM = np.zeros((noS,noS,4,T)) #oldstate,newstate,reward,time
    for t in range(T):
        print(t)
        for a in range(noS): 
            occur = np.where(storetraj[:,t] == a)[0] #find index of where value occured.
            sdash = storetraj[np.transpose(occur),t+1]
            reward = storerewards[np.transpose(occur),t]
            nextstates,frequency = np.unique(sdash,return_counts=True)
            for i in range(len(nextstates)):
                s = nextstates[i]
                temp = reward[np.where(sdash==int(s))[0][0]]
                if temp == 1:
                    rewardindex = 0
                elif temp == 0:
                    rewardindex = 1
                elif temp == -1:
                    rewardindex = 2
                elif temp == -10:
                    rewardindex = 3
                newM[a,int(s),rewardindex,t] = frequency[i]/np.sum(frequency)
    
    M0 = newM[:,:,:,0]
    M1 = newM[:,:,:,1]
    M2= newM[:,:,:,2]
    M3 = newM[:,:,:,3]
    M4 = newM[:,:,:,4]
    M5 = newM[:,:,:,5]
    
    n0= np.tensordot(M0,rewardvector,((2),(0)))
    n1 = np.tensordot(M1,rewardvector,((2),(0)))
    n2 = np.tensordot(M2,rewardvector,((2),(0)))
    n3 = np.tensordot(M3,rewardvector,((2),(0)))
    n4 = np.tensordot(M4,rewardvector,((2),(0)))
    n5 = np.tensordot(M5,rewardvector,((2),(0)))
    
    
    
    
    temp1 = np.tensordot(M0,M1,((1),(0))) #p0,reward0,output1,reward1
    temp2 = np.tensordot(temp1,M2,((2),(0)))#p0,reward0,reward1,output2,reward2...
    temp3 = np.tensordot(temp2,M3,((3),(0)))    
    temp4 = np.tensordot(temp3,M4,((4),(0)))    
    temp5 = np.tensordot(temp4,M5,((5),(0)))    
    contraction1 = np.tensordot(temp5,p0,((0),(0)))
    contraction2 = np.tensordot(contraction1,flatstate,((5),(0))) #reward0,reward1,reward2...
    
    contraction3 = np.tensordot(contraction2, flatreward,((0),(0)))
    contraction4 = np.tensordot(contraction3, flatreward,((0),(0)))
    contraction5 = np.tensordot(contraction4, flatreward,((0),(0)))
    contraction6 = np.tensordot(contraction5, flatreward,((0),(0)))
    contraction7 = np.tensordot(contraction6, flatreward,((0),(0)))
    rewardfinalstep = np.tensordot(contraction7, rewardvector,((0),(0)))

    
    
    for i in range(T):
        print(np.mean(storerewards[:,i]))
            
    
    # policy = DRMGpolicyoptimisation(flatreward, policy, M, W, W1, p0, copy, MT, WT, flatstate, T)
    # plt.figure()
    # contracteverything(M,MT,W,W1,WT,flatreward,policy,p0,flatstate,T,copy)


