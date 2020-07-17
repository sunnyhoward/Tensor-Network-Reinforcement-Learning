
from bigM import (createflatstate, initializepolicy, createM,createMT,
                                createW,createW1,createWT,createrewardmatrix,
                                contracteverything,DRMG,choosedistribution,createcopytensor3,createcopytensor2,plotgreedy)

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches


#function for generating trajectories and updating model.
def modellearning(T,policy,p0,M,MT,copy2,rewardvector,modelM,modelMT,Ntraj,exploration,lrate,flatstate,noS):
    #generate the trajectories
    storetraj1 = np.zeros((Ntraj,T+1))
    storetraj2 = np.zeros((Ntraj,T+1))

    storeaction1 = np.zeros((Ntraj,T))
    storeaction2 = np.zeros((Ntraj,T))

    storerewards1 = np.zeros((Ntraj,T))
    storerewards2 = np.zeros((Ntraj,T))

    for i in range(Ntraj):
        storetraj1[i], storetraj2[i],storeaction1[i], storeaction2[i],storerewards1[i],storerewards2[i] = generatetrajectoryarray(policy,T,p0,M,MT,copy2,exploration)
        
    newmodelM0 = np.zeros((noS,noS,6,6,noS,noS,2,2)) 
    newmodelM1 = np.zeros((noS,noS,6,6,noS,noS,2,2)) 
    newmodelM2 = np.zeros((noS,noS,6,6,noS,noS,2,2)) 
    newmodelM3 = np.zeros((noS,noS,6,6,noS,noS,2,2)) 
    newmodelM4 = np.zeros((noS,noS,6,6,noS,noS,2,2)) 
    newmodelMT = np.zeros((noS,noS,6,6,noS,noS,2,2)) 

    for t in range(T):
        for s1 in range(noS):
            index1 = np.where(storetraj1[:,t] == s1)[0]
            state2t = storetraj2[index1,t]
            actions1 = storeaction1[index1,t]
            actions21 = storeaction2[index1,t]
            rewards1 = storerewards1[index1,t]
            rewards21 = storerewards2[index1,t]
            nextstate1 = storetraj1[index1,t+1]      
            nextstate21 = storetraj2[index1,t+1]
            
            state2taken= np.unique(state2t) 
            for s2 in state2taken:
                index2 = np.where(state2t == s2)[0]
                action2 = actions1[index2]
                action22 = actions21[index2]
                rewards2 = rewards1[index2]
                rewards22 = rewards21[index2]
                nextstate2 = nextstate1[index2]
                nextstate22 = nextstate21[index2]
                
                action1taken = np.unique(action2)
                
                for a1 in action1taken:
                    index3 = np.where(action2==a1)[0]
                    action23 = action22[index3]
                    rewards3 = rewards2[index3]
                    rewards23 = rewards22[index3]
                    nextstate3 = nextstate2[index3]
                    nextstate23 = nextstate22[index3]
                    
                    action2taken = np.unique(action23)
                    
                    for a2 in action2taken:
                        index4 = np.where(action23==a2)[0]
                        rewards4 = rewards3[index4]
                        rewards24 = rewards23[index4]
                        nextstate4 = nextstate3[index4]
                        nextstate24 = nextstate23[index4]
                        
                        define = rewards4*0.17247 + rewards24*0.193 + nextstate4 *0.2314 + nextstate24*0.27
                        
                        possibilities,frequency = np.unique(define,return_counts=True)
                        
                        for i in range(len(possibilities)):
                            d = possibilities[i]
                            index5 = np.where(define == d)[0]
                            state1 = nextstate4[index5][0]
                            state2 = nextstate24[index5][0]
                            temp1 = rewards4[index5][0]
                            temp2 = rewards24[index5][0]

                                
                            if temp1 == 1:
                                rewardindex1 = 0
                            elif temp1 == 0:
                                rewardindex1 = 1
                            elif temp1 == -1:
                                rewardindex1 = 2
                            elif temp1 == -2:
                                rewardindex1 = 3
                            elif temp1 == -3:
                                rewardindex1 = 4
                            elif temp1 == -10:
                                rewardindex1 = 5
                                
                            if temp2 == 1:
                                rewardindex2 = 0
                            elif temp2 == 0:
                                rewardindex2 = 1
                            elif temp2 == -1:
                                rewardindex2 = 2
                            elif temp2 == -2:
                                rewardindex2 = 3
                            elif temp2 == -3:
                                rewardindex2 = 4
                            elif temp2 == -10:
                                rewardindex2 = 5
                            
                            if t == 0:
                                newmodelM0[int(state1),int(state2),int(rewardindex1),int(rewardindex2),int(s1),int(s2),int(a1),int(a2)] = (frequency[i]/np.sum(frequency))
                            elif t ==1:
                                newmodelM1[int(state1),int(state2),int(rewardindex1),int(rewardindex2),int(s1),int(s2),int(a1),int(a2)] = (frequency[i]/np.sum(frequency))
                            elif t ==2:
                                newmodelM2[int(state1),int(state2),int(rewardindex1),int(rewardindex2),int(s1),int(s2),int(a1),int(a2)] = (frequency[i]/np.sum(frequency))
                            elif t ==3:
                                newmodelM3[int(state1),int(state2),int(rewardindex1),int(rewardindex2),int(s1),int(s2),int(a1),int(a2)] = (frequency[i]/np.sum(frequency))
                            elif t ==4:
                                newmodelM4[int(state1),int(state2),int(rewardindex1),int(rewardindex2),int(s1),int(s2),int(a1),int(a2)] = (frequency[i]/np.sum(frequency))
                            elif t ==T-1:
                                newmodelMT[int(state1),int(state2),int(rewardindex1),int(rewardindex2),int(s1),int(s2),int(a1),int(a2)] = (frequency[i]/np.sum(frequency))

    
    
    print('trajectories completed')
    
    summodelM = newmodelM0+newmodelM1+newmodelM2+newmodelM3+newmodelM4
    
    
    #normalize M tensors as they represent probabilities.
    for s1 in range(noS):
        for s2 in range(noS):
            for a1 in range(2):
                for a2 in range(2):
                    normalize = np.sum(summodelM[:,:,:,:,s1,s2,a1,a2])
                    if normalize != 0:
                        summodelM[:,:,:,:,s1,s2,a1,a2] =summodelM[:,:,:,:,s1,s2,a1,a2]/normalize
                        modelM[:,:,:,:,s1,s2,a1,a2] = modelM[:,:,:,:,s1,s2,a1,a2]+ lrate*(summodelM[:,:,:,:,s1,s2,a1,a2] - modelM[:,:,:,:,s1,s2,a1,a2])
                    #else:
                        #summodelM[:,:,:,:,s1,s2,a1,a2] = 1/(noS*noS*6*6)
                            
   
    #modelM = modelM + lrate*(summodelM - modelM)
    
    #normalize MT
    for s1 in range(noS):
        for s2 in range(noS):
            for a1 in range(2):
                for a2 in range(2):         
                    normalizeT = np.sum(newmodelMT[:,:,:,:,s1,s2,a1,a2])
                    if normalizeT != 0 :
                        newmodelMT[:,:,:,:,s1,s2,a1,a2] =newmodelMT[:,:,:,:,s1,s2,a1,a2]/normalizeT
                        modelMT[:,:,:,:,s1,s2,a1,a2] = modelMT[:,:,:,:,s1,s2,a1,a2]+ lrate*(newmodelMT[:,:,:,:,s1,s2,a1,a2] - modelMT[:,:,:,:,s1,s2,a1,a2])
                    #else:
                        #newmodelMT[:,:,:,:,s1,s2,a1,a2] = 1/(noS*6*noS*6)
                        
   # modelMT = modelMT + lrate*(newmodelMT - modelMT)
    
                        
    return modelM, modelMT


def plottrajectory(policy,T,p0,M,MT,copy2,exploration):
    storetraj1, storetraj2,storeaction1, storeaction2,storerewards1,storerewards2 = generatetrajectoryarray(policy,T,p0,M,MT,copy2,exploration)
    plt.plot(range(T+1),storetraj1-T)
    plt.plot(range(T+1),storetraj2-T)
    plt.plot(range(-1,T+2),np.zeros(T+3),'r:')
    plt.xlim(0,T)
    plt.ylim(-T,T)
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.xticks(np.arange(0,T+1))
    plt.yticks(np.arange(-T,T))
    plt.legend(['Agent 1', 'Agent 2'])
    plt.grid()



def generatetrajectoryarray(policy,T,p0,M,MT,copy2,exploration):
   state = p0
   s1,s2 = np.where(state==1)
   shist1=np.zeros(T+1)
   shist1[0] = s1
   shist2=np.zeros(T+1)
   shist2[0] = s2
   ahist1 = np.zeros(T)
   rhist1 = np.zeros(T)
   ahist2 = np.zeros(T)
   rhist2 = np.zeros(T)
   
   for t in range(T-1):
       #get action
       actions = np.tensordot(state,policy[t],((0,1),(0,1)))
       
       a = np.random.choice([0,1,2,3],p = actions.flatten())
       
       exp1 = np.random.rand(1)
       exp2 = np.random.rand(1)
       if exp1 <exploration:
           if a == 0:
               a=1
           elif a == 1:
               a=0
           elif a == 2:
               a=3
           elif a == 3:
               a=2
       
       if exp2 <exploration:
           if a == 0:
               a=2
           elif a == 2:
               a=0
           elif a == 1:
               a=3
           elif a == 3:
               a=1
               
       
       action = np.zeros(4)
       action[a] = 1
       action = np.reshape(action,[2,2])
       
       a = np.where(action==1)
       
       ahist1[t] = a[0]
       ahist2[t] = a[1]
        
        #st1,st2,r1,r2,a1,a2
       temp1 = np.tensordot(state,M,((0,1),(4,5)))
       #st1,st2,r1,r2
       temp1 = np.tensordot(action,temp1,((0,1),(4,5)))
       possible = np.asarray(np.where(temp1>0))
       probs = temp1[temp1>0]
       s = np.random.choice(np.arange(len(possible[0,:])),p = probs)
          
       sinfo = possible[s]
       
       s1 = possible[0,s]
       s2 = possible[1,s]
       r1 = possible[2,s]
       r2 = possible[3,s]
       shist1[t+1] = s1
       shist2[t+1] = s2
       rhist1[t] = rewardvector[r1]
       rhist2[t] = rewardvector[r2]


       state = np.zeros((noS,noS))
       state[s1,s2] = 1
       

   #get action
   actions = np.tensordot(state,policy[T-1],((0,1),(0,1)))
    
   a = np.random.choice([0,1,2,3],p = actions.flatten())
    
   exp1 = np.random.rand(1)
   exp2 = np.random.rand(1)
   if exp1 <exploration:
        if a == 0:
            a=1
        elif a == 1:
            a=0
        elif a == 2:
            a=3
        elif a == 3:
            a=2
    
   if exp2 <exploration:
        if a == 0:
            a=2
        elif a == 2:
            a=0
        elif a == 1:
            a=3
        elif a == 3:
            a=1
            
    
   action = np.zeros(4)
   action[a] = 1
   action = np.reshape(action,[2,2])
    
   a = np.where(action==1)
    
   ahist1[T-1] = a[0]
   ahist2[T-1] = a[1]
     
     #st1,st2,r1,r2,a1,a2
   temp1 = np.tensordot(state,MT,((0,1),(4,5)))
   #st1,st2,r1,r2
   temp1 = np.tensordot(action,temp1,((0,1),(4,5)))
   possible = np.asarray(np.where(temp1>0))
   probs = temp1[temp1>0]
   s = np.random.choice(np.arange(len(possible[0,:])),p = probs)
       
    
   s1 = possible[0,s]
   s2 = possible[1,s]
   r1 = possible[2,s]
   r2 = possible[3,s]
   shist1[T] = s1
   shist2[T] = s2
   rhist1[T-1] = rewardvector[r1]
   rhist2[T-1] = rewardvector[r2]
   return shist1,shist2,ahist1,ahist2,rhist1,rhist2


def plotpolicy(policy,noS,T):
    plt.figure(figsize=(20,5))
    goal = np.ones((1,noS))*0.5
    goal[0,int((noS+1)/2-1)] = -1
    array = np.append(policy[:,:,0],goal,axis=0)
    maskedarray = np.ma.masked_where(array==-1,array)
    
    cmap = matplotlib.cm.RdBu
    cmap.set_bad(color='green')
    
    im = plt.imshow(np.transpose(maskedarray),cmap=cmap,aspect = 1/3,extent = [-0.5,T+0.5,-0.5-(noS+1)/2,noS-0.5-(noS+1)/2],origin='lower')
    ax = plt.gca()
    ax.set_xticks(np.arange(0, T+1, 1));
    ax.set_yticks(np.arange(0-(noS+1)/2, noS-(noS+1)/2, 1));
    ax.set_xticklabels(np.arange(0, T+1, 1));
    ax.set_yticklabels(np.arange(1-(noS+1)/2, noS+1-(noS+1)/2, 1));
    ax.set_xticks(np.arange(-.5, T, 1), minor=True);
    ax.set_yticks(np.arange(-.5-(noS+1)/2, noS-(noS+1)/2, 1), minor=True);
    ax.set_xlabel('T')
    ax.grid(which='minor', linestyle='-', linewidth=1, color = 'k',alpha = 0.5)
    ax.set_ylabel('State')
    ax.plot(np.arange(-1,T+2),np.ones(T+3) *-1.5,'g',linewidth = 3 )
    ax.set_xlim(-0.5,T+0.5)
    
    values = np.unique(array)
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[1],label = "Up"),mpatches.Patch(color=colors[3],label = "Down"),mpatches.Patch(color='green',label = "Goal")]
    plt.legend(handles = patches,bbox_to_anchor = (1.02,1),loc=2,borderaxespad=0.)
    
    plt.show()

def plotexpectedreturn(tstepsNO,expectedreturnmodel,expectedreturn,optimalreturn):
    plt.figure()
    plt.plot(range(tstepsNO+1),expectedreturnmodel)
    plt.plot(range(tstepsNO+1),expectedreturn)
    plt.xlabel('Epoch')
    plt.ylabel('Expected Return')
    plt.plot(np.linspace(-1,tstepsNO+2,100),np.ones(100)*optimalreturn)
    plt.xlim([-1,tstepsNO+1])
    plt.legend(['Expected Return wrt Model','Expected Return wrt true enviro','Optimal Expected Return'])
    plt.xticks(np.arange(0,tstepsNO+1))
    plt.show()




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
    
    prob1,prob2 = choosedistribution('deterministic', 1)
    M = createM(noS,prob1,prob2,rewardvector)
    MT = createMT(noS,prob1,prob2,rewardvector)
    rewardmatrix = createrewardmatrix(rewardvector)
    #W tensors are same for all timesteps apart from timesteps 1 and T
    W = createW(rewardmatrix)
    W1 = createW1(rewardmatrix)
    WT = createWT(rewardmatrix)    
    
    #copy tensor is essentially a 3D identity. 
    copy3 = createcopytensor3(noS)
    copy2 = createcopytensor2(noS) 
    
    #generateinitial guess for model, same prob for everything.
    modelM = np.ones((noS,noS,6,6,noS,noS,2,2))/(noS*6*noS*6)
    modelMT = np.ones((noS,noS,6,6,noS,noS,2,2))/(noS*6*noS*6)

    optimalpolicy = policy.copy()

    optimalpolicy = DRMG(M,p0,optimalpolicy,W1,W,copy2,MT,WT,flatstate,T,flatreward)
    optimalreturn = contracteverything(M, p0, optimalpolicy, W1, W, copy2, MT, WT, flatstate,T)
    
    #now iterate through training model on samples, and optimising policy
    tstepsNO = 10
    Ntraj = 200
    exploration = 0.2
    lrate = 0.8
 
    
   
    expectedreturn = np.zeros(tstepsNO+1)
    expectedreturnmodel = np.zeros(tstepsNO+1)
    expectedreturn[0] = contracteverything(M, p0, policy, W1, W, copy2, MT, WT, flatstate,T)
    expectedreturnmodel[0] = contracteverything(modelM, p0, policy, W1, W, copy2, modelMT, WT, flatstate,T)
    merror = np.zeros(tstepsNO+1)
    merror[0] = np.sum(np.absolute(M - modelM))
    
    savepolicy = np.zeros((tstepsNO+1,T,noS,noS,2,2))
    savepolicy[0] = policy
    
    
    for timestep in range(tstepsNO):
        print('timestep = ' + str(timestep))
        modelM,modelMT = modellearning(T,policy,p0,M,MT,copy2,rewardvector,modelM,modelMT,Ntraj,exploration,lrate,flatstate,noS)
        policy = DRMG(modelM,p0,policy,W1,W,copy2,modelMT,WT,flatstate,T,flatreward)
        policy[1,7,5] = optimalpolicy[1,7,5] 
        expectedreturn[timestep+1] = contracteverything(M, p0, policy, W1, W, copy2, MT, WT, flatstate,T)
        expectedreturnmodel[timestep+1] = contracteverything(modelM, p0, policy, W1, W, copy2, modelMT, WT, flatstate,T)
        merror[timestep+1] = np.sum(np.absolute(M - modelM))
        savepolicy[timestep+1] = policy.copy()
        
    plotexpectedreturn(tstepsNO, expectedreturnmodel, expectedreturn,optimalreturn)
    plotgreedy(policy,T,p0,modelM,modelMT,copy2,noS)
    
 

