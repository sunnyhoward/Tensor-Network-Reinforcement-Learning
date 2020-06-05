# from randomwalkertensor import (createflatstate, initializepolicy, createM,createMT,
#                                 createW,createW1,createWT,createcopytensor,
#                                 contracteverything,DRMGpolicyoptimisation)
from randomwalkertensornoiseboth import (createflatstate, initializepolicy, createM,createMT,
                                createW,createW1,createWT,createcopytensor,
                                contracteverything,DRMGpolicyoptimisation, choosedistribution)
# from randomwalkertensornoiseup import (createflatstate, initializepolicy, createM,createMT,
#                                 createW,createW1,createWT,createcopytensor,
#                                 contracteverything,DRMGpolicyoptimisation)
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

def generatetrajectoryarray(policy,T,p0,exploration,M,MT):
   state = p0
   s = np.where(state==1)[0]
   shist=np.zeros(T+1)
   shist[0] = s
   ahist = np.zeros(T)
   rhist = np.zeros(T)
   
   for t in range(T-1):
       #get action
       #action = np.argmax(np.tensordot(state,policy[t],((0),(0))))
       if np.tensordot(state,policy[t],((0),(0)))[1] > np.random.rand(1):
           action = np.tensordot(state,policy[t],((0),(0)))
       else:
           action = 1 - np.tensordot(state,policy[t],((0),(0)))
           
       if np.random.rand(1)<exploration:
           action = 1-action
          
       ahist[t] = np.argmax(action)
       
       c1 = np.tensordot(state,M,((0),(0)))
       c2 = np.tensordot(c1,action,((1),(0)))
       stateprobs = np.sum(c2,axis=1)#
       possiblestates = np.where(stateprobs>0)[0]
       stateprobs = stateprobs[stateprobs>0]
       s = np.random.choice(possiblestates,p = stateprobs)
       
       state = np.zeros(2*T+1)
       state[s] = 1
       shist[t+1] = s
       if s<(len(p0)+1)/2-1:
               rhist[t] = -1
   #now do final timestep
   if np.tensordot(state,policy[T-1],((0),(0)))[1] > np.random.rand(1):
       action = np.tensordot(state,policy[T-1],((0),(0)))
   else:
       action = 1 - np.tensordot(state,policy[T-1],((0),(0)))
           
   if np.random.rand(1)<exploration:
       action = 1-action
      
   ahist[T-1] = np.argmax(action)
   
   c1 = np.tensordot(state,MT,((0),(0)))
   c2 = np.tensordot(c1,action,((1),(0)))
   stateprobs = np.sum(c2,axis=1)#
   possiblestates = np.where(stateprobs>0)[0]
   stateprobs = stateprobs[stateprobs>0]
   s = np.random.choice(possiblestates,p = stateprobs)
   
   state = np.zeros(2*T+1)
   state[s] = 1
   shist[T] = s
   if s==(len(p0)+1)/2-1:
       rhist[T-1] = 1
   else:
       rhist[T-1] = -10
               
   return shist,ahist,rhist


#function for generating trajectories and updating model.
def modellearning(T,policy,p0,noS,modelM,modelMT,Ntraj,exploration,lrate,M,MT):
    #generate the trajectories
    storetraj = np.zeros((Ntraj,T+1))
    storeaction = np.zeros((Ntraj,T))
    storerewards = np.zeros((Ntraj,T))
    for i in range(Ntraj):
        storetraj[i], storeaction[i], storerewards[i] = generatetrajectoryarray(policy, T, p0,exploration,M,MT)
    
    newmodelM = np.zeros((noS,noS,2,4,T)) #oldstate,newstate,action,reward,time
    for t in range(T):
        for s1 in range(noS): 
            occur = np.where(storetraj[:,t] == s1)[0] #find index of where value occured.
            sdash = storetraj[np.transpose(occur),t+1]
            reward = storerewards[np.transpose(occur),t]
            actions = storeaction[np.transpose(occur),t]
            nextstates,frequencystate = np.unique(sdash,return_counts=True)
            for i in range(len(nextstates)):
                s2 = nextstates[i]
                temp = reward[np.where(sdash==int(s2))[0][0]]
                if temp == 1:
                    rewardindex = 0
                elif temp == 0:
                    rewardindex = 1
                elif temp == -1:
                    rewardindex = 2
                elif temp == -10:
                    rewardindex = 3
                possibleactions = actions[np.where(sdash==int(s2))[0]]
                actionstaken,frequencyaction = np.unique(possibleactions,return_counts=True) 
                for j in range(len(actionstaken)):
                    a=actionstaken[j]
                    newmodelM[s1,int(s2),int(a),rewardindex,t] = (frequencyaction[j]/np.sum(frequencyaction))#*(frequencystate[i]/np.sum(frequencystate))
    
    
    summodelM = 0
    for i in range(T-1):
        summodelM += newmodelM[:,:,:,:,i]
    
    newmodelMT = newmodelM[:,:,:,:,T-1]
    
    #normalize M tensors as they represent probabilities.
    for s in range(noS):
        for a in range(2):
            normalize = np.sum(summodelM[s,:,a,:])
            normalizeT = np.sum(newmodelMT[s,:,a,:])
            for sdash in range(noS):
                for r in range(4):
                    if normalize != 0:
                        summodelM[s,sdash,a,r] =summodelM[s,sdash,a,r]/normalize
                        modelM[s,sdash,a,r] = modelM[s,sdash,a,r] + lrate*(summodelM[s,sdash,a,r]-modelM[s,sdash,a,r])
                    if normalizeT != 0 :
                        newmodelMT[s,sdash,a,r] =newmodelMT[s,sdash,a,r]/normalizeT
                        modelMT[s,sdash,a,r] = modelMT[s,sdash,a,r] + lrate*(newmodelMT[s,sdash,a,r]-modelMT[s,sdash,a,r])

    
                    
    return modelM, modelMT


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
    plt.ylabel('Expected Return wrt Model')
    plt.plot(np.linspace(-1,tstepsNO+2,100),np.ones(100)*optimalreturn)
    plt.xlim([-1,tstepsNO+1])
    plt.legend(['Expected Return wrt Model','Expected Return wrt true enviro','Optimal Expected Return'])
    plt.xticks(np.arange(0,tstepsNO+1))
    plt.show()

def generatetrajectoryplot(policy,T,p0,M):
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
          state[s+1] = 1
           
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


if __name__ == '__main__':
    
    #can choose value of T
    T = 10
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

    prob1,prob2 = choosedistribution('normal', 1)

    #M tensors are same for all time steps apart from last (MT)
    M=createM(noS,prob1,prob2)
    MT=createMT(noS,prob1,prob2)
    
    #W tensors are same for all timesteps apart from timesteps 1 and T
    W = createW(rewardvector)
    W1 = createW1(rewardvector)
    WT = createWT(rewardvector)    
    
    #copy tensor is essentially a 3D identity. 
    copy = createcopytensor(noS)
    # contracteverything(M,MT,W,W1,WT,flatreward,policy,p0,flatstate,T,copy,False)
    
    #generateinitial guess for model, same prob for everything.
    modelM = np.ones((noS,noS,2,4))/(noS*4)
    modelMT = np.ones((noS,noS,2,4))/(noS*4)

    #also calculate the optimal expected return 
    optimalpolicy = policy.copy()
    optimalpolicy = DRMGpolicyoptimisation(flatreward, optimalpolicy, M, W, W1, p0, copy, MT, WT, flatstate, T)
    optimalreturn = contracteverything(M, MT, W, W1, WT, flatreward, optimalpolicy, p0, flatstate, T, copy, False)
    plotpolicy(optimalpolicy,noS,T)
    
    #now iterate through training model on samples, and optimising policy
    tstepsNO = 15
    Ntraj = 1000
    exploration = 0.2
    learningrate = 0.5
    
    expectedreturn = np.zeros(tstepsNO+1)
    expectedreturnmodel = np.zeros(tstepsNO+1)
    expectedreturn[0] = contracteverything(M,MT,W,W1,WT,flatreward,policy,p0,flatstate,T,copy,False)
    expectedreturnmodel[0] = contracteverything(modelM,modelMT,W,W1,WT,flatreward,policy,p0,flatstate,T,copy,False)

    for timestep in range(tstepsNO):
        print('timestep = ' + str(timestep))
        modelM,modelMT = modellearning(T, policy, p0, noS, modelM, modelMT,Ntraj,exploration,learningrate,M,MT)
        policy = DRMGpolicyoptimisation(flatreward, policy, modelM, W, W1, p0, copy, modelMT, WT, flatstate, T)
        plt.figure()
        expectedreturn[timestep+1] = contracteverything(M,MT,W,W1,WT,flatreward,policy,p0,flatstate,T,copy,False)
        expectedreturnmodel[timestep+1] = contracteverything(modelM,modelMT,W,W1,WT,flatreward,policy,p0,flatstate,T,copy,False)
        plotpolicy(policy,noS,T)

    plotexpectedreturn(tstepsNO, expectedreturnmodel, expectedreturn,optimalreturn)
    
    #plotpolicy(policy, noS, T)


# plt.figure()
# plt.plot(range(tstepsNO+1),erntraj2lr5)
# plt.plot(range(tstepsNO+1),erntraj5lr5)
# plt.plot(range(tstepsNO+1),erntraj10lr5)
# plt.plot(range(tstepsNO+1),erntraj50lr5)
# plt.plot(range(tstepsNO+1),erntraj100lr5)
# plt.plot(range(tstepsNO+1),erntraj300lr5)
# plt.plot(range(tstepsNO+1),expectedreturnmodel)



# plt.xlabel('Epoch')
# plt.ylabel('Expected Return wrt Model')
# plt.plot(np.linspace(-1,tstepsNO+2,100),np.ones(100),'r')
# plt.xlim([-1,tstepsNO+1])
# plt.legend([r'$N_{traj} = 2$',r'$N_{traj} = 5$',r'$N_{traj} = 10$',r'$N_{traj} = 50$',r'$N_{traj} = 100$',r'$N_{traj} = 300$',r'$N_{traj} = 1000$'])
# plt.xticks(np.arange(0,tstepsNO+1))
# plt.show()
