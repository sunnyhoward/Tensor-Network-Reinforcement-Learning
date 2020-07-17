
from FinalTNMARL import (createflatstate, initializepolicy, createM1,createM2,generatetrajectoryarray,createMT,
                                createW,createW1,createWT,createrewardmatrix,
                                contracteverything,DRMGnew,choosedistribution,createcopytensor3,createcopytensor2)

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches


#function for generating trajectories and updating model.
def modellearning(T,policy1,policy2,p10,p20,M1,M2,MT,copy2,rewardvector,modelM1,modelM2,modelMT,Ntraj,exploration,lrate,flatstate,noS):
    #generate the trajectories
    storetraj1 = np.zeros((Ntraj,T+1))
    storetraj2 = np.zeros((Ntraj,T+1))

    storeaction1 = np.zeros((Ntraj,T))
    storeaction2 = np.zeros((Ntraj,T))

    storerewards1 = np.zeros((Ntraj,T))
    storerewards2 = np.zeros((Ntraj,T))

    for i in range(Ntraj):
        storetraj1[i], storetraj2[i],storeaction1[i], storeaction2[i],storerewards1[i],storerewards2[i] = generatetrajectoryarray(policy1,policy2,T,p10,p20,exploration,M1,M2,MT,copy2,rewardvector)
    
    newmodelM1 = np.zeros((noS,noS,noS,2,6,T)) 
    newmodelM2 = np.zeros((noS,noS,noS,2,6,T)) #oldstate,newstate,action,reward,time

    for t in range(T):
        for s1 in range(noS):
            index1 = np.where(storetraj1[:,t] == s1)[0]
            actions1 = storeaction1[index1,t]

            rewards1 = storerewards1[index1,t]
            
            nextstate1 = storetraj1[index1,t+1]      
            agent2state1 = storetraj2[index1,t+1]
            
            actionstaken= np.unique(actions1) 
            for a in actionstaken:
                index2 = np.where(actions1 == a)[0]
                rewards2 = rewards1[index2]
                nextstate2 = nextstate1[index2]
                agent2state2 = agent2state1[index2]
                possibleagent2state = np.unique(agent2state2)
                for s2 in possibleagent2state:
                    index3 = np.where(agent2state2 == s2)[0]
                    rewards3 = rewards2[index3]
                    nextstate3 = nextstate2[index3]
                    
                    nextstates,frequencystate = np.unique(nextstate3,return_counts=True)
                    reward,frequencyreward = np.unique(rewards3,return_counts=True)
                    for i in range(len(nextstates)):
                        snew = nextstates[i]
                        for j in range(len(reward)):
                            temp = reward[j]
                            
                            if temp == 1:
                                rewardindex1 = 0
                            elif temp == 0:
                                rewardindex1 = 1
                            elif temp == -1:
                                rewardindex1 = 2
                            elif temp == -2:
                                rewardindex1 = 3
                            elif temp == -3:
                                rewardindex1 = 4
                            elif temp == -10:
                                rewardindex1 = 5
                                
                            newmodelM1[s1,int(snew),int(s2),int(a),rewardindex1,t] = (frequencystate[i]/np.sum(frequencystate)) * (frequencyreward[i]/np.sum(frequencyreward))
    #agent2
    for t in range(T):
        for s1 in range(noS):
            index1 = np.where(storetraj2[:,t] == s1)[0]
            actions1 = storeaction2[index1,t]

            rewards1 = storerewards2[index1,t]
            
            nextstate1 = storetraj2[index1,t+1]      
            agent2state1 = storetraj1[index1,t+1]
            
            actionstaken= np.unique(actions1) 
            for a in actionstaken:
                index2 = np.where(actions1 == a)[0]
                rewards2 = rewards1[index2]
                nextstate2 = nextstate1[index2]
                agent2state2 = agent2state1[index2]
                possibleagent2state = np.unique(agent2state2)
                for s2 in possibleagent2state:
                    index3 = np.where(agent2state2 == s2)[0]
                    rewards3 = rewards2[index3]
                    nextstate3 = nextstate2[index3]
                    
                    nextstates,frequencystate = np.unique(nextstate3,return_counts=True)
                    reward,frequencyreward = np.unique(rewards3,return_counts=True)
                    for i in range(len(nextstates)):
                        snew = nextstates[i]
                        for j in range(len(reward)):
                            temp = reward[j]
                            
                            if temp == 1:
                                rewardindex1 = 0
                            elif temp == 0:
                                rewardindex1 = 1
                            elif temp == -1:
                                rewardindex1 = 2
                            elif temp == -2:
                                rewardindex1 = 3
                            elif temp == -3:
                                rewardindex1 = 4
                            elif temp == -10:
                                rewardindex1 = 5
                                
                            newmodelM2[s1,int(snew),int(s2),int(a),rewardindex1,t] = (frequencystate[i]/np.sum(frequencystate)) * (frequencyreward[i]/np.sum(frequencyreward))
    
    summodelM1 = 0
    summodelM2 = 0
    for i in range(T-1):
        summodelM1 += newmodelM1[:,:,:,:,:,i]
        summodelM2 += newmodelM2[:,:,:,:,:,i]
    
    newmodelMT = newmodelM1[:,:,:,:,:,T-1] + newmodelM2[:,:,:,:,:,T-1]
    newmodelMT = np.tensordot(newmodelMT,flatstate,((2),(0)))
    
    #normalize M tensors as they represent probabilities.
    for s in range(noS):
        for s2 in range(noS):
            for a in range(2):
                normalize1 = np.sum(summodelM1[s,:,s2,a,:])
                normalize2 = np.sum(summodelM2[s2,:,s,a,:])
                for sdash in range(noS):
                    for r in range(6):
                        if normalize1 != 0:
                            summodelM1[s,sdash,s2,a,r] =summodelM1[s,sdash,s2,a,r]/normalize1
                        else:
                            summodelM1[s,sdash,s2,a,r] = 1/(noS*6)
                        if normalize2 != 0:
                            summodelM2[s2,sdash,s,a,r] =summodelM2[s2,sdash,s,a,r]/normalize2
                        else:
                            summodelM2[s2,sdash,s,a,r] = 1/(noS*6)
   
    modelM1 = modelM1 + lrate*(summodelM1 - modelM1)
    modelM2 = modelM2 + lrate*(summodelM2 - modelM2)
    
    #normalize MT
    for s in range(noS):
        for a in range(2):         
            normalizeT = np.sum(newmodelMT[s,:,a,:])
            for sdash in range(noS):
                for r in range(6):
                    if normalizeT != 0 :
                        newmodelMT[s,sdash,a,r] =newmodelMT[s,sdash,a,r]/normalizeT
                    else:
                        newmodelMT[s,sdash,a,r] = 1/(noS*6)
                        
    modelMT = modelMT + lrate*(newmodelMT - modelMT)
    
                        
    return modelM1,modelM2, modelMT


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


def fixmodelM(modelM1,modelM2,modelMT,M1,M2,MT,noS):
    fixedM1 = modelM1.copy()
    fixedM2 = modelM2.copy()
    fixedMT = modelMT.copy()
    for i in range(noS):
        for j in range(noS):
            for k in range(2):
                if len(np.unique(modelM1[i,:,j,k,:])) == 1:
                    fixedM1[i,:,j,k,:] = M1[i,:,j,k,:]
                if len(np.unique(modelM2[i,:,j,k,:])) == 1:
                    fixedM2[i,:,j,k,:] = M2[i,:,j,k,:]
    
    for i in range(noS):
        for k in range(2):
            if len(np.unique(modelMT[i,:,k,:])) == 1:
                fixedMT[i,:,k,:] = MT[i,:,k,:]
    return fixedM1,fixedM2,fixedMT


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
    
    #generateinitial guess for model, same prob for everything.
    modelM1 = np.ones((noS,noS,noS,2,6))/(noS*6)
    modelM2 = np.ones((noS,noS,noS,2,6))/(noS*6)
    modelMT = np.ones((noS,noS,2,6))/(noS*6)

    optimalpolicy1 = policy1.copy()
    optimalpolicy2 = policy2.copy()

    optimalpolicy1,optimalpolicy2 = DRMGnew(flatreward,optimalpolicy1,optimalpolicy2,M1,M2,W,W1,p10,p20,copy3,copy2,MT,WT,flatstate,T,N=1)
    optimalreturn = contracteverything(M1,M2,MT,W,W1,WT,flatreward,optimalpolicy1,optimalpolicy2,p10,p20,flatstate,T,copy3,copy2,False)
    
    #now iterate through training model on samples, and optimising policy
    tstepsNO = 14
    Ntraj = 1000
    exploration = 0.2
    learningrate = 0.4
    Merror = np.zeros(tstepsNO+1)
    Merror[0] = np.sum(np.absolute(modelM1-M1))  
    
    saveM = np.zeros((noS,noS,noS,2,6,tstepsNO+1))
    saveM[:,:,:,:,:,0] = modelM1
    expectedreturnfix = np.zeros(tstepsNO)
    expectedreturn = np.zeros(tstepsNO+1)
    expectedreturnmodel = np.zeros(tstepsNO+1)
    expectedreturn[0] = contracteverything(M1,M2,MT,W,W1,WT,flatreward,policy1,policy2,p10,p20,flatstate,T,copy3,copy2,False)
    expectedreturnmodel[0] = contracteverything(modelM1,modelM2,modelMT,W,W1,WT,flatreward,policy1,policy2,p10,p20,flatstate,T,copy3,copy2,False)

    for timestep in range(tstepsNO):
        print('timestep = ' + str(timestep))
        modelM1,modelM2,modelMT = modellearning(T,policy1,policy2,p10,p20,M1,M2,MT,copy2,rewardvector,modelM1,modelM2,modelMT,Ntraj,exploration,learningrate,flatstate,noS)
        policy1,policy2 = DRMGnew(flatreward,policy1,policy2,modelM1,modelM2,W,W1,p10,p20,copy3,copy2,modelMT,WT,flatstate,T,N=1)
        expectedreturn[timestep+1] = contracteverything(M1,M2,MT,W,W1,WT,flatreward,policy1,policy2,p10,p20,flatstate,T,copy3,copy2,False)
        expectedreturnmodel[timestep+1] = contracteverything(modelM1,modelM2,modelMT,W,W1,WT,flatreward,policy1,policy2,p10,p20,flatstate,T,copy3,copy2,False)
        Merror[timestep+1] = np.sum(np.absolute(modelM1-M1))
        saveM[:,:,:,:,:,timestep+1] = modelM1
        fixedM1,fixedM2,fixedMT = fixmodelM(modelM1,modelM2,modelMT,M1,M2,MT,noS)
        expectedreturnfix[timestep] = contracteverything(fixedM1,fixedM2,fixedMT,W,W1,WT,flatreward,policy1,policy2,p10,p20,flatstate,T,copy3,copy2,False)
        
    #plotexpectedreturn(tstepsNO, expectedreturnmodel, expectedreturn,optimalreturn)
    

    plt.plot(range(tstepsNO+1),expectedreturnmodel)
    plt.plot(range(tstepsNO+1),expectedreturn)
    plt.xlabel('Epoch')
    plt.ylabel('Expected Return')
    plt.plot(np.linspace(-1,tstepsNO+2,100),np.ones(100)*optimalreturn)
    plt.plot(range(tstepsNO),expectedreturnfix)
    plt.xlim([-1,tstepsNO+1])
    plt.legend(['Expected Return wrt Model','Expected Return wrt true enviro','Optimal Expected Return','Fixed Expected Return'])
    plt.xticks(np.arange(0,tstepsNO+1))
    plt.show()
                



# plt.figure()
# plt.plot(range(tstepsNO+1),saveM[6,8,7,1,4,:])
# plt.plot(range(tstepsNO+1),saveM[6,8,7,1,4,:])
# plt.plot(range(tstepsNO+1),saveM[6,8,7,1,4,:])
# plt.plot(range(tstepsNO+1),saveM[6,8,7,1,4,:])
# plt.plot(range(tstepsNO+1),M1[6,8,7,1,4]*np.ones(tstepsNO+1))






# plt.xlabel('Epoch')
# plt.ylabel('Expected Return wrt Model')
# plt.plot(np.linspace(-1,tstepsNO+2,100),np.ones(100),'r')
# plt.xlim([-1,tstepsNO+1])
# plt.legend([r'$N_{traj} = 2$',r'$N_{traj} = 5$',r'$N_{traj} = 10$',r'$N_{traj} = 50$',r'$N_{traj} = 100$',r'$N_{traj} = 300$',r'$N_{traj} = 1000$'])
# plt.xticks(np.arange(0,tstepsNO+1))
# plt.show()
