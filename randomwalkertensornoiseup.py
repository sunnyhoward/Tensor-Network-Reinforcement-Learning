import numpy as np
import matplotlib.pyplot as plt

#This implementation includes the noise when going up. When going down nothing
# has changed, but when going up 68% chance of going up 1, 16% of 2 or staying same

#function to create the initial policy
def initializepolicy(T,noS):
    # policy = np.random.rand(T,noS,1) #the (noS+1)/2 element is 0
    # #policy = policy[:,:,:]>0.5
    # policy = np.append(policy,1-policy,axis=2)
    # return policy
    policy = np.ones((T,noS,2)) * 0.5
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
   plt.plot(range(T+1),shist-T)
   plt.plot(range(-1,T+2),np.zeros(T+3),'r:')
   plt.xlim(0,T)
   plt.ylim(-T,T)
   plt.xlabel('Time')
   plt.ylabel('Position')
   plt.xticks(np.arange(0,T+1))
   plt.yticks(np.arange(-T,T))
   plt.grid()
    

#function to create the rank 4 tensor. 
def createM(noS):
    M = np.zeros((noS,noS,2,4)) 
    
    # M[St-1,St,a,r]
    #indices for start state, end state, actions and rewards.
    #possible values for action are down (0) and up (1), 
    #possible values for reward are 1, 0, -1, -10
    
    #note that this M tensor will not be the one at t = T, so that there is a 0 chance
    #of recieving a reward of -10 or 1 for all indices according to rules in section
    #5.1 of Ed's paper.
    
    #the tensor is filled using the update rules described in the paper.
    
    #the ((noS+1)/2)-1 th indicy signifys a position of s = 0.
    
    #first will update the M tensor so that if the initial state is 0 or below, and we take
    #an action to go down then there is a prob of 1 of getting reward  = -1 
    for x in range(1,int((noS+1)/2)):
        M[x,x-1,0,2] = 1
        
    #similarly, if we go up from anything below -2, we also will recieve a negative reward.
    for x in range(0,int((noS+1)/2-3)):
        M[x,x+1,1,2] = 0.68
        M[x,x+2,1,2] = 0.16
        M[x,x,1,2] = 0.16
        
    #if in state -2, if we go up 2 we get reward = 0
    M[int((noS+1)/2 -3),int((noS+1)/2 -2),1,2]=0.68
    M[int((noS+1)/2 -3),int((noS+1)/2 -1),1,1]=0.16
    M[int((noS+1)/2 -3),int((noS+1)/2 -3),1,2]=0.16
    
    


    #now update that when 0 or above, going up will give reward of 0 
    for x in range(int((noS+1)/2-1),noS-2):
        M[x,x+1,1,1] = 0.68
        M[x,x+2,1,1] = 0.16
        M[x,x,1,1] = 0.16
    
    #if instate -1, going up 1 or 2 gives 0 reward, staying same gives -1 reward
    M[int((noS+1)/2 -2),int((noS+1)/2-1),1,1]=0.68
    M[int((noS+1)/2 -2),int((noS+1)/2),1,1]=0.16 
    M[int((noS+1)/2 -2),int((noS+1)/2 -2),1,2]=0.16    
    
    #now update that when above 0, going down will give reward of 0 
    for x in range(int((noS+1)/2),noS):
        M[x,x-1,0,1] = 1
    
    #Finally lets ammend the top 2 states so that we cant jump out of cell.
    M[int(noS-2),:,:,:] = 0
    M[int(noS-1),:,:,:] = 0
    M[int(noS-2),int(noS-2),1,1] = 0.32 #if jump up from 2nd top cell then stay same
    M[int(noS-2),int(noS-1),1,1] = 0.68 #if jump up from 2nd top cell then go up 1
    M[int(noS-2),int(noS-3),0,1] = 1 #if go down from 2nd top cell then go down 
    M[int(noS-1),int(noS-1),1,1] = 1 #if jump up from top cell then stay same
    M[int(noS-1),int(noS-2),0,1] = 1 #if jump down from  top cell then go down
    return M
    
    
#function to create the end of M tensor for the last time step. 
def createMT(noS):
    MT = np.zeros((noS,noS,2,4)) 
    
    #in this case, we simply get a -10 reward for not being at position 0, at T
    #and a 1 reward for being there. 
    
    # M[St-1,St,a,r]
    #possible values for reward are 1, 0, -1, -10
    
    #if we go down from any position between 2 to the max then we recieve -10 reward 
    for x in range(int((noS+1)/2)+1,noS):
        MT[x,x-1,0,3] = 1
     
    #if we go down from 0 or below we will recieve -10 reward.
    for x in range(1,int((noS+1)/2)):
        MT[x,x-1,0,3] = 1    
        
    #similarly, if we go up from 1 or above we will recieve -10 reward.
    for x in range(int((noS+1)/2),noS-2):
        MT[x,x+1,1,3] = 0.68
        MT[x,x+2,1,3] = 0.16
        MT[x,x,1,3] = 0.16
    
    #going up from zero can score you either +1 or -10
    MT[int((noS+1)/2)-1,int((noS+1)/2)-1,1,0] = 0.16
    MT[int((noS+1)/2)-1,int((noS+1)/2),1,3] = 0.68
    MT[int((noS+1)/2)-1,int((noS+1)/2)+1,1,3] = 0.16
    
    #If we go up or down from highest state we score -10
    MT[int(noS-1),int(noS-1),1,3] = 1
    MT[int(noS-1),int(noS-2),0,3] = 1
    
    #if we go up from second highest state we score -10
    MT[int(noS-2),int(noS-2),1,3] = 0.32
    MT[int(noS-2),int(noS-1),1,3] = 0.68

    
    #if we go up from a state between the min and -3 we will recieve -10 reward.
    for x in range(0,int((noS+1)/2-3)):
        MT[x,x+1,1,3] = 0.68
        MT[x,x+2,1,3] = 0.16
        MT[x,x,1,3] = 0.16
        
    #going up from -2 can yield 2 possibiliies
    MT[int((noS+1)/2-3),int((noS+1)/2-1),1,0] = 0.16
    MT[int((noS+1)/2-3),int((noS+1)/2-2),1,3] = 0.68
    MT[int((noS+1)/2-3),int((noS+1)/2-3),1,3] = 0.16

    
    
    #finally, if we go down from 1 we recieve a +1 reward. 
    MT[int((noS+1)/2),int((noS+1)/2-1),0,0] = 1
    
    #if we go up from -1 then a few things can happen
    MT[int((noS+1)/2-2),int((noS+1)/2-1),1,0] = 0.68
    MT[int((noS+1)/2-2),int((noS+1)/2-2),1,3] = 0.16
    MT[int((noS+1)/2-2),int((noS+1)/2),1,3] = 0.16

    
    return MT
        
#this code creates the flattensor for states |-_{s}>
def createflatstate(T):
    # flatstate = np.asarray([])
    # for n in range(2*T+1):
    #     flatstate = np.append(flatstate,n-T)
    flatstate = np.ones(2*T+1)
    return flatstate

def createW(rewardarray):
    W = np.zeros((2,2,4,4))
    #create identity operator matrices and reward operator matrix.
    for i in range(4):
        W[0,0,i,i] = 1
        W[1,1,i,i] = 1
        W[0,1,i,i] = rewardarray[i]
    return W

def createW1(rewardarray):
    W1 = np.zeros((2,4,4))
    #create identity operator matrices and reward operator matrix.
    for i in range(4):
        W1[0,i,i] = 1
        W1[1,i,i] = rewardarray[i]
    return W1

def createWT(rewardarray):
    WT = np.zeros((2,4,4))
    #create identity operator matrices and reward operator matrix.
    for i in range(4):
        WT[0,i,i] = rewardarray[i]
        WT[1,i,i] = 1
    return WT

def createcopytensor(noS):
    copy = np.zeros((noS,noS,noS))
    for i in range(noS):
        copy[i,i,i] = 1
    return copy
    

def contractlayer1(flatreward,policy,M,W1,p0):
    #contracting initial prob dist with M1
    #gives something of (end state,action,reward)
    contraction1 = np.tensordot(M,p0,((0),(0)))
    
    
    #find the actions chosen by contracting initial prob with policy pi0
    actions = np.tensordot(p0,policy[0],((0),(0)))
    
    #contract actions with M matrix
    #gives shape (next state, reward)
    contraction2 = np.tensordot(contraction1,actions,((1),(0)))
    
    #contracting W1 
    #gives shape (next state,virtualindex,flatreward)
    contraction3 = np.tensordot(contraction2,W1,((1),(1)))
    
    #contract with flat reward
    #gives shape (next state, virtualindex)
    contraction4 = np.tensordot(contraction3, flatreward,((2),(0)))
    
    return contraction4
    #now the first layer is fully contracted. repeat this process for every layer but the last.
    
    
    
# #this function will contract everything for a layer (not first or end)
# def contractlayer(flatreward, W, M):
    
#         #first contract M with W to give object of type
#     #(state1,state2,action,virtual1,virtual2,reward2)
#     contraction1 = np.tensordot(M,W,((3),(2)))
    
#     #contract flat vector, to give (state1,state2,action,virtual1,virtual2)
#     contraction2 = np.tensordot(contraction1,flatreward,((5),(0)))
    
#     return contraction2


#function to manage the contracts betweenlayers, 
#such as policy and virtual index. This wont work if at layers 1 or end. 
def contractlayer(previouslayeroutput,policy,copy,flatreward,W,M):
    
    #first contract M with W to give object of type
    #(state1,state2,action,virtual1,virtual2,reward2)
    contraction1 = np.tensordot(M,W,((3),(2)))
    
    #contract flat vector, to give (state1,state2,action,virtual1,virtual2)
    layer = np.tensordot(contraction1,flatreward,((5),(0)))
    
    
    #now contract previouslayeroutput with copy tensor, to give 
    #(virtual index,copy2,copy3), this is needed as the previous state,
    # is needed twice.
    temp1 = np.tensordot(previouslayeroutput,copy,((0),(0)))
    
    
    #contract with policy to get new action.
    #resulting object is (virtual index, copy3,action)
    temp2 = np.tensordot(temp1,policy,((1),(0)))
    
    #the resultant of the total contraction is (new state, new virtual index)
    newoutput = np.tensordot(temp2,layer,((0,1,2),(3,0,2)))
    
    return newoutput

#contracting last layer with the rest.
def contractlayerT(MT,WT,flatstate,copy,previouslayeroutput,policy,flatreward):
    
    #gives shape(input state,action,reward)
    contraction1 = np.tensordot(MT,flatstate,((1),(0)))
    
    #contract with WT (inputstate,action,virtualindex,reward2)
    contraction2 = np.tensordot(contraction1,WT,((2),(1)))
    
    #contract with flatreward (input state,action,virtualindex)
    contraction3 = np.tensordot(contraction2,flatreward,((3),(0)))
    
    #find the action by first using copy
    contraction4 = np.tensordot(previouslayeroutput,copy,((0),(0)))
    
    action = np.tensordot(contraction4,policy,((1),(0)))
    
    #do the final contraction to obtain expected reward.
    
    expectedreturn = np.tensordot(contraction3,action,((0,1,2),(1,2,0)))
    
    return expectedreturn

#function to contract all layers to give the expected return. 
def contracteverything(M,MT,W,W1,WT,flatreward,policy,p0,flatstate,T,copy,plot):
    
    #first contract layer 1. 
    output = contractlayer1(flatreward, policy, M, W1,p0)

    #now contract all middle layers. 
    for i in range(1,T-1):
        output = contractlayer(output,policy[i],copy,flatreward,W,M)
    
    #now contract final layer
    
    expectedreturn = contractlayerT(MT,WT,flatstate,copy,output,policy[T-1],flatreward)
    
    print(expectedreturn)
    if plot==True:
        plotgreedy(policy,T,p0)
    return expectedreturn


#function to change the values in policy array, used in function below.
def changepolicy(policy,contraction,T):
    for i in range(2*T+1):
        if contraction[i,0] - contraction[i,1] > 0:
            policy[i,0] = 1
            policy[i,1] = 0
        elif contraction[i,0] - contraction[i,1] < 0:
            policy[i,0] = 0
            policy[i,1] = 1
    return policy
        
#it is easier to also have a function to contract all stuff after the layer being
#optimised.
def contractafteroptimiselayer(output,M,W,flatreward,copy,policy):
    #the input named 'output' is an object of (state2,virtual2,actionchosen,copychosen)
    #the actionchosen and copychosen must go to the policy of chosen layer.
    #as usual contract M,W and flatreward    
    contraction1 = np.tensordot(M,W,((3),(2)))
    #below gives (state1,state2,newaction,virtual1,virtual2)
    layer = np.tensordot(contraction1,flatreward,((5),(0)))
    
    #now can contract the output from chosen layer with copy. this 
    #gives shape (outputvirtual,actionchosen,copychosen,copyfromnewlayer1,copyfromnewlayer2)
    temp1 = np.tensordot(output,copy,((0),(0)))
    #now find the action by contracting one of the copyfromnewlayers with the
    #policy for this timestep. (outputvirtual,actionchosen,copychosen,copyfromnewlayer2,actionnewlayer)
    temp2 = np.tensordot(temp1,policy,((3),(0)))
    
    #finally contract this with the new layer. Need to contract the old
    #output virtual with new input virtual, and new input state with old copy, and
    #new action.
    #gives (outputstate,outputvirtual2,actionchosen,copychosen)
    output = np.tensordot(layer,temp2,((3,0,2),(0,3,4)))
    return output


#function to optimise policy. Start from T-1 layer and contract everything else
def DRMGpolicyoptimisation(flatreward,policy,M,W,W1,p0,copy,MT,WT,flatstate,T):
    ##we first optimize final layer.
    
    #first contract all layers but last.     
    #first contract layer 1. 
    output = contractlayer1(flatreward, policy, M, W1,p0)

    #now contract all middle layers. 
    for i in range(1,T-1):
        output = contractlayer(output,policy[i],copy,flatreward,W,M)
    
    #now must contract all final layer, except the policy. 
    
    #gives shape(input state,action,reward)
    contraction1 = np.tensordot(MT,flatstate,((1),(0)))
    
    #contract with WT (inputstate,action,virtualindex,reward2)
    contraction2 = np.tensordot(contraction1,WT,((2),(1)))
    
    #contract with flatreward (input state,action,virtualindex)
    contraction3 = np.tensordot(contraction2,flatreward,((3),(0)))
    
    #now contract copy tensor with previous output to give (virtual,copy1,copy2)
    temp1 = np.tensordot(output,copy,((0),(0)))
    
    #contract the 2 bits together to give (copy1, action). This is now whats used
    #to optimise policy. 
    temp2 = np.tensordot(temp1,contraction3,((0,1),(2,0)))
    
    #now optimise the policy.
    policy[T-1] = changepolicy(policy[T-1],temp2,T)
    
    ############################################################
    
    #Now need to optimise all other layers. This will be done by first contracting 
    #all layers until the one we're optimising. Then contracting everything after
    #it. 
    
    #for all layers except first and last.
    for x in range(1,T-1):
        #t = layer being optimised. 
        t = T-x
        
        #first contract layer 1. 
        output = contractlayer1(flatreward, policy, M, W1,p0)
        #now contract all middle layers until chosen layer. 
        for i in range(1,t-1):
            output = contractlayer(output,policy[i],copy,flatreward,W,M)
        
        #Now create the layer that we wish to optimise.
        #(state1,state2,action,virtual1,virtual2,reward2)
        contraction1 = np.tensordot(M,W,((3),(2)))
        
        #contract flat vector, to give (state1,state2,action,virtual1,virtual2)
        layer = np.tensordot(contraction1,flatreward,((5),(0)))
        
        
        #now contract previouslayeroutput with copy tensor, to give 
        #(virtual index,copy2,copy3), this is needed as the previous state,
        # is needed twice.
        temp1 = np.tensordot(output,copy,((0),(0)))
        
        #temp2 gives (state2,action,virtual2,copy3), state2 and virtual2
        #is what goes into next layer, copy3 and action goes to policy.
        temp2 = np.tensordot(layer,temp1,((0,3),(1,0)))
       
        #swapaxis of action and virtual 2 for convience. now it is
        #(state2,virtual2,action,copy3)
        output = np.moveaxis(temp2,1,2)
        #now contract all layers after chosen layer up until final layer
        for i in range(t,T-1):
            #output has form (state2,virtual2,action,copy) where we need to keep latter 2.
            output = contractafteroptimiselayer(output,M,W,flatreward,copy,policy[i])
        
        #last thing to do is to contract final layer.
        #gives shape(input state,action,reward)
        contraction1 = np.tensordot(MT,flatstate,((1),(0)))
        
        #contract with WT (inputstate,action,virtualindex,reward2)
        contraction2 = np.tensordot(contraction1,WT,((2),(1)))
        
        #contract with flatreward (input state,action,virtualindex)
        contraction3 = np.tensordot(contraction2,flatreward,((3),(0)))
        
        #gives (virtual2,actionchosen,copychosen,copynew1,copynew2)
        temp1 = np.tensordot(output,copy,((0),(0)))
        # gives (virtual2,actionchosen,copychosen,copynew2,actionnew)
        temp2 = np.tensordot(temp1,policy[T-1],((3),(0)))
        #gives (action,copychosen) which is ready to be put into optimiser.
        output = np.tensordot(temp2,contraction3,((0,3,4),(2,0,1)))
        #also need to swap the axis.
        output = np.moveaxis(output,0,1)
        policy[t-1] = changepolicy(policy[t-1],output,T)
#################################################################

    #Finally we also need to optimise the very first layer. To do this contract
    #all layer 1 except policy and every other layer. 
    
    #gives (copy1,copy2)
    contraction0 = np.tensordot(p0,copy,((0),(0)))
    #gives something of (end state,action,reward,copy2)
    contraction1 = np.tensordot(M,contraction0,((0),(0)))
    
    #contracting W1 
    #gives shape (end state,action,copy2,virtualindex,flatreward)
    contraction2 = np.tensordot(contraction1,W1,((2),(1)))
    
    #contract with flat reward, gives shape (end state, action, copy2 virtualindex)
    contraction3 = np.tensordot(contraction2, flatreward,((4),(0)))
    
    #to feed into function we need the order to be (endstate,virtualindex,action,copy)
    output = np.moveaxis(contraction3,3,1)

    #Now contract with all layers except last
    for i in range(1,T-1):
        #output has form (state2,virtual2,action,copy) where we need to keep latter 2.
        output = contractafteroptimiselayer(output,M,W,flatreward,copy,policy[i])
        
    #last thing to do is to contract final layer.
    #gives shape(input state,action,reward)
    contraction1 = np.tensordot(MT,flatstate,((1),(0)))
    
    #contract with WT (inputstate,action,virtualindex,reward2)
    contraction2 = np.tensordot(contraction1,WT,((2),(1)))
    
    #contract with flatreward (input state,action,virtualindex)
    contraction3 = np.tensordot(contraction2,flatreward,((3),(0)))
    
    #gives (virtual2,actionchosen,copychosen,copynew1,copynew2)
    temp1 = np.tensordot(output,copy,((0),(0)))
    # gives (virtual2,actionchosen,copychosen,copynew2,actionnew)
    temp2 = np.tensordot(temp1,policy[T-1],((3),(0)))
    #gives (action,copychosen) which is ready to be put into optimiser.
    output = np.tensordot(temp2,contraction3,((0,3,4),(2,0,1)))
    #also need to swap the axis.
    output = np.moveaxis(output,0,1)
    policy[0] = changepolicy(policy[0],output,T)

    return policy
    
    
#here is practice setting up the model. 
if __name__ == '__main__':
    
    #can choose value of T
    T = 8
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

    contracteverything(M,MT,W,W1,WT,flatreward,policy,p0,flatstate,T,copy,True)
    
    policy = DRMGpolicyoptimisation(flatreward, policy, M, W, W1, p0, copy, MT, WT, flatstate, T)
    plt.figure()
    contracteverything(M,MT,W,W1,WT,flatreward,policy,p0,flatstate,T,copy,True)
    
