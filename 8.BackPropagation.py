import numpy as np
import pandas as pd

X = np.array(([2, 9], [1, 5], [3, 6],[1,1],[2,2]), dtype=float)
Y = np.array(([92], [86], [89],[10],[20]), dtype=float)
X = X/np.amax(X,axis=0)
Y = Y/100

# dataset = pd.read_csv('./dataset/Social_Network_Ads.csv')
# X=np.array(dataset.drop(['Purchased'], axis=1))
# Y=np.array(dataset['Purchased'])
# Y=np.reshape(Y, (len(Y),1))

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

epoch=100
lr=0.5

inputlayer_neurons = 2
hiddenlayer_neurons = 3
outputlayer_neurons = 1

#weight and bias initialization
w_h=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
b_h=np.random.uniform(size=(1,hiddenlayer_neurons))
w_o=np.random.uniform(size=(hiddenlayer_neurons,outputlayer_neurons))
b_o=np.random.uniform(size=(1,outputlayer_neurons))


#draws a random range of numbers uniformly of dim x*y
for i in range(epoch):
    #Forward Propogation
    h_ip=np.dot(X,w_h) + b_h
    h_op = sigmoid(h_ip)
    o_in= np.dot(h_op,w_o)+b_o
    output = sigmoid(o_in)
    
    #Backpropagation
    #for ouput layer
    EO = Y-output
    outgrad = derivatives_sigmoid(output)
    d_output = EO * outgrad
    #for hidden layer
    EH = d_output.dot(w_o.T)
    hiddengrad = derivatives_sigmoid(h_op)
    d_hidden = EH * hiddengrad
    #update the weights
    w_o += h_op.T.dot(d_output) *lr
    w_h += X.T.dot(d_hidden) *lr
    
    # print ("-----------Epoch-", i+1, "Starts---------\n")
    # print("Input : " + str(X)) 
    # print("Actual Output : " + str(y))
    # print("Predicted Output : " ,output)
    # print ("-----------Epoch-", i+1, "Ends----------\n")
        
print("Input: \n", X) 
print("Actual Output: \n", Y)
print("Predicted Output: \n" ,output)
