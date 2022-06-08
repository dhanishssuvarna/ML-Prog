import numpy as np

class Perceptron():
    def __init__(self, input_size, lr=1, epochs=5):
        self.W = np.zeros(input_size)
        self.b=0
        self.epochs = epochs
        self.lr = lr
    
    def activation_fn(self, x):
        return 1 if x > 0 else 0
 
    def predict(self, x):
        z = self.W.T.dot(x)
        output = self.activation_fn(z+self.b)
        return output
 
    def fit(self, input, target):
        for _ in range(self.epochs):
            for i in range(target.shape[0]):
                x=input[i]
                y = self.predict(x)
                e = target[i] - y
                self.W = self.W + (self.lr * e * x)
                self.b=self.b + e
    
               
if __name__ == '__main__':
    input = np.array([
        [0, 0],
        [0, 1],
        [1, 0],  
        [1, 1]
    ])
    target = np.array([0, 0, 0, 1])
 
    perceptron = Perceptron(input_size=2)
    perceptron.fit(input, target)

    print("Final weights of AND gate : ",perceptron.W)
    print("Final bias of AND gate : ",perceptron.b) 
    
   # Testing new input
    test=np.array([0,0])
    print("Test value is: ",test)
    z=perceptron.W.T.dot(test)
    output=perceptron.activation_fn(z+perceptron.b)
    print("Actual AND gate output : ",0)
    print("Predicted AND gate output : ",output)