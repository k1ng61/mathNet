# mathNet
Libraries such as Pytorch and Tensorflow are essential in building deep learning models but they can also create a false illusion of understanding. Many people understand the scripts to run their models yet they lack the understanding of how they work. This is a huge problem if you are conducting research where you are trying build your own neural networks, where you can't just run scripts from popular libraries. To make new strides in the field of AI, you have to get a good understanding of how the state of the art algorithms work and there is no better way to do to that than looking at the math equations that encompass every alogirithm of AI. 

## Introduction:
Today, we are going to be looking at what Neural networks are and then going deep into a type of neural networ known as Feed Forward Neural Networks. 

### What are neural networks?

Over the past decade, the best-performing state of the art Artificial intelligence applications have been derived from a subset of AI known as Deep Learning. From self driving cars to Neuro-Imaging, this frontier of AI has been solving problems that have never been solved before. But what exactly is Deep Learning? To answer this question we have to dive deep into a computing system known as Neural Networks.

Neural networks is a structure to teach an AI that is designed to simulate animal brains. It’s made of a multitude of connected nodes called neurons which form multiple layers inside the neural network structure. Each Neural Network is made up of three different types of layers: an input layer, an output layer, and a couple of hidden layers. As shown in the figure below, the input layer takes in data and passes it on to the hidden layers, which then compute all the calculations and send in the final probabilities to the output layer.

![alt text](https://miro.medium.com/max/1700/0*_SH7tsNDTkGXWtZb.png)


The main computation here occurs in the hidden layers. Essentially, each node has a value and as that node passes its value onto the next node, that node calculates its own value and so on. Now, lets break things down and look at exactly how each node calculates its value.


## Nodes:

Each node takes in the values of all the other nodes in the previous layers and multiplies them with different weights. As shown in the equation below, to calculate the value of a node you have to first take a sum of the dot product of all values and weights. The values and weights, in this case, are going to be two arrays where each value in the values array corresponds to a weight in the array of weights. After getting the sum of all the dot products, we also have to add the variable b. Lowercase b stands for a bias, which exists to help the model better fit the given dataset. 

![alt text](https://miro.medium.com/max/960/1*0lejoYyyQWjYzEP_BNW2nw.jpeg)
 
After you have sum and the bias you have to feed this value into a function f. The function f denotes an activation function whose job is to normalize the value so it’s between 0 and 1. There are many activation functions that can do this job but let's look at a popular one known as a sigmoid function. The sigmoid function is denoted by the following equation - 

F(x) = 11+e-x

![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/1200px-Logistic-curve.svg.png)

For every value of x, the sigmoid function returns a value between 0 and 1. As you look at the graph of this function, this makes sense as the range of this function is between 0 and 1 for all real values of x. 

Let’s ground this idea further by implementing it in python. As seen below, we have a class called Node in which each node has an inputs, weights, and outputs array. There are also two variables for the value of the node and the bias. Inside this class, we have a function called calculate that performs the computation described earlier. In the first line of the function, it loops through the inputs and weights array and multiplies each input to its corresponding weight, and stores it into the outputs array. Then, it loops through the output array to get a sum of all the values calculated. After adding the bias, the value is taken as an input into the sigmoid function which normalizes it to be between 0 and 1.
```bash
Import math

class Net:
   def __init__(self, inputs, weights, bias):
       self.inputs = inputs
       self.weights = weights
       self.outputs = []
       self.value = 0
       self.bias = bias

def sigmoid(self, value):  # activation function

   return 1 / (1 + math.exp(-value))

def calculate(self):
   for i in range(0, len(self.inputs)):
       self.outputs[i] = self.inputs[i] * self.weights[i]

   for i in self.outputs:
       self.value += i

   self.value += self.bias
  
   self.value = self.sigmoid(self.value)

   return self.value
```

Now that we have implemented the basic structure, we have to figure out how we can optimize the weights in order to fit the data because in its present state, the structure multiplies the values in the dataset to a random set of weights. In order to finish off the neural network, we have to let the network make predictions about the output and then compare it to the actual output which would allow it to fine-tune the weights and bias to make the prediction better, a process known as training. 

# Training:

To train a neural network, you have to look at the cost of each prediction. Looking at the cost will allow you to see how accurate the prediction was and then make a decision about if you need to make the weights larger or smaller. There are many cost functions that do the job but let's look at a popular called the mean squared error, which is denoted by the following equation -

![alt text](https://i.imgur.com/vB3UAiH.jpg)

To calculate the mean square error you have to calculate the sum of the squared difference between all the actual and predicted variables. Then, you have to divide that value by the number of predictions to find the average of all the square errors. 

Now that we have an equation to calculate the cost, we have to fine-tune the bias and the weights in order to minimize the cost function. We have to approach this as an optimization problem where we are trying to find the absolute minima. There are several ways to do this but let's look at the gradient descent algorithm. The main idea behind gradient descent is that you are trying to take small steps repeatedly towards the minima of a gradient. The gradient represents the error value based on the values of the weights and bias so logically, we are trying to minimize that number. In many aspects of programming, we are taught to maximize your gains but the gradient descent algorithm takes a different approach. Instead of maximizing your gains, the algorithm minimizes the loss. In essence, both of these ideas are doing the same thing but the approach is completely different. The algorithm can be represented mathematically with this equation:

![alt text](https://hackernoon.com/hn-images/0*8yzvd7QZLn5T1XWg.jpg)


The function J at the end of this equation represents the cost function which in this case, is the mean squared error. The greek letter a represents the learning rate which is defined as how big or small steps you’re taking per iteration of this loop in order to minimize the loss. The learning rate is important because if it’s too big, you take So, first we take the derivative of the cost function with respect to the weights/bias then after multiplying it with the learning rate, we subtract it with the current weight values. The theory behind this is that the derivative of the cost function would tell us if the weights are increasing or decreasing the loss. If the loss increases when the weights increase, the derivative will result in a positive value which would then be subtracted by the current rates, making them smaller. Logically, if increasing the weights result in a bigger loss then decreasing the weights would result in a smaller loss and vice versa. 
Let’s put this theory into practice, look at the code below: 

```bash
def gradient_descent(self, epoch, lr):
   for epoch in range(epoch):
       prediction = self.calculate()

       # derivative of mean squared error
       dcost = prediction - self.outputs
       dpred = self.sigmoid(prediction) * (1 - self.sigmoid(prediction))

       zdelta = dcost * dpred

       for weight in self.weights:
           weight -= lr * (self.inputs * zdelta)

       for num in zdelta:
           self.bias -= lr * num
```
