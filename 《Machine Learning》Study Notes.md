- [Lecture 1 Introduction](#lecture-1-introduction)
  - [2. Cost function optimization for regularization](#2-cost-function-optimization-for-regularization)

---------------------

# Lecture 1 Introduction
## 1. What is Machine Learning
### 1.1 Definition

Tom Michel (1999) "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."

### 1.2 Types of learning algorithms
1. **Supervised learning**: each the computer how to do something, then let it use it;s new found knowledge to do it
2. **Unsupervised learning**: Let the computer learn how to do something, and use this to determine structure and patterns in data
3. **Reinforcement learning**
4. **Recommender systems**

## 2. Supervised learning - introduction
### 2.1 Regression Problem
- Predict continuous valued output (price)
- No real discrete delineation 
ex. predict housing prices

We gave the algorithm a data set where a "right answer" was provided. The idea is we can learn what makes the price a certain value from the training data.

### 2.2 Classification problem
ex. Classify data into one of two discrete classes - no in between, either malignant or not based on age and size

## 3. Unsupervised learning
### 3.1 clustering algorithm
we get unlabeled data and cluster data into to groups

### 3.2 Cocktail party algorithm

# Lecture 2 Linear Regression with One Variable
## 1. Model Representation
A hypothesis takes in some variable
Uses parameters determined by a learning system
Outputs a prediction based on that input

<a href="http://www.codecogs.com/eqnedit.php?latex=h_\Theta&space;(x)&space;=&space;\Theta&space;_0&space;&plus;&space;\Theta&space;_1x" target="_blank"><img src="http://latex.codecogs.com/gif.latex?h_\Theta&space;(x)&space;=&space;\Theta&space;_0&space;&plus;&space;\Theta&space;_1x" title="h_\Theta (x) = \Theta _0 + \Theta _1x" /></a>
- Algorithm outputs a function (denoted h ) (h = hypothesis)
- m = number of training examples
- θ are parameters

## 2. Cost Function
A cost function lets us figure out how to fit the best straight line to our data, choosing values for θi (parameters)

picture 2.2 1

This cost function is also called the **squared error cost function**

1/2m the 2 makes the math a bit easier

picture 2.2 2

The optimization objective for the learning algorithm is find the value of θ1 which minimizes J(θ1). So, here θ1 = 1 is the best value for θ1

Generates a 3D surface plot where axis are X = θ1 Z = θ0 Y = J(θ0,θ1)\
We can see that the height (y) indicates the value of the cost function, so find where y is at a minimum\
Doing this by eye/hand is a pain in the ass. What we really want is an efficient algorithm fro finding the minimum for θ0 and θ1

picture 2.2 3

## 3. Gradient Descent
Problem
- We have J(θ0, θ1)
- We want to get min J(θ0, θ1)

### 3.1 How does it work?
- Each time you change the parameters, you select the gradient which reduces J(θ0,θ1) the most possible 
- Has an interesting property--**Local Minimum**. Where you start can determine which minimum you end up

picture

### 3.2 Formal definition
picture

Update θj by setting it to (θj - α) times the partial derivative of the cost function with respect to θj

**α (alpha)**
Is a number called the **learning rate**

For j = 0 and j = 1 means we simultaneously update both. we need a temp value. Then, update θ0 and θ1 at the same time.

**Derivative**\
Lets take the tangent at the point and look at the slope of the line
So moving towards the mimum (down) will greate a negative derivative, alpha is always positive, so will update j(θ1) to a smaller value

**Alpha term (α)**
- Too small\
Takes too long
- Too large\
Can overshoot the minimum and fail to converge

When you get to a local minimum\
Gradient of tangent/derivative is 0.
So derivative term = 0
alpha * 0 = 0
So θ1 = θ1- 0 
So θ1 remains the same

## 4. Linear regression with gradient descent
picture

picture

The linear regression cost function is always a **convex function** - always has a single minimum---one global optima

This is actually **Batch Gradient Descent**\
Refers to the fact that over each step you look at all the training data

There exists a numerical solution for finding a solution for a minimum function---**Normal equations method**解方程

# Lecture 4 Linear Regression with Multiple Variables
## 1. Multiple features
### 1.1 Notations
In original version we had
- X = house size, use this to predict
- y = house price

If in a new scheme we have more variables (such as number of bedrooms, number floors, age of the home)\
x1, x2, x3, x4 are the four features
- x1 - size (feet squared)
- x2 - Number of bedrooms
- x3 - Number of floors
- x4 - Age of home (years)

y is the output variable (price)

- n: number of features (n = 4)
- m: number of examples (i.e. number of rows in a table)
- <a href="http://www.codecogs.com/eqnedit.php?latex=x_j^i" target="_blank"><img src="http://latex.codecogs.com/gif.latex?x_j^i" title="x_j^i" /></a>: The value of feature j in the ith training example

### 1.2 Formula

**hθ(x) = θ0 + θ1x1 + θ2x2 + θ3x3 + θ4x4**\
X0=1. So now your **feature vector X** is n + 1 dimensional feature vector indexed from 0\
**Parameters theta** are also in a 0 indexed n+1 dimensional vector

In matrix, 
<a href="http://www.codecogs.com/eqnedit.php?latex=h_\theta(x)&space;=\theta^T&space;X" target="_blank"><img src="http://latex.codecogs.com/gif.latex?h_\theta(x)&space;=\theta^T&space;X" title="h_\theta(x) =\theta^T X" /></a>\
[1 x (n+1)] * [(n+1) x 1] 先不管这个的行列安排，feature vector和parameter vector都是竖着的vector

## 2. Gradient descent for multiple variables
### 2.1 cost function and gradient descent
Our cost function is
picture L4 2 1

Gradient Descent
picture L4 2 2
### 2.2 Gradient Decent in practice: 1 Feature Scaling
If you have a problem with multiple features. You should make sure those features have a similar scale . Means gradient descent will converge more quickly

May want to get everything into -1 to +1 range (approximately)

picture L4 2.2 1

### 2.3 Gradient Decent in practice: 2 Learning Rate
#### 2.3.1 Iteration
Plot min J(θ) vs. no of iterations

picture

Very hard to tel in advance how many iterations will be needed
Can often make a guess based a plot like this after the first 100 or so iterations\
If, for example, after 1000 iterations you reduce the parameters by nearly nothing you could chose to only run 1000 iterations in the future

Another problem: too big

#### 2.3.2 J is increasing
If you plot J(θ) vs iterations and see the value is increasing - means you probably need a smaller α
#### 2.3.3 J looks like waves
Here again, you need a smaller α

if α is too small then rate is too slow

**Typically**\
Try a range of alpha values\
Plot J(θ) vs number of iterations for each version of alpha\
Go for roughly threefold increases\
0.001, 0.003, 0.01, 0.03. 0.1, 0.3

## 3. Features and polynomial regression
### 3.1 New features
Choice of features and how you can get different learning algorithms by choosing appropriate features

You don't have to use just two features
Can create new features
Might decide that an important feature is the land area
So, create a new feature = frontage

Often, by defining new features you may get a better model

### 3.2 Polynomial regression
May fit the data better

θ0 + θ1x + θ2x^2 e.g. here we have a quadratic function

x1 = x\
x2 = x^2\
x3 = x^3\
By selecting the features like this and applying the linear regression algorithms you can do polynomial linear regression

## 4. Normal Equation
Normal equation solves θ analytically\
Solve for the optimum value of theta

### 4.1 How does it work
1. Take derivative of J(θ) with respect to θ
2. Set that derivative equal to 0
3. Allows you to solve for the value of θ which minimizes J(θ)\
Take the partial derivative of J(θ) with respect θj and set to 0 for every j\
Do that and solve for θ0 to θn\
This would give the values of θ which minimize J(θ)

### 4.2 Example
1. Construct a matrix (X - the design matrix) which contains all the training data features in an [m x n+1] matrix
2. Construct a column vector y vector [m x 1] matrix
3. Using the following equation (X transpose * X) inverse times X transpose y

picture\
picture

If you're using the normal equation then no need for feature scaling

### 4.3  Gradient descent VS Normal Equation
#### 4.3.1 Gradient descent
- Need to chose learning rate
- Needs many iterations - could make it slower
- Works well even when n is massive (millions)
- Better suited to big data\
  100 or even a 1000 is still (relativity) small\
  If n is 10 000 then look at using gradient descent

#### 4.3.2 Normal Equation
- Normal equation needs to compute (X^T X)^(-1)
With most implementations computing a matrix inverse grows by O(n3 )
So not great
- Slow of n is large\
Can be much slower

### 4.4 What if (X^T X) is non-invertible
Normally two common causes
1. Redundant features in learning mode\
e.g.
x1 = size in feet
x2 = size in meters squared\
Look at features --> are features linearly dependent?
2. Too many features\
 Trying to fit 101 parameters from 10 training examples\
 To solve this we\
a. Delete features\
b. Use **regularization** (let's you use lots of features for a small training set)

# Lecture 6 Logistic Regression
## 1. what is Logistic Regression
Classification problems:
- Email -> spam/not spam?
- Online transactions -> fraudulent?
- Tumor -> Malignant/benign

Variable in these problems is Y. Y is either 0 or 1

**Logistic regression generates a value where is always either 0 or 1.
Logistic regression is a classification algorithm**

## 2. Hypothesis representation
We want our classifier to output values between 0 and 1

picture

g(z) = 1/(1 + e^-z)
This is the sigmoid function, or the logistic function


When our hypothesis (hθ(x)) outputs a number, we treat that value as the estimated probability that y=1 on input x

hθ(x) = P(y=1|x ; θ) Probability that y=1, given x, parameterized by θ

P(y=1|x ; θ) + P(y=0|x ; θ) = 1

## 3. Decision boundary
### 3.1 linear decision boundaries

z = θ0 + θ1x1 + θ2x2 = 0 we graphically plot our decision boundary (Concretely, the straight line is the set of points where hθ(x) = 0.5)

-3x0 + 1x1 + 1x2 >= 0 then we predict y = 1

picture

### 3.2 Non-linear decision boundaries
<a href="http://www.codecogs.com/eqnedit.php?latex=h\Theta&space;(x)&space;=&space;g(\Theta_0&space;&plus;&space;\Theta_1x1&plus;&space;\Theta_3x_1^2&space;&plus;&space;\Theta_4x_2^2)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?h\Theta&space;(x)&space;=&space;g(\Theta_0&space;&plus;&space;\Theta_1x1&plus;&space;\Theta_3x_1^2&space;&plus;&space;\Theta_4x_2^2)" title="h\Theta (x) = g(\Theta_0 + \Theta_1x1+ \Theta_3x_1^2 + \Theta_4x_2^2)" /></a>

Predict that "y = 1" if -1 + x12 + x22 >= 0

picture

By using higher order polynomial terms, we can get even more complex decision boundaries

## 4. Cost Function
### 4.1 why not cost function in linear regression
if we use the cost function in Linear regression,  this is a non-convex function for parameter optimization

picture

Our hypothesis function has a non-linearity (sigmoid function of hθ(x) )
This is a complicated non-linear function\
If you take hθ(x) and plug it into the Cost() function, and them plug the Cost() function into J(θ) and plot J(θ) we find many local optimum -> **non-convex function**

what should we do?\
To get around this we need a different, convex Cost() function which means we can apply gradient descent
### 4.2 A convex logistic regression cost function
picture

picture

picture

## 5. Simplified cost function and gradient descent

picture

in summary, our cost function for the θ parameters can be defined as

picture

Why do we chose this function when other cost functions exist?
- This cost function can be derived from statistics using the principle of maximum likelihood estimation. Note this does mean there's an underlying Gaussian assumption relating to the distribution of features 
- Also has the nice property that it's convex

流程：
1. To fit parameters θ:
Find parameters θ which minimize J(θ)
This means we have a set of parameters to use in our model for future predictions
2. Then, if we're given some new example with set of features x, we can take the θ which we generated, and output our prediction using\
picture
3. This result is p(y=1 | x ; θ)

### 5.1 How to minimize the logistic regression cost function
Now we need to figure out how to minimize J(θ). Use gradient descent as before.
Repeatedly update each parameter using a learning R

picture

Can do the same thing here for logistic regression.

Feature scaling for gradient descent for logistic regression also applies here.

## 6. Advanced Optimization
Alternatively, instead of gradient descent to minimize the cost function we could use
- Conjugate gradient
- BFGS (Broyden-Fletcher-Goldfarb-Shanno)
- L-BFGS (Limited memory - BFGS)

advantages:
- No need to manually pick alpha (learning rate)
- Have a clever inner loop (line search algorithm) which tries a bunch of alpha values and picks a good one
- Often faster than gradient descent
- Can be used successfully without understanding their complexity

Disadvantages
- Could make debugging more difficult
- Should not be implemented themselves

## 7. Multiclass classification problems
**One vs. all classification**
Split the training set into three separate binary classification problems
i.e. create a new fake training set
- Triangle (1) vs crosses and squares (0) hθ1(x) C与S视为一类，再与T分类\
P(y=1 | x1; θ)
- Crosses (1) vs triangle and square (0) hθ2(x)\
P(y=1 | x2; θ)
- Square (1) vs crosses and square (0) hθ3(x)\
P(y=1 | x3; θ)

picture

Overall
1. Train a logistic regression classifier hθ(i)(x) for each class i to predict the probability that y = i 训练三个0即三个分类器
2. On a new input, x to make a prediction, pick the class i that maximizes the probability that hθ(i)(x) = 1 对新的数据判断哪个h最大

# Lecture 7 Regularization
## 1. The problem of overfitting

### 1.1 Underfitting (high bias)
ex. Fit a linear function to the data - not a great model\
Bias is a historic/technical one - if we're fitting a straight line to the data we have a strong preconception that there should be a linear fit

### 1.2 Overfitting (high variance)
ex. a high order polynomial gives and overfitting (high variance hypothesis)

#### 1.2.1 Addressing overfitting
1. **Reduce number of features**
    - If you have lots of features and little data - overfitting can be a problem. Manually select which features to keep
    - Model selection algorithms are discussed later (good for reducing number of features)
    - But, in reducing the number of features we lose some information
2. **Regularization**
    - Keep all features, but reduce magnitude of parameters θ
    - Works well when we have a lot of features, each of which contributes a bit to predicting y

## 2. Cost function optimization for regularization
picture

picture

The addition in blue is a modification of our cost function to help penalize θ3 and θ4. So here we end up with θ3 and θ4 being close to zero (because the constants are massive). \
So we're basically left with a quadratic function

Small values for parameters corresponds to a simpler hypothesis (you effectively get rid of some of the terms)

With regularization, take cost function and modify it to shrink all the parameters

picture

**λ is the regularization parameter**
Controls a trade off between our two goals
1. Want to fit the training set well.
2. Want to keep parameters small

**If λ is very large** we end up penalizing ALL the parameters (θ1, θ2 etc.) so all the parameters end up being close to zero. it's like we got rid of all the terms in the hypothesis
This results here is then **underfitting**

## 3. Regularized linear regression and logistic regression
picture

picture

<a href="http://www.codecogs.com/eqnedit.php?latex=1-\alpha&space;\frac{\lambda}{m}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?1-\alpha&space;\frac{\lambda}{m}" title="1-\alpha \frac{\lambda}{m}" /></a>\
Is going to be a number less than 1 usually. This in effect means θj gets multiplied by 0.99. Means the squared norm of θj a little smaller

**It is the same for logistic regression,** except obviously the hypothesis is very different

# Lecture 8 Neural Networks - Representation
## 1. Why do we need neural networks?
ex. If 100 x 100 RB then --> 50 000 000 features. simple logistic regression here is not appropriate for large complex systems
Neural networks are much better for a complex nonlinear hypothesis even when feature space is huge

## 2. Model representation
picture

First layer is the input layer\
Final layer is the output layer - produces value computed by a hypothesis\
Middle layer(s) are called the hidden layers

picture

To take care of the extra bias unit add a02 = 1 
So add a02 to a2 making it a 4x1 vector

<a href="http://www.codecogs.com/eqnedit.php?latex=\theta_{ji}^l" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\theta_{ji}^l" title="\theta_{ji}^l" /></a>
- j (first of two subscript numbers)= ranges from 1 to the number of units in layer l+1
- i (second of two subscript numbers) = ranges from 0 to the number of units in layer l
- l is the layer you're moving FROM


This process is also called **forward propagation**
1. Start off with activations of input unit
i.e. the x vector as input
2. Forward propagate and calculate the activation of each layer sequentially


- logistic regression: you would have to calculate your own exciting features to define the best way to classify or describe something
- NN: the mapping from layer 1 to layer 2 (i.e. the calculations which generate the **a2** features) is determined by another set of parameters - Ɵ1. instead of being constrained by the original input features, a neural network can **learn its own features** to feed into logistic regression

## 3. Neural network example

一个简单的例子展示NN的内在过程如何计算AND
picture

picture

## 4. Multiclass classification
Recognizing pedestrian, car, motorbike or truck\
Build a neural network with four output units. 

picture

Just like one vs. all described earlier. Here we have four logistic regression classifiers in the final layer

# Lecture 9 Neural Network: Learning
## 1. Cost Function
notation:
- Training set is {(x1, y1), (x2, y2), (x3, y3) ... (xn, ym)
- L = number of layers in the network
- sl = number of units (not counting bias unit) in layer l 
- k distinct classifications. So y is a k-dimensional vector of real numbers.

picture

the first half:\
For each training data example (i.e. 1 to m - the first summation)
Sum for each position in the output vector\
the second half:\
also called a weight decay term

## 2. Back propagation algorithm
To minimize a cost function we just write code which computes the following
- J(Ɵ)
- Partial derivative terms  PICTURE

forward propagation
PICTURE

Before we dive into the mechanics, let's get an idea regarding the intuition of the algorithm\
For each node we can calculate **(δjl) - this is the error of node j in layer l**\
EX. <a href="http://www.codecogs.com/eqnedit.php?latex=\delta&space;_j^4&space;=&space;a_j^4&space;-&space;y_j" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\delta&space;_j^4&space;=&space;a_j^4&space;-&space;y_j" title="\delta _j^4 = a_j^4 - y_j" /></a>


PICTURE

进一步推导，
<a href="http://www.codecogs.com/eqnedit.php?latex=\delta&space;^3&space;=&space;(\Theta&space;^3)^T&space;\delta&space;^4&space;.&space;*(a^3&space;.&space;*&space;(1&space;-&space;a^3))" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\delta&space;^3&space;=&space;(\Theta&space;^3)^T&space;\delta&space;^4&space;.&space;*(a^3&space;.&space;*&space;(1&space;-&space;a^3))" title="\delta ^3 = (\Theta ^3)^T \delta ^4 . *(a^3 . * (1 - a^3))" /></a>

<a href="http://www.codecogs.com/eqnedit.php?latex=\delta&space;^2&space;=&space;(\Theta&space;^2)^T&space;\delta&space;^3&space;.&space;*(a^2&space;.&space;*&space;(1&space;-&space;a^2))" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\delta&space;^2&space;=&space;(\Theta&space;^2)^T&space;\delta&space;^3&space;.&space;*(a^2&space;.&space;*&space;(1&space;-&space;a^2))" title="\delta ^2 = (\Theta ^2)^T \delta ^3 . *(a^2 . * (1 - a^2))" /></a>\
. * is the element wise multiplication between the two vectors

**Why do we do calculate the error term?**

 we want the δ terms because through a very complicated derivation you can use δ **to get the partial derivative of Ɵ with respect to individual parameters** (if you ignore regularization, or regularization is 0, which we deal with later)
 
 <a href="http://www.codecogs.com/eqnedit.php?latex=\frac{\alpha&space;}{\Theta&space;_{ij}^{(l)}}J(\Theta&space;)=a_j^l\delta&space;_i^{(l&plus;1)}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\frac{\alpha&space;}{\Theta&space;_{ij}^{(l)}}J(\Theta&space;)=a_j^l\delta&space;_i^{(l&plus;1)}" title="\frac{\alpha }{\Theta _{ij}^{(l)}}J(\Theta )=a_j^l\delta _i^{(l+1)}" /></a>

then, sum it all together to get the partial derivatives!

use Δ to accumulate the partial derivative terms

picture

After executing the body of the loop, exit the for loop and compute 

picture

<a href="http://www.codecogs.com/eqnedit.php?latex=\frac{\alpha&space;}{\Theta&space;_{ij}^{(l)}}J(\Theta&space;)=D_{ij}^{(l)}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\frac{\alpha&space;}{\Theta&space;_{ij}^{(l)}}J(\Theta&space;)=D_{ij}^{(l)}" title="\frac{\alpha }{\Theta _{ij}^{(l)}}J(\Theta )=D_{ij}^{(l)}" /></a>

We have calculated the partial derivative for each parameter\
We can then use these in gradient descent or one of the advanced optimization algorithms

## 3. Back propagation intuition
forward propagation
picture

Back propagation
picture

this is a part of J

δ is
picture


picture

## 4. Gradient Checking
picture

picture

1. Create a vector of partial derivative approximations
2. Using the vector of gradients from backprop (DVec)
3. **Check that gradApprox is basically equal to DVec**. Gives confidence that the Backproc implementation is correc


GradAprox stuff is very computationally expensive\
In contrast backprop is much more efficient (just more fiddly)\
所以不用这种办法来计算梯度

## 5. Random initialization
Pick random small initial values for all the theta values. **Between 0 and 1, then scale by epsilon (where epsilon is a constant)**

If you start them on zero (which does work for linear regression) then the algorithm fails - all activation values for each layer are the same. theta不能都一样

## 6. Summary
1. pick a network architecture
    - Input units - number of dimensions x (dimensions of feature vector)
    - Output units - number of classes in classification problem
    - Hidden units
2. Training a neural network
    1. Randomly initialize the weights
    2. Implement forward propagation to get hƟ(x)i for any xi
    3. Implement code to compute the cost function J(Ɵ)
    4. **Implement back propagation to compute the partial derivatives**
3. Use gradient checking. Disable the gradient checking code for when you actually run it, because it is very slow
4. Use **gradient descent**or an advanced optimization method with back propagation to try to minimize J(Ɵ) as a function of parameters Ɵ\
start from some random point and move downhill, avoiding local minimum

# Lecture 10 Advice for applying Machine Learning
## 1. Deciding what to try next
when you test on new data you find it makes unacceptably large errors in its predictions. What should you try next?

There are many things you can do:
- Get more training data
- Try a smaller set a features
- Try getting additional features
- Adding polynomial features
- Building your own, new, better features
- Try decreasing or increasing λ

Machine learning diagnostics: Tests you can run to see what is/what isn't working for an algorithm. 找到真正的问题所在

## 2. Evaluating a hypothesis
Split data into two portions
- 1st portion is training set
- 2nd portion is test set

Compute the **test error**\
Jtest(θ) = average square error as measured on the test set

picture

或者直接简单计算错误率

## 3. Model selection and training validation test sets
How to chose regularization parameter or degree of polynomial (model selection problems)

Take these parameters and look at the test set error for each using the previous formula. **See which model has the lowest test set error**\
Jtest(θ1)\
Jtest(θ2)\
...\
Jtest(θ10)\
BUT, this is going to be an optimistic estimate of generalization error, **because our parameter is fit to that test set**

Improved model selection
Given a training set instead split into three pieces
1. Training set (60%) - m values
2. **Cross validation (CV) set (20%)mcv**
3.  Test set (20%) mtest 

So
1. Minimize cost function for each of the models as before 
2. Test these hypothesis on the cross validation set to generate the cross validation error
3. **Pick the hypothesis with the lowest cross validation error**
e.g. pick θ5
4. Finally
Estimate **generalization error of model using the test set**

## 4. Diagnosis - bias vs. variance

If you get bad results usually because of one of
- **High bias** - **under-fitting problem**
- **High variance** - **over-fitting problem**

We want to minimize both errors. CV(Cross validation) error and test set error

picture

we can find that:
- if d is too small --> this probably corresponds to a **high bias problem. both cross validation and training error are high**
- if d is too large --> this probably corresponds to a **high variance problem. cross validation error is high but training error is low**

## 5. Regularization and bias/variance
- λ = large. So high bias -> under fitting data
- λ = small. So high variance -> Get overfitting

Often increment by factors of 2 so\
model(1)= λ = 0\
model(2)= λ = 0.01\
model(3)= λ = 0.02\
model(4) = λ = 0.04\
model(5) = λ = 0.08\
.\
.\
.\
model(p) = λ = 10\
This gives a number of models which have different λ\
now we have a set of parameter vectors corresponding to models with different λ values\
Measure average squared error on cross validation set. **Pick the model which gives the lowest error**

## 5. learning curve
Jtrain (average squared error on training set) or Jcv (average squared error on cross validation set)

What do these curves look like if you have
1. **High bias**
    - The problem with high bias is because cross validation and training error are both high
    - picture
    - the function just doesn't fit the data. increase in data will not help it fit
    - It's too simplistic
2. **High variance**
    - there's a big gap between training error and cross validation error
    - picture
    - more data is probably going to help. Jtrain slowly increases



## 6. What to do next (revisited)
How do these ideas help us chose how we approach a problem?发现问题是overfitting还是high bias之后开始解决问题

- Get more examples --> helps to fix high variance
Not good if you have high bias

- Smaller set of features --> fixes high variance (overfitting)
Not good if you have high bias

- Try adding additional features --> fixes high bias (because hypothesis is too simple, make hypothesis more specific)

- Add polynomial terms --> fixes high bias problem

- Decreasing λ --> fixes high bias

- Increases λ --> fixes high variance

### 6.2 network architecture
- small neural network: computationally cheaper; prone to under fitting
- Larger network: computational expensive
Prone to over-fitting
(Use regularization)

# Lecture 11  Machine Learning System Design

## 1. Prioritizing what to work on--represent features
How do represent x (features of the email

picture

Encode this into a reference vector. In practice its more common to have a training set and pick the most frequently n words, where n is 10 000 to 50 000

## 2. Error analysis
Recommended Approach:
1. Spend at most 24 hours developing an **initially bootstrapped algorithm**\
Implement and test on cross validation data
2. **Plot learning curves** to decide if more data, features etc will help algorithmic optimization\
Hard to tell in advance what is important\
Learning curves really help with this\
Way of avoiding premature optimization 要根据事实来判断，而不能靠感觉来优化
3. Manually examine the samples (in cross validation set) that your algorithm made errors on.\
See if you can work out why
Systematic patterns发现有没有系统性错误\

Importance of numerical evaluation. See if a change improves an algorithm or not当然有一个数值方法也很重要

## 3. Error metrics for skewed analysis
### 3.1 Precision and recall
picture

- precsion: 预测得正确\
Of all patients we predicted have cancer, what fraction of them actually have cancer
- recall: 与漏判率成反比。RECALL越高，漏盘越低\
Of all patients in set that actually have cancer, what fraction did we correctly detect

### 3.2 Trading off precision and recall

Trained a logistic regression classifier :\
Predict 1 if hθ(x) >= 0.3\
Predict 0 if hθ(x) < 0.3\
i.e. 30% chance they have cancer. So now we have have a **higher recall**, but lower precision

picture


This curve can take many different shapes depending on classifier details. Is there a way to automatically chose the threshold? can we convert P & R into one number?

**F1 Score (fscore)** = 2 * (PR/ [P + R])
- If P = 0 or R = 0 the Fscore = 0
- If P = 1 and R = 1 then Fscore = 1
- The remaining values lie between 0 and 1

If you're trying to automatically set the threshold, one way is to try a range of threshold values and evaluate them on your cross validation set 

Then pick the threshold which gives the best fscore.

## 4. Large data rational

**With supervised learning algorithms - performance is pretty similar**

**What matters more often** is:
The amount of training data
 and Skill of applying algorithms

- Low bias <-- use complex algorithm
- Low variance <-- use large training set

使用复杂的模型和大量的数据，training error和CV error都很低的情况下，test error应该也很低

如果数据量小的话，Training error should be small

# Lecture 12 Support vector machine

a cleaner way of learning non-linear functions

## 1. Optimizaiton Object
As for logistic regression, the cost function is

picture

If y = 1 then only the first term in the objective matters
picture

If y = 0 then only the second term matters
picture

To build a SVM we must redefine our **cost functions**
picture

For the SVM we take our two logistic regression y=1 and y=0 terms described previously and replace with\
cost1(θT x)
cost0(θT x)\
so we get

picture

**SVM notation is slightly different:**
1. Get rid of the 1/m terms
2. For logistic regression we had two terms, So we could describe it as A + λB.\
For SVMs the convention is to use a different parameter called C
So do **CA + B**

If C were equal to 1/λ then the two functions (CA + B and A + λB) would give the same value

## 2. Large margin intuition
Unlike logistic, hθ(x) doesn't give us a probability, but instead we get a **direct prediction of 1 or 0**

SVM wants a bit more than that - doesn't want to *just* get it right, but have the value be quite a bit bigger than zero (as the formula shown above)

**what are the consequences of these a little big value?**\
Consider a case where we set C to be huge, C = 10000. If C is huge we're going to pick an A value so that A is equal to zero in order to minimize the cost function.

更进一步 if we think of our optimization problem a way to ensure that this first "A" term is equal to 0, we re-factor our optimization problem into just minimizing the "B" (regularization) term可以只剩B，把A列为subject条件

picture

picture

as you can see, green and magenta lines are functional decision boundaries which could be chosen by logistic regression. That black line has a larger minimum distance (margin) from any of the training examples
C大相当于λ小，拟合程度高，对异常值更敏感

picture

If you were just using large margin then SVM would be very **sensitive to outliers**. So the idea of SVM being a large margin classifier is only really relevant when you have no outliers.\
C值很大的时候，并且在没有异常值的时候，才能作为Large margin classifier

## 3. Large margin classification mathematics 
### 3.1 Inner product
||u|| = SQRT(u12 + u22)

picture

<a href="http://www.codecogs.com/eqnedit.php?latex=p&space;*&space;||u||&space;=&space;u_1v_1&plus;&space;u_2v_2" target="_blank"><img src="http://latex.codecogs.com/gif.latex?p&space;*&space;||u||&space;=&space;u_1v_1&plus;&space;u_2v_2" title="p * ||u|| = u_1v_1+ u_2v_2" /></a>

### 3.2 SVM decision boundary
picture

our optimization function can be redefined as 
<a href="http://www.codecogs.com/eqnedit.php?latex=\frac{1}{2}||\theta||^2" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\frac{1}{2}||\theta||^2" title="\frac{1}{2}||\theta||^2" /></a>

can be replaced as
<a href="http://www.codecogs.com/eqnedit.php?latex=(\theta^T&space;x^i&space;)=p^i*||\theta||=\theta_1x^i_1&plus;\theta_2x^i_2" target="_blank"><img src="http://latex.codecogs.com/gif.latex?(\theta^T&space;x^i&space;)=p^i*||\theta||=\theta_1x^i_1&plus;\theta_2x^i_2" title="(\theta^T x^i )=p^i*||\theta||=\theta_1x^i_1+\theta_2x^i_2" /></a>

The constraints we defined earlier \
(θT x) >= 1 if y = 1\
(θT x) <= -1 if y = 0\
Can be replaced/substituted with the constraints\
pi * ||θ|| >= 1 if y = 1\
pi * ||θ|| <= -1 if y = 0

picture

θ is always at 90 degrees to the decision boundary 

picture

We know we need p1 * ||θ|| to be bigger than or equal to 1 for positive examples. 
**If p is small, 
means that ||θ|| must be pretty large**, and it is the same as negative example

picture

Now if you look at the projection of the examples to θ we find that p1 becomes large and ||θ|| can become small

## 4. Kernels: Adapting SVM to non-linear classifiers
Come up with a complex set of polynomial features to fit the data. Have hθ(x) which Returns 1 if the combined weighted sum of vectors (weighted by the parameter vector) is less than or equal to 0 非线性分类里面暂且降低要求，只要大于0即返回1. (从以前学过的多项式回归演变来)

## 4.1 Gaussian Kernel

hθ(x) = θ0+ θ1f1+ θ2f2 + θ3f3 f是high polynomial terms，除此之外还有没有替代多项式的表达方式呢？Is there a better choice of feature f than the high order polynomials?

Pick three points in that space and they are called landmarks

<a href="http://www.codecogs.com/eqnedit.php?latex=f_1=&space;exp(-&space;\frac{||&space;x&space;-&space;l_1&space;||^2&space;}{2\sigma&space;_2})" target="_blank"><img src="http://latex.codecogs.com/gif.latex?f_1=&space;exp(-&space;\frac{||&space;x&space;-&space;l_1&space;||^2&space;}{2\sigma&space;_2})" title="f_1= exp(- \frac{|| x - l_1 ||^2 }{2\sigma _2})" /></a>\
This **similarity functio**n is called a kernel\
This function is a **Gaussian Kernel**
- || x - l1 || is the euclidean distance between the point x and the landmark l1 squared
- σ is the standard deviation. 
σ2 is commonly called the variance
- Say x is far from a landmark. then the f is close to zero

## 4.2 What does σ do?
Below σ^2 = 0.5
picture

below σ^2 = 3
picture

## 4.3 what kinds of hypotheses can we learn?
With training examples x we predict "1" when
θ0+ θ1f1+ θ2f2 + θ3f3 >= 0\
For our example, lets say we've already run an algorithm and got the
θ0 = -0.5
θ1 = 1
θ2 = 1
θ3 = 0

picture
first point,  f1 will be close to 1, but f2 and f3 will be close to 0, θ0+ θ1f1+ θ2f2 + θ3f3 >= 0\
-0.5 + 1 + 0 + 0 = 0.5 --> predict 1

another point far away from all three, This equates to -0.5. 
So we predict 0

## 4.4 Implementation of kernels
### 4.4.1 Choosing the landmarks
One landmark per location per training example\
Means our features measure how close to a training set example something is

If we had a training example - features we compute would be using (xi, yi)\
So we just cycle through each landmark, calculating how close to that landmark actually xi is\
f1i, = k(x**i**, l1)\
f2i, = k(x**i**, l2)\
...\
fmi, = k(x**i**, lm)

### 4.4.2 SVM hypothesis prediction with kernels
Predict y = 1 if (θ^T f) >= 0\
θ = [m+1 x 1]   f = [m +1 x 1] 

### 4.4.3 SVM training with kernels
picture

By solving this minimization problem you get the parameters θ for your SVM

In this setup, m = n, 
Because number of features is the number of training data examples we have, and also the number of parameter θ

### 4.4.4 SVM parameters (C)
C plays a role similar to 1/LAMBDA (where LAMBDA is the regularization parameter)
- Large C gives a hypothesis of low bias high variance --> overfitting
- Small C gives a hypothesis of high bias low variance --> underfitting

### 4.4.5 SVM parameters (σ2)
Parameter for calculating f values
- Large σ2 - f features vary more smoothly - higher bias, lower variance
- Small σ2 - f features vary abruptly - low bias, high variance

## 5. SVM - implementation and use
### 5.1 Choosing a kernel
- When would you chose a Gaussian? If n is small and/or m is large
- When would you chose a linear kernel? If n is large and m is small then
Lots of features, few examples

Other choice of kernel
### 5.2 Logistic regression vs. SVM
- If n (features) is large vs. m (training set):  logistic regression or SVM with a linear kernel
- n is small and m is intermediate: Gaussian kernel is good
- n is small and m is large: you should manually create or add more features, then use logistic regression of SVM with a linear kernel

**Logistic regression and SVM with a linear kernel are pretty similar**.
Do similar things. Get similar performance. SVM has a convex optimization problem - so you get a global minimum

**For all these regimes a well designed NN should work**

# Lecture 13 Cluster
## 1. Unsupervised learning - introduction
- Supervised learning\
Given a set of labels, fit a hypothesis to it
- Unsupervised learning\
Try and determining structure in the data. 
Clustering algorithm groups data together based on data features

## 2. K-means algorithm
### 2.1 Algorithm overview
Take unlabeled data and group into two clusters

1. Randomly allocate two points as the cluster centroids随机定几个中心点
2. Cluster assignment step. Go through each example and depending on if it's closer to the red or blue centroid assign each point to one of the two clusters看example离哪个中心点近，就标记为那个中心点所属颜色\
picture
3. Move centroid step. Take each centroid and move to the average of the correspondingly assigned data-points 移动中心点到对应颜色的examples平均数位置\
picture
4. Repeat 2) and 3) until convergence

**Formal definition**:
1. **Input**: 
    - K (number of clusters in the data)
    - Training set {x1, x2, x3 ..., xn) 
2. Algorithm:
    - Randomly initialize K cluster centroids as {μ1, μ2, μ3 ... μK}
    - picture

### 2.2 K-means for non-separated clusters
K-means is applied to datasets where there aren't well defined clusters
混沌状态

picture

picture

So creates three clusters, even though they aren't really there

## 3. K means optimization objective
K-means has an optimization objective like the supervised learning functions we've seen
- this is useful because it helps for debugging
- Helps find better clusters

notation:
- c^i is the index of clusters {1,2, ..., K} to which x^i is currently assigned
- μ_k, is the cluster associated with centroid k
- <a href="http://www.codecogs.com/eqnedit.php?latex=\mu&space;_c^i," target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mu&space;_c^i," title="\mu _c^i," /></a>, is the cluster centroid of the cluster to which example xi has been assigned to

our optimization value:
picture
picture

Means that when the example is very close to the cluster, this value is small

This is sometimes called the distortion (or distortion cost function)

Summary:
1. cluster assigned step is minimizing J(...) with respect to c1, c2 ... ci 保持μ不变调整c使J下降
2. move centroid step 保持c不变调整μ使J下降

## 4. Random initialization
Have number of centroids set to less than number of examples (K < m) (if K > m we have a problem)0\
Randomly pick K training examples and Set μ1 up to μK to these example's values

Risk of **local optimum**\
If this is a concern, we can do multiple random initializations
See if we get the same result - many same results are likely to indicate a global optimum

picture

picture

1. Randomly initialize K-means
2. For each 100 random initialization run K-means
3. Then compute the distortion on the set of cluster assignments and centroids at convergent
4. End with 100 ways of cluster the data
Pick the clustering which **gave the lowest distortion** 不同初始化下取J的最小值

If K is larger than 10, then multiple random initializations are less likely to be necessary

## 5. Choose the number of clusters
Normally use visualizations to do it manually
### 5.1 Elbow method
As K increases J(...) minimum value should decrease\
Look for the "elbow" on the graph

picture

Risks\
Normally you don't get a a nice line -> no clear elbow on curve

### 5.2 For a later/downstream purpose

Could consider the cost of making extra sizes vs. how well distributed the products are

applied problem may help guide the number of clusters

# Lecture 14 Dimensionality Reduc1on
## 1. Motivation
### 1.1 Motivation 1: Data compression
- Speeds up algorithms
- Reduces space used by data for them

you've collected many features - maybe more than you need.
Can you "simply" your data set in a rational and useful way?\
ex. Helicopter flying - do a survey of pilots (x1 = skill, x2 = pilot enjoyment) \
These features may be highly correlated

picture

picture

<a href="http://www.codecogs.com/eqnedit.php?latex=R^2" target="_blank"><img src="http://latex.codecogs.com/gif.latex?R^2" title="R^2" /></a> 表示二维

### 1.2 Motivation 2: Visualization
improve how we display information

picture

picture

Reduce 50D -> 2D

Typically you don't generally ascribe meaning to the new features. It's up to you to asses what of the features can be grouped to form summary features

## 2. Principle Component Analysis (PCA): Problem Formulation
### 2.1 VS linear regression
PCA:
- orthogonal distance
- features are treated equally

linear regression:
- VERTICAL distance between point
- trying to predict "y"


### 2.2 algorithm

Find k vectors (u(1), u(2), ... u(k)) onto which to project the data to minimize the **projection error**

picture

As an aside, you should normally do **mean normalization** and **feature scaling** on your data before PCA\
(xj - μj) / sj 

picture

Need to compute two things:
- Compute the **u vectors: The new planes**
- Need to compute the z vectors: **z vectors are the new, lower dimensionality feature vectors**

#### algorithm description
1. Preprocessing
2. Calculate sigma (covariance matrix)
    - picture
    - Σ (greek upper case sigma) - NOT summation symbol
3. Calculate eigenvectors with svd
    - [U,S,V] = svd(sigma)
    - svd = singular value decomposition
    - Turns out the columns of **U are the u vectors** we want!
4. Take k vectors from U (Ureduce= U(:,1:k);)
    - picture
5. Calculate z (z =Ureduce' * x;)
    - <a href="http://www.codecogs.com/eqnedit.php?latex=z&space;=&space;(U_{reduce})^T&space;*&space;x" target="_blank"><img src="http://latex.codecogs.com/gif.latex?z&space;=&space;(U_{reduce})^T&space;*&space;x" title="z = (U_{reduce})^T * x" /></a>
    - Generates a matrix which is k * 1

## 3. Reconstruction from Compressed Representation
<a href="http://www.codecogs.com/eqnedit.php?latex=x_{approx}&space;=&space;U_{reduce}&space;*&space;z" target="_blank"><img src="http://latex.codecogs.com/gif.latex?x_{approx}&space;=&space;U_{reduce}&space;*&space;z" title="x_{approx} = U_{reduce} * z" /></a>

We lose some of the information (i.e. everything is now perfectly on that line) but it is now projected into 2D space

pciture

## 4. Choosing the number of Principle Components

picture

Ratio between averaged squared projection error with total variation in data\
Want ratio to be small - means we retain 99% of the variance 保留了99%的方差

The numerator is small when xi = xapproxi\
i.e. we **lose very little information** in the dimensionality reduction, so when we decompress we regenerate the same data\
So we chose k in terms of this ratio

## 5. Advice for Applying PCA
1. Apply PCA to x vectors. This gives you a new training set 
Each vector can be re-associated with the labels z.
2. Take the reduced dimensionality data set and feed to a learning algorithm\
Use y as labels and z as feature vector
3. we use those learned parameters for our
Cross validation data,
Test set (also reduced dimension)

Before implementing PCA, first try running whatever you want to 
do with the original/raw data. Only if that doesn't do what 
you want, then implement PCA.

# Lecture 15 Anomaly Detection
## 1. problem motivation
access this model using p(x). What is the probability that example x is normal?

if p(xtest) < ε --> flag this as an anomaly\
if p(xtest) >= ε --> this is OK\
**ε is some threshold probability value which we define, depending on how sure we need/want to be**

## 2. The Gaussian(Normal) distribution
**也就是正态分布**

P(x : μ , σ^2) (probability of x, parameterized by the mean and squared variance)
picture

picture

area always the same

### algorithm
The problem of estimation this distribution is sometimes call the problem of **density estimation**\

Unlabeled training set of m examples, Model P(x) from the data set\
picture

## 3. Developing and evaluating and anomaly detection system
**an example**
You have some labeled data 
Split into \
- Training set: 6000 good engines (y = 0)
- CV set: 2000 good engines, 10 anomalous
- Test set: 2000 good engines, 10 anomalous
- Ratio is 3:1:1

Algorithm evaluation
1. Take trainings set { x1, x2, ..., xm }
**Fit model p(x)** 
2. On cross validation and test set, test the example x
    - y = 1 if p(x) < epsilon (anomalous)
    - y = 0 if p(x) >= epsilon (normal)
3. Think of algorithm a trying to predict if something is anomalous.
But you have a label so can check!
4. Compute F1-score. Then **pick the value of epsilon which maximizes the score on your CV set**
5. Do final algorithm evaluation on the test set

## 4. Anomaly detection vs. supervised learning
Anomaly detection
- Very small number of positive examples 极少数是不正常的y=1
- Have a very large number of negative examples 大部分是正常的y=0
- Many "types" of anomalies, but knows what they don't look like
- ex. Fraud detection; Fraud detection; 

Supervised learning
- Reasonably large number of positive and negative examples
- Have enough positive examples to give your algorithm 
- ex. Weather prediction; Email/SPAM classification

## 5. Choosing features to use
### 5.1 Non-Gaussian features. 
Plot a histogram of data to check\
Often still works if data is non-Gaussian

i.e. if you have some feature x1, replace it with 
- log(x1)
- Or do log(x1+c)
- Or do x1/2
- Or do x1/3

### 5.2 Error analysis
Can looking at that example help **develop a new feature (x2)** which can help distinguish further anomalous 

ex. We suspect CPU load and network traffic grow linearly with one another\
New feature - CPU load/network traffic

## 6. Multivariate Gaussian distribution
because our Normal Distribution function makes probability prediction in concentric circles around the the means of both 高斯分布成一个同心圆

picture

我们要的是这样
picture

### Algorithm
picture

μ - which is an n dimensional vector (where n is number of features)
Σ - which is an [n x n] matrix - the covariance matrix

picture

### Original model vs. Multivariate Gaussian
Original Gaussian model
- need to make extra features
- cheaper computationally
- Scales much better to very large feature vectors. Scales much better to very large feature vectors
- Scales much better to very large feature vectors

Multivariate Gaussian model
- Needs for m > 10n or more, no redundant features  --> or non-invertible
- Less computationally efficient
- Can capture feature correlation

























