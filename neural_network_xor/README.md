# Neural Network From Scratch (XOR Solver)

A simple **2–2–1 Feedforward Neural Network** implemented from scratch using only **NumPy**.

The network is trained using **manual backpropagation** and **gradient descent** to solve the XOR classification problem, demonstrating how neural networks can learn patterns that are not linearly separable.

## Network Architecture

```
Input Layer (2)
        \/
Hidden Layer (2 Neurons, ReLU)
        \/
Output Layer (1 Neuron, Sigmoid)
```

## Features

* Feedforward Neural Network
* 2 Input Neurons
* 2 Hidden Neurons (ReLU Activation)
* 1 Output Neuron (Sigmoid Activation)
* Binary Cross-Entropy Loss
* Manual Backpropagation
* Gradient Descent
* Fully Vectorized NumPy Implementation

## Concepts Practiced

* Forward Propagation
* Backpropagation
* Activation Functions (ReLU & Sigmoid)
* Binary Classification
* Binary Cross-Entropy Loss
* Gradient Computation
* Weight & Bias Updates
* Matrix Operations

## Experiments

The same network architecture was tested on multiple Boolean logic gates:

* XOR
* XNOR
* AND
* OR

The network successfully learned all four tasks, while XOR and XNOR demonstrate the advantage of neural networks over linear models such as Logistic Regression.

## Technologies Used

* Python
* NumPy

---

## Output

```
Epoch 0:
 W1 = [[0.17634169 0.04004897]
 [0.09784836 0.22410263]] b1 = [ 0.00227513 -0.00119056]
 W2 = [[ 0.18668244]
 [-0.09777192]] b2 = [-0.00031764]
 Loss = 0.6931941134672226

Predictions:
[[1]
 [0]
 [1]
 [1]]
Probabilities:
[[0.5       ]
 [0.49909469]
 [0.507258  ]
 [0.50635286]]
Hidden activations:
[[0.         0.        ]
 [0.0978738  0.22408932]
 [0.17640523 0.04001572]
 [0.27427903 0.26410504]]

Epoch 1000:
 W1 = [[1.66997967 2.3590802 ]
 [1.66882522 2.35856142]] b1 = [ 0.00497144 -2.36300943]
 W2 = [[ 2.35473424]
 [-4.05913616]] b2 = [-1.31674823]
 Loss = 0.10746883074116279

Predictions:
[[0]
 [1]
 [1]
 [0]]
Probabilities:
[[0.21183481]
 [0.92910222]
 [0.92914399]
 [0.04381099]]
Hidden activations:
[[0.         0.        ]
 [1.66644342 0.00823177]
 [1.66760032 0.00874632]
 [3.33483252 2.37005472]]

Epoch 2000:
 W1 = [[2.27899206 3.09579416]
 [2.2788545  3.09565071]] b1 = [-5.30006787e-04 -3.09656341e+00]
 W2 = [[ 3.21635853]
 [-5.33870577]] b2 = [-2.9139577]
 Loss = 0.021375159494042607

Predictions:
[[0]
 [1]
 [1]
 [0]]
Probabilities:
[[0.05151108]
 [0.98786129]
 [0.98785736]
 [0.00815211]]
Hidden activations:
[[0.00000000e+00 0.00000000e+00]
 [2.27670723e+00 1.77179861e-03]
 [2.27684448e+00 1.91577782e-03]
 [4.55537846e+00 3.09809851e+00]]

Epoch 3000:
 W1 = [[2.48902246 3.35750626]
 [2.48896557 3.35744208]] b1 = [ 4.03547794e-05 -3.35761319e+00]
 W2 = [[ 3.5137023 ]
 [-5.79335662]] b2 = [-3.57358234]
 Loss = 0.010750965436054572

Predictions:
[[0]
 [1]
 [1]
 [0]]
Probabilities:
[[0.02730281]
 [0.99434325]
 [0.99434438]
 [0.00396902]]
Hidden activations:
[[0.         0.        ]
 [2.48821266 0.        ]
 [2.48826965 0.        ]
 [4.97708697 3.3556107 ]]

Epoch 4000:
 W1 = [[2.61110124 3.51036036]
 [2.61106766 3.51031477]] b1 = [-5.62356635e-04 -3.51060010e+00]
 W2 = [[ 3.68655121]
 [-6.05910542]] b2 = [-3.98377162]
 Loss = 0.007008531636167519

Predictions:
[[0]
 [1]
 [1]
 [0]]
Probabilities:
[[0.0182812 ]
 [0.99643991]
 [0.99643937]
 [0.00244733]]
Hidden activations:
[[0.00000000e+00 0.00000000e+00]
 [2.60997205e+00 5.91111401e-04]
 [2.61000558e+00 6.36785100e-04]
 [5.22097069e+00 3.51112010e+00]]

Epoch 5000:
 W1 = [[2.6958503  3.61674317]
 [2.6958268  3.61671609]] b1 = [-8.10729692e-05 -3.61709655e+00]
 W2 = [[ 3.80654941]
 [-6.2440413 ]] b2 = [-4.28036689]
 Loss = 0.0051408832476212175

Predictions:
[[0]
 [1]
 [1]
 [0]]
Probabilities:
[[0.0136522 ]
 [0.99747259]
 [0.99747239]
 [0.00175432]]
Hidden activations:
[[0.00000000e+00 0.00000000e+00]
 [2.69535806e+00 2.55443604e-04]
 [2.69538154e+00 2.82563346e-04]
 [5.39113477e+00 3.61711933e+00]]

Epoch 6000:
 W1 = [[2.76022467 3.69784447]
 [2.76020685 3.69782189]] b1 = [-7.50115233e-04 -3.69778040e+00]
 W2 = [[ 3.8976994 ]
 [-6.38471156]] b2 = [-4.51225482]
 Loss = 0.004037144249556482

Predictions:
[[0]
 [1]
 [1]
 [0]]
Probabilities:
[[0.01085947]
 [0.99806598]
 [0.99806611]
 [0.00135714]]
Hidden activations:
[[6.33867149e-05 0.00000000e+00]
 [2.76021402e+00 0.00000000e+00]
 [2.76023185e+00 0.00000000e+00]
 [5.52038249e+00 3.69723611e+00]]

Epoch 7000:
 W1 = [[2.81186842 3.76278239]
 [2.81185402 3.76276233]] b1 = [ 1.99631674e-04 -3.76293634e+00]
 W2 = [[ 3.97082339]
 [-6.49765417]] b2 = [-4.70242629]
 Loss = 0.0033112703988252556

Predictions:
[[0]
 [1]
 [1]
 [0]]
Probabilities:
[[0.0089932 ]
 [0.99843956]
 [0.99843944]
 [0.00108717]]
Hidden activations:
[[0.00000000e+00 0.00000000e+00]
 [2.81180477e+00 2.33242853e-04]
 [2.81181917e+00 2.53324485e-04]
 [5.62362620e+00 3.76309253e+00]]

Epoch 8000:
 W1 = [[2.85484688 3.81692257]
 [2.85483446 3.81690617]] b1 = [-5.32694982e-04 -3.81704839e+00]
 W2 = [[ 4.03167731]
 [-6.59168861]] b2 = [-4.86349983]
 Loss = 0.0028006884870246575

Predictions:
[[0]
 [1]
 [1]
 [0]]
Probabilities:
[[7.66748396e-03]
 [9.98701253e-01]
 [9.98701177e-01]
 [9.06054739e-04]]
Hidden activations:
[[6.96260852e-05 0.00000000e+00]
 [2.85486450e+00 2.01235703e-04]
 [2.85487692e+00 2.17651412e-04]
 [5.70967180e+00 3.81718853e+00]]

Epoch 9000:
 W1 = [[2.89155495 3.86323531]
 [2.89154426 3.86322193]] b1 = [-2.85814336e-04 -3.86321666e+00]
 W2 = [[ 4.08365272]
 [-6.67205262]] b2 = [-5.00314486]
 Loss = 0.002422492878794526

Predictions:
[[0]
 [1]
 [1]
 [0]]
Probabilities:
[[6.67284258e-03]
 [9.98890793e-01]
 [9.98890842e-01]
 [7.74869465e-04]]
Hidden activations:
[[0.         0.        ]
 [2.89107695 0.        ]
 [2.89108764 0.        ]
 [5.78259777 3.86285284]]
```
