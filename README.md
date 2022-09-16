# Neuron with Steady Response Leads to Better Genralization

This repository is the official implementation of [Neuron with Steady Response Leads to Better
Generalization](https://arxiv.org/pdf/2111.15414.pdf), accepted by **NeurIPS 2022**. 

## Abstract
Regularization can mitigate the generalization gap between training and inference by introducing inductive bias. Existing works have already proposed various inductive biases from diverse perspectives. However, none of them explores inductive bias from the perspective of class-dependent response distribution of individual neurons. In this paper, we conduct a substantial analysis of the characteristics of such distribution. Based on the analysis results, we articulate the Neuron Steadiness Hypothesis: the neuron with similar responses to instances of the same class leads to better generalization. Accordingly, we propose a new regularization method called Neuron Steadiness Regularization (NSR) to reduce neuron intra-class response variance. Based on Complexity Measure, we theoretically guarantee the effectiveness of NSR for improving generalization. We conduct extensive experiments on Multilayer Perceptron, Convolutional Neural Network, and Graph Neural Network with popular benchmark datasets of diverse domains, which show that our Neuron Steadiness Regularization consistently outperforms the vanilla version of models with significant gain and low additional computational overhead. 

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
```
## Training

To train the models (MLP-4 by default) in the paper, run this command:

Vanilla model:
```
python fdnn_vanilla.py
```
Vanilla + NSR:
```train
python fdnn_nsr.py
```
## Model
This table shows the network architecture of DNN model.
|Model| Hidden layer dimension|
|-----|----------------|
|MLP-3|[100]|
|MLP-4|[256, 100]|
|MLP-6|[256, 128, 64, 32]|
|MLP-8|[256, 128, 64, 32, 32, 16]|
|MLP-10|[256, 128, 64, 64, 32, 32, 16, 16]|
## Results

Our model achieves the following results on [MNIST](http://www.dia.fi.upm.es/~lbaumela/PracRF11/MNIST.html) (for MLP model), [cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html) (for ResNet-18 and VGG-19), ImageNet (for ResNet-50). Notice that for ImageNet, the report metric is acc on hit@5, while others are accuracy. 

| Model        | MLP-3     | MLP-4     | MLP-6     | MLP-8     | MLP-10    | ResNet-18 | VGG-19    | ResNet-50 |
| ------------ | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| Vanilla      | 3.09±0.10 | 2.29±0.07 | 2.44±0.09 | 2.87±0.09 | 3.06±0.06 | 4.22±0.07 | 9.19±0.17 | 7.82±0.07 |
| Vanilla +NSR | 2.80±0.08 | 1.64±0.04 | 1.76±0.06 | 1.98±0.09 | 1.72±0.14 | 3.74±0.08 | 8.09±0.10 | 6.98±0.08 |
| Gain         | 9.39%     | 28.38%    | 27.87%    | 30.87%    | 43.79%    | 11.37%    | 11.97%    | 10.74%    |

The hyper parameter (lambda for balancing the cross entropy loss and NSR) we used for producing above results are shown below. (Hyperparameter is automatically selected by the NNI framework)
| Model          | MLP-3    | MLP-4   | MLP-6   | MLP-8   | MLP-10  | ResNet-18 | VGG-19    |
| -------------- |--------- | ------- | --------|---------|-------  | ----------|-----------|
| lambda         | 0.02475  | 7.4865  |0.0538   |0.2542   |0.4158   |0.5        |0.11       |

## Clarification

This work is conducted during my internship in Microsoft Reserch Asia. The paper has been revised for many times. And the model hyperparameters, i.e., learning rate, batch size, has changed many times as the anonymous reviewers suggested. We are checking that whether some existing training framework hyperparameters are correct since we lose the access to some servers without proper backup. If you have any questions on reproduce our results, please do not hesitate to write the issue or directly email me: haitaoma@msu.edu. Also, if you do not understand the code, email me. We can have a zoom meeting to help you figure out our code.

## License

All content in this repository is licensed under the [MIT license](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt).
