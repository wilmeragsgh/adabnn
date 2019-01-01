# AdaBnn

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/wilmeragsgh/adabnn/blob/master/LICENSE)

Code related to thesis work: **"AdaBnn: Binarized Neural Networks trained with adaptive structural learning"** [[pdf]](https://mega.nz/#!SkFhxCSI!YQO-ZYQl5tFlEGpkl2nR13zzFAzJeT5iCgZt8AzvIsQ)

This repository contains currently two colaboratoy notebooks, [English](https://colab.research.google.com/github/wilmeragsgh/adabnn/blob/master/experiments_en.ipynb\) and [Spanish](https://colab.research.google.com/github/wilmeragsgh/adabnn/blob/master/experiments_es.ipynb\)
that document experiments made with an experimental **Keras** based implementation of the AdaNet algorithm presented in “[AdaNet: Adaptive Structural Learning of Artificial Neural Networks](http://proceedings.mlr.press/v70/cortes17a.html)” at [ICML 2017](https://icml.cc/Conferences/2017), for learning the structure of a neural network as an ensemble of subnetworks. Also, AdaBnn is presented as a modification of AdaNet that imposed binary constraints on running time to try to increase performance in terms of time and as a way of regularization based on "[Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830)".

Also, separated code is included containing Adanet and AdaBnn implementations with its documentation.

## Some findings

According to the experiments provided in the notebooks:

* Binarizing the weights of the network, in the case of adaptive structural learning, have similar effects of having a high mutation rate in genetic algorithms, the pattern of learning is much more hard to follow between iterations, doesn't hold incremental performance in T iterations.
* Adam optimization it's more appropiate for such AdaBnn structure in most cases as well as fewer iterations (T parameter on the paper).
* Currently Binarizing AdaNet doesn't hold that much of improvement but it could be a start point for adding constraints for weights/activations as regularization method for adaptive structural learning.

## Futher work

Further work may include the binarization process as part of the convolution sub network, which was the initial proposal of (M Courbariaux ,2016).

## Example

After importing dependencies and declaring each model (proposed implementation of AdaNet and AdaBnn) they can be trained and used with:

```python
epochs = 25 
B = 150 # neurons on each layer of the sub network  
T = 2 # number of iterations of the adanet algorithm (maximum complexity of the model)
conf = dict({
    'network': {
        'activation': BinaryRelu, 
        'output_activation': 'sigmoid',# softmax for multiple clases
        'optimizer': keras.optimizers.Adam(lr=0.0001),#Adam
        'loss': 'binary_crossentropy' # F for adanet traditional
        },
    'training':{
        'batch_size': 100,
        'epochs': epochs
        },
    'adabnn':{
        'B': B,
        'T': T,
        'delta': 1.01,
        'seed': 42
        }
    })
model, rend = AdaBnn(x_train,y_train,conf,verbose=0) # rend will have accuracy/loss at each iteration T
training_results = model.evaluate(x_test,y_test,verbose=0)
```

## Citing this Work

If you use these implementations for academic research, you are encouraged to cite the following:

    @misc{gonzalez2018adabnn,
      author    = {Wilmer Gonzalez},
      title     = {AdaBnn: Binarized Neural Networks trained with adaptive structural learning},
      year      = {2018},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/wilmeragsgh/adabnn}},
    }

## License

Included functions and notebooks are released under the [Apache License 2.0](LICENSE).
