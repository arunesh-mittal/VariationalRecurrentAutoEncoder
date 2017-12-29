# Variational Recurrent Autoencoder (VRAE) in TensorFlow


Implementation of VRAE paper: "Fabius, Otto, and Joost R. van Amersfoort. "Variational recurrent auto-encoders." arXiv preprint arXiv:1412.6581 (2014)." in Tensorflow on MIDI data.

Original Paper: https://arxiv.org/abs/1412.6581  

Primary Requirements:  
[Tensorflow](https://www.tensorflow.org/): [Release 1.4](https://github.com/tensorflow/tensorflow/releases)  
[Python 3.0](https://www.python.org/download/releases/3.0/)

## Summary

The Variational Recurrent Auto-Encoder (VRAE) [1] is a generative model for unsupervised learning of time-series data. It combines the strengths of recurrent neural networks (RNNs) and stochastic gradient variational bayes (SGVB) [2]. In the typical VAE, the log-likelihood for a data point $i$ can be written as:

<img src="https://github.com/arunesh-mittal/VRAE/blob/master/readme_misc/Eq_1.png" width="450" />


The evidence lower bound (ELBO) is:


<img src="https://github.com/arunesh-mittal/VRAE/blob/master/readme_misc/Eq_2.png" width="450" />

 Where, the variational distribution and likelihood function are parametrized with recurrent neural networks. In the standard VAE, these networks are dense feed forward networks. VRAE extends this framework by replacing the encoder and decoder dense feed forward neural networks with encoder and decoder *recurrent* neural nets.


## TensorBoard Graph
<img src="https://github.com/arunesh-mittal/VRAE/blob/master/readme_misc/VRAE_Graph.png" width="450" />

## Data
The MIDI data used is available [here](http://www-etud.iro.umontreal.ca/~boulanni/icml2012)

##

[1] Fabius, Otto, and Joost R. van Amersfoort. "Variational recurrent auto-encoders." arXiv preprint arXiv:1412.6581(2014).  

[2] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).
