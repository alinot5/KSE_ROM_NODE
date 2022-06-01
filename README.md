# KSE_ROM_NODE

This repository contains code for training autoencoders on data generated from the Kuramoto-Sivashinsky Equation (KSE) at a domain size of L=22. The code varies the latent dimension of each autoencoder keeping everything else the same. Then the reconstruction error on test data is plotted for all of these autoencoders. A clear drop in error appears at a dimension of 8, which coincides with the expected dimension of the attractor at this domain size.

Autoencoder.py is a standard dense autoencoder and Autoencoder_hyb.py is the hybrid autoencoder described in https://arxiv.org/abs/2001.04263 and https://arxiv.org/abs/2109.00060. The losses for the two are different, but the mean squared error (MSE) output at the end is the same.

Conda_list.txt contains a list of all the packages used in the Anaconda environment I used to validate the code. Some key packages include: numpy, matplotlib, pandas, tensorflow, scipy, and sklearn.
