# KSE_ROM_NODE

This repository contains code for training autoencoders on data generated from the Kuramoto-Sivashinsky Equation (KSE) at a domain size of L=22. The code varies the latent dimension of each autoencoder keeping everything else the same. Then the reconstruction error on test data is plotted for all of these autoencoders. A clear drop in error appears at a dimension of 8, which coincides with the expected dimension of the attractor at this domain size.

Autoencoder.py is a standard dense autoencoder and Autoencoder_hyb.py is the hybrid autoencoder described in https://arxiv.org/abs/2001.04263 and https://arxiv.org/abs/2109.00060. The losses for the two are different, but the mean squared error (MSE) output at the end is the same.

Conda_list.txt contains a list of all the packages used in the Anaconda environment I used to validate the code. Some key packages include: numpy, matplotlib, pandas, tensorflow, scipy, and sklearn.

The training data in 'Data_Train.p' is 20000 snapshots of data separated every 2.5 time units and the testing data is 8000 snapshots of data separated every 2.5 time units. Each autoencoder is trained for 1000 epochs with a learning rate scheduler that drops from 0.001 to 0.0001 halfway through training. Due to the memory limitations of Github and the computational time required to run this code, this is less data and fewer epochs than what was used in https://arxiv.org/abs/2109.00060. With longer training and more data the drop in error becomes more pronounced because the autoencoder performance often tends to slowly increase at higher latent dimensions.

Below are two examples of the MSE vs dimension plots generated when running Autoencoder.py and Autoencoder_hyb.py, resepectively. Due to the stochastic nature of training there will be some variability trial to trial when running this code. Each script took around 4 hours to run on a 2.3 GHz Dual-Core Intel Core i5 2017 MacBook Pro.

![Comparison](https://user-images.githubusercontent.com/46662557/171500127-1900641a-6911-4a16-af77-c4a7fcfc54b5.png)

In addition to the code for training the autoencoders, also included in this repository are trained autoencoders and neural ODEs for KSE data at domain sizes of L=22, 44, and 66 and latent dimensions of d_h=8, 18, and 28. The Traj.py code loads these models and plots short trajectories for them. 
