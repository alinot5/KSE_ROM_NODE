# Data-Driven Reduced-Order Modeling of Spatiotemporal Chaos with Neural Ordinary Differential Equations

This repository contains code for training autoencoders on data generated from the Kuramoto-Sivashinsky Equation (KSE) at a domain size of L=22, and code for running short trajectories of the reduced order models described in https://arxiv.org/abs/2109.00060 at domain sizes of L=22, 44, and 66.

The codes to run are **./Autoencoder.py, ./Autoencoder_hyb.py, ./Trajectories/Traj22.py, ./Trajectories/Traj44.py, and ./Trajectories/Traj66.py**. Running all of these codes will require many of the packages in Conda_list.txt (not all are necessary). This file contains all of the packages in the Anaconda environment I used to validate the code. Some key packages include: tensorflow, pytorch, sklearn, pandas, seaborn, numpy, scipy, and matplotlib.

In the autoencoder codes, the latent dimension of each autoencoder is varied keeping everything else the same. Then the reconstruction error on normalized test data is plotted for all of these autoencoders. A clear drop in error appears at a dimension of 8, which coincides with the expected dimension of the attractor at this domain size.

Autoencoder.py is a standard dense autoencoder trained on data projected onto the full PCA basis and Autoencoder_hyb.py is the hybrid autoencoder described in https://arxiv.org/abs/2001.04263 and https://arxiv.org/abs/2109.00060. The losses for the two are different, but the mean squared error (MSE) output at the end is the same. Refer to https://arxiv.org/abs/2109.00060 or buildmodel.py for the architecture of the autoencoders.

The training data in 'Data_Train.p' is 20000 snapshots of data separated every 2.5 time units and the testing data is 8000 snapshots of data separated every 2.5 time units. Each autoencoder is trained for 1000 epochs with a learning rate scheduler that drops from 0.001 to 0.0001 halfway through training. Due to the memory limitations of Github and the computational time required to run this code, this is less data and fewer epochs than what was used in https://arxiv.org/abs/2109.00060. With longer training and more data the drop in error becomes more pronounced because the autoencoder performance often tends to slowly increase at higher latent dimensions.

Below are two examples of the MSE vs dimension plots generated when running Autoencoder.py and Autoencoder_hyb.py, resepectively. Due to the stochastic nature of training there will be some variability trial to trial when running this code. Autoencoder.py took ~10 hours to run on a 2.3 GHz Dual-Core Intel Core i5 MacBook Pro. While both Autoencoder.py and Autoencoder_hyb.py took ~4-5 hours to run on a Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz. The models from this training are included in the Models directory.

# Standard Autoencoder
![Comparison](https://user-images.githubusercontent.com/46662557/171653547-c6842626-1877-4a7a-bdec-ce43088709de.png)


# Hybrid Autoencoder
![ComparisonHyb](https://user-images.githubusercontent.com/46662557/171655158-29cf83f0-5c3c-4225-bcca-07082246d951.png)

In addition to the code for training the autoencoders, the Trajectories directory contains trained autoencoders and neural ODEs for KSE data at domain sizes of L=22, 44, and 66 and latent dimensions of d_h=8, 18, and 28. Running these codes (Traj22.py, Traj44.py, and Traj66.py) will take initial conditions from u22.p, u44.p, and u66.p and run them through their respective autoencoder and neural ODE. Then the code will save a plot comparing this trajectory to the true system.
