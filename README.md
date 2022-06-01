# KSE_ROM_NODE

This repository contains code for training autoencoders on data generated from the Kuramoto-Sivashinsky Equation (KSE) at a domain size of L=22. The code varies the latent dimension of each autoencoder keeping everything else the same. Then the reconstruction error on test data is plotted for all of these autoencoders. A clear drop in error appears at a dimension of 8, which coincides with the expected dimension of the attractor at this domain size.
