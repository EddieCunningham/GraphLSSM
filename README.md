# Graphical Generative Models

This is a Python library that extends traditional generative time series models, such as hidden markov models, linear dynamical systems and their extensions, to graphical models.

With this framework, a user can do Bayesian inference over graphical structures.  One use case is doing inference over a [pedigree chart](https://en.wikipedia.org/wiki/Genogram), where phenotypes (observations) are produced through some generative process that is a function of the person's genotype (latent state) and the genotypes of individuals are linked through their ancestor tree.

This library extends previous graphical generative model techniques by adding support for hypergraphs (to incorporate ordering constraints on parents) and provides an efficient way to deal with directed cycles (for efficient message passing that is independent of the cycle's diameter).

The structure of this repo is heavily inspired by https://github.com/mattjj/pyhsmm and its other associated components (pybasicbayes, pyslds, etc.)

# Parts of the library
**Distributions**
- Completed
	- Exponential family base
		- Inference over
			- P( x | Ѳ; α ), P( Ѳ | x; α ), P( x, Ѳ; α ), P( Ѳ; α )
		- Conjugate modeling
		- Natural parameters, sufficient statistics, partition function & base measure
		- Uses most of the exponential family "tricks"
	- Normal
	- Inverse Wishart
	- Regression
	- Normal Inverse Wishart
	- Matrix Normal Inverse Wishart
	- Categorical
	- Dirichlet
	- Transition
	- Dirichlet Transition Prior
	- Tensor Normal
		- Generalization of Matrix Normal distribution
	- Tensor Regression
		- Generalization of Regression distribution (intended for graphical Kalman Filtering)
	- Tensor Categorical
		- Generalization of Categorical distribution (intended for graphical Forward Backward Filtering)
	- Gibbs sampling
	- Metropolis Hastings
- TODO
	- Maximum likelihood
	- MAP estimate
	- Expectation maximization
	- Stochastic Variational Inference
	- Cholesky implementation of Normal distribution (and everything else that uses covariance matrices)
	- Tensor versions of other distributions
	- Optimized implementations of classes

**States**
- Completed
	- Forward Backward Filtering
	- Gaussian Forward Backward
	- Switching Linear Dynamical System Mode Filtering
	- Kalman Filtering
	- SLDS State Filtering
	- Graphical Message Passing
		- Can be easily parallelized
		- Not recursive
	- Discrete State Graphical Filtering (Up-Down)
		- Filter over graphical structure
		- Can have directed cycles
		- Uses feedback vertex set cuts to filter
	- Hidden Markov Model State
	- Linear Dynamical System State
- TODO
	- Hidden Semi-Markov Model
	- The rest of the model implementations (SLDS, HDP-HMM, Factorial HMM, etc.)
	- Continuous latent state graphical filtering (Graphical Kalman Filtering)
	- Graphical states and models
	- Add more graphical message passing algorithms
		- Loopy belief propogation
		- Junction tree algorithm
		- Approximating messages via buffering

**Models**
- Completed
	- Hidden Markov Model
	- Linear Dynamical System
- TODO
	- HSMM
	- Other stuff listed above

**Priors**
- Only basic model priors so far
	- MNIW model prior over LDS
	- Dirichlet model prior over rows of HMM parameters
- TODO
	- Dirichlet Process
	- Hierarchical Dirichlet Process
	- ARD Priors
	- Recurrent SLDS Prior

**Other TODO Items**
	- Tutorial
	- Animations
	- Structured Variational Auto-Encoder
	- More tests
	- Pandas integration
	- Tensorflow (?) integration
