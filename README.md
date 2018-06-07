# Graphical Generative Models

This is a Python library that extends traditional generative time series models, such as hidden markov models, linear dynamical systems and their extensions, to graphical models.

With this framework, a user can do Bayesian inference over graphical structures.  One use case is doing inference over a [pedigree chart](https://en.wikipedia.org/wiki/Genogram), where phenotypes (observations) are produced through some generative process that is a function of the person's genotype (latent state) and the genotypes of individuals are linked through their ancestor tree.

This library extends previous graphical generative model techniques by adding support for hypergraphs (to incorporate ordering constraints on parents) and provides an efficient way to deal with directed cycles (for efficient message passing that is independent of the cycle's diameter).

The structure of this repo is heavily inspired by https://github.com/mattjj/pyhsmm and its other associated components (pybasicbayes, pyslds, etc.)

# Project Goal
** Problem Scope **
The current machine learning development pipeline at the highest level looks something like this:
- Data -> [ MACHINE LEARNING ] -> Predictions <- Expert opinion

This pipeline is present in all branches of machine learning from supervised learning to generative models.  At the end of the day, data is fed through a black box that transforms inputs into outputs as it sees fit.  It is then the expert's responsibility to evaluate the model's performance.

The problem with this workflow is that an expert has no say in what the model should learn.  Feature engineering and latent state models are two methods used to try to assign meaning to what the model is learning, but don't really solve the fundamental problem of controlling what the model learns.  At the end of the day, the machine learning black box finds a way to correlate data in a way that maximizes some objective function.

An ideal pipeline would look like this:
- ( Expert opinion, data ) -> [ MACHINE LEARNING ] -> Predictions

In this pipeline, an expert should be able to influence what the black box model learns.  The expert should be able to use knowledge of the problem at hand to define a deterministic algorithm to solve the problem, and leave the parts that are usually left to "human intuition" to the machine learning black box.

This project sets out to implement the ideal pipeline using graphical generative models.

The key insight in this project is using hyper-graphs instead of regular graphs.  One of the key advantages hyper-graphs have over graphs are that it is possible to keep an ordering of the parents for every child node.  This means that it is possible to build algorithms using latent state variables in a human interpretable way.  Then by using Bayesian modeling, it should, in theory, be possible to condition the model on known values ( both observed and unobserved ) and define priors over the system dynamics.

Here's a concrete example of a problem that can be solved using this approach that cannot be solved using existing machine learning methods:

Background:
- Genetic diseases are passed down from generation to generation depending on the parent's genotype.  This means that the probability of a child inheriting a disease is a function of the genetic makeup of each parent.
- 3 common inheritance pattern types are autosomal dominant, autosomal recessive and x-linked recessive.
- It is common to use punnett squares [LINK] to show the transition probabilities associated with one generation.
- [EXAMPLE]
- When a patient goes to a genetic specialist, the specialist creates a pedigree to help determine what the inheritance pattern of the disease is.
- [EXAMPLE]
- The specialist uses logic and intuition determine the inheritance pattern and therefore reduce the number of possible diseases afflicting the patient.
- No deterministic algorithm exists to solve this problem.
- Although punnett squares show probabilities associated with a single generation, the combinations grow super-exponentially with every family member that is introduced.
- In addition to this, sometimes things don't go as expected at the genetic level.
	- De-novo mutations introduce the disease to a family line
	- Incomplete penetrance changes the phenotype associated with the true genotype for a person, so they may never exhibit symptoms of the disease they carry
	- Something else may go wrong
	- Also a person can be undiagnosed
- The probabilities of any of these happening are also unknown

The Problem:
- Is it possible to determine the inheritance pattern of a disease from a pedigree?

Why this problem is difficult:
- Need to use structural information where order matters ( male vs female )
- Must have probabilistic bounds associated with every value because false negatives and false positives have very high impact in the medical world
- Data is limited, usually on paper, and in the form of a drawing
- The complexity of the human body is reduced to a shape that is either shaded ( disease is diagnosed ) or unshaded
- Unshaded means not diagnosed, which can mean unaffected, carrying with incomplete penetrance or carrying but undiagnosed.

The solution this project seeks:
- Assume data is readily available in hyper-graph form (see my other repo)
- Model a pedigree as a switching count regression graphical model
	- Need a hypergraph message passing algorithm to capture relationships between all family members
	- Hidden markov model isn't good enough (see other repo)
	- Need a continuous latent state to capture everything that can go wrong at the genetic level
	- Modes are true autosome / chromosome pair
	- Latent state is the "human intuition" layer where the machine learning black box does its magic
	- Emissions are bernoulli random variables that model the shade of the shape associated with a person

Advantages of this solution:
- Uses structural information of genetic trees
- Full bayesian inference
- Can place prior over mode transitions (hard code mendellian genetics!)
- Can condition on modes easily (if a person is diagnosed with an AR disease, they MUST have 2 mutated alleles)
- Once the repo is built, it should be easy to use other generative model techniques on top of the implemented models.


Future long term goals:
- SVAE to combine advantages of human direction and neural network data correlation
- Train in Info-GAN fashion so that the neural network layer actually tries to use the human coded algorithm in its predictions
- Implement graph resampling techniques to learn algorithms themselves


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
