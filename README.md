# Graphical Latent State Space Models

This is a Python library that extends traditional generative time series models, such as hidden markov models, linear dynamical systems and their extensions, to graphical models.

With this framework, a user can do Bayesian inference over graphical structures.  One use case is doing inference over a [pedigree chart](https://en.wikipedia.org/wiki/Genogram), where phenotypes (observations) are emitted based on each person's genotype (latent state), and the genotypes of individuals are linked through their ancestor tree.

<img src="examples/hmm_em.gif">
