# Maximum Causal Entropy IRL

In this project, an extension of the classic Maximum Causal Entropy Inverse Reinforcement Learning algorithm was implemented within the process of a Bachelor thesis. The theoretical basis is given by Ziebart et. al. [^1]. The code for the original version of the algorithm was provided y one of the authors of [^2] in which this code was developed. This basis was modified to fit the specific experiment and also include the new approach of this work.

Within this approach we extend the classic idea of feature expectation matching to feature variance matching as there can be multiple policies that match the feature expectation of the demonstrator that might not be equally good. The variance should be another measure to model the behavior of the demonstrator even more closely. This idea can also be generalized to higher order moments.

## Problem

TBW

## Environment

TBD

## Evaluation

TBW



## References

[^1]: Brian D. Ziebart. Modeling purposeful adaptive behavior with the principle of maximum causal entropy. PhD thesis, USA, 2010. AAI3438449 [Link to the source](https://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf). 

[^2]: Sebastian Tschiatschek, Ahana Ghosh, Luis Haug, Rati Devidze, Adish Singla. Learner-aware Teaching: Inverse Reinforcement Learning with Preferences and Constraints [Link to the source](https://arxiv.org/abs/1906.00429).