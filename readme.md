# stable-Lipchitz SCMs: D-separation and Cyclic Causality

Despite being the natural modeling choice for phenomena involving feedback, cyclic causality has seen limited use because many of the convienent guarentees of acylic SCMs fail to hold for general cyclic SCMs. One well-known hurdle to this modeling option is that d-separation fails to hold for general cyclic SCMs.

We present research about the validity of d-separation (aka. the dGMP: directed global Markov property) for cyclic SCMs. Specifically, we propose the class of stable-Lipschitz SCMs, which generalize acyclic SCMs and are contained by simple SCMs. Stable-Lipschitz SCMs behave like linear SCMs asyptotically, and indeed, are proven to satsify the dGMP (currently with some additional constraints).
These results are verified numerically, and the validity of the backdoor adjustment criterion is proven as well. We further show that stable-Lipschitz SCMs are closed under interventions, and that the interventional distributions of simple SCMs satisfy the dGMP.

Lastly, we discuss future directions for research, including the Pearl Causal Hierarchy, the do-calculus, and most exciting, the possibility of causal identification with multiple equilibria (e.g. in game theory, economics).