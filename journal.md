### Week of 1/16

1/19
Sythesized research thoughts from the last few months down to 11 proposals. Scored and ranked these proposals.

1/20
Ran proposals past several peers to get feedback and adjust rankings.

1/21
Leaning towards researching "Prior over Counterfactuals", as a way to inform structural learning.

### Week of 1/23

1/23-25
Investigated Causal Influence Diagrams, as a source for alternative proposals. 

1/26
Adjusted leaning towards "Causal Bounds over PDAGs", as it's significantly more applied and concrete than "Prior over Counterfactuals".

1/27
Setup repository. Came up with a more concrete CID proposal which is surprisingly promising...waiting to see if it survives the 24-hour test.

<!-- Lead up to the old "Incentivized Counterfactual Unfairness" proposal -->

1/28
Updated to the CID proposal, since after a full day it still seems at least as applied, concrete, and tractable as my previous proposal, but much more exciting! Current title: "Quantifying Incentives inherent to Linear Causal Influence Models". 

RESEARCH ROADMAP: I expect there's even more low-hanging, impactful directions to go with this proposal than my previous ones. It's likely that quantitative research hasn't been explored quite yet, but even if it has I think I've come up with enough ideas that there's room for my research. Initially my research steps should consist just of replicating the linearity analysis of Pearlian causal theory (if anything, this is the step I expect may have been done already - I'll contact one of the authors and coordinate); after that, I'm interested in trying to establish conditions on which Goodhart's law implies that lower bounds will almost-surely coincide with the upper bounds, so they in fact become estimates (assessment: somewhat skeptical this will work, but since it'd be so cool it's worth some time to try it!). Perhaps this estimate can then be used to counter-weight the effect of unfairness, etc; but that smells like IPW, which is only valid when Z is admissible for adjustment, so perhaps there's graphical criteria which establish the validity of a fairness-adjustment. Lastly, while fairness is probably the most commonly discussed incentive issue, there are several others well-defined for SCIMs that should be amenable to analogous analysis: value of information, response incentives, value of control and instrumental control incentive. 

WHY EXCITED: What really gets me excited about this avenue of research, is that these results would be algorithm-agnostic, derived solely from the nature of the problem environment and the assumption that the algorithms are optimal. Which actually makes sense! It's basically saying, "some problems are inherently really difficult to navigate fairly", or "some situations will almost always incentivize manipulation" which kind of matches with recurring themes in human stories. 

ETHICAL CONSIDERATIONS: As far as the ethical implications of this research: I'm mostly excited, because SCIMs have already proven themselves to be a necessary and very expressive language; quantifying the implications (even in "first-order Taylor expansion" kind of way) allows us to raise warnings more concretely (the same as quatifying the effects of climate change may have helped governments take it more seriously). My primary ethical hesitiation is that providing a first-order approximation of unfairness, for exmaple, might understate it in some cases -  and then people say "oh I can stomach that amount" and so it justifies inaction. Or, if it turns out that adjustment is extremetly senstive in practice (even if possible theoretically), then people may feel enboldened to optimize these influence-structures anyways since "they can just put a baindaid on afterwards", when really it means they should avoid that influence structure entirely. Currently I think there are ways to address these ethical concerns, but I want to be sure to spend some time analyzing them and including them in whatever publications result.

1/28 continued.
Updated repository name.
Ran this idea by Tiffany Cai. A few valuable notes about her response:
- initially seemed unmotivating, until I highlighted that it was algorithm-agnostic (otherwise, she believed fairness, etc. would be already well-trodden research)
- As expected, she was skeptical of my motivation-idea, of "quantifying makes the warning more concrete and motivating" (I may have induced this skepticism myself though, since I intentionlly drew an anlogy to climate change. I guess I need to admit to myself that I'm skeptical of this motivation...)
- But what does an "optimal algorithm" even mean in the SCIM context??

### Week of 1/30

1/30
I have a few thoughts based on the introductory reading of linear causal models (Pearl's Primer):
- I think linear analysis can be extended to nonlinear models (probably with bounding-like results) via Lipschitz analysis (similar to what I did during my Master's thesis).
- If I understand right, the basic model which is easiset to get results about, has linear structure, and Gaussian error terms? (Is this Gaussian in both regression errors and exogenous variables?)
- ^Admittedly I'm a little surprised, because I thought that it was important to have either nonlinearity or non-gaussian for doing structure learning...so why the difference here?
- All sorts of things are 'estimable' (<- nontechnical use warning) on linear models. We can infer direct effects, total effects, counterfactuals, sensitivity (< is this sensitivity of the graph structure or the SCM functions?). Probably more, still looking.
- So, it seems reasonable that I should be able to quantify the various incetive notions, in a linear model. A great place to start!

I think my next steps are to do a more thorough lit. review: double check this is an open area for CIDs, make the research roadmp/results more concrete by doing linear-SCM reading, and gain more confidence that there exist datasets/previous applications that I can compare against.

2/3
Narrowed down to top proposals: see etc/Feb_8_top_proposals.pdf

Read the paper
- Agent Incentives: A Causal Perspective
	- https://arxiv.org/pdf/2102.01685.pdf

2/4
Read the paper
- Counterfactual Fairness (2018)
	- https://arxiv.org/pdf/1703.06856.pdf

### Week of 2/6

2/7
Identified the following:

My relative strengths:
- Discrete Dynamical systems
- Linear Algebra
- optimization
- algorithms
- Theory, generally speaking
- Analysis (as in, the mathematical field of analysis)

What research 'buckets' am I interested in?
- Ideally, Fairness-Discrimination
- Backup: Equivalence classes

2/8
Changed my proposal to "Bounding Counterfacutal Fairness over PDAGs"

2/8
Polished my top contending proposals again: see etc/Feb_8_top_proposals.pdf

First pass on the papers
- R-30: The Causal Explanation Formula
	- https://causalai.net/r30.pdf
- Path-Specific Counterfactual Fairness
	- https://arxiv.org/pdf/1802.08139.pdf

Decided that the research bucket I want to stick to from now on, is "Incentivised Unfairness". My proposal may still change within that, but I'll stick to that bucket.

2/9
Updated to latest proposal: "Incentived Counterfactual Discrimination in Non-Markovian settings"


2/10
Updated my repo to reflect recent progress. Adjusted my default schedule so that at the end of every day I have a reminder to update my research journal, because I've been forgetting to do that. I also need to stop using my person Google docs to record my progress, and do it in the repo so commits are actually representive of my progress.

Oh, I just had a tantalizing idea! The usual identification problem goes "Given P(V) and G, output P(V|do(X)) as a function of P(V)". What if we flipped that on it's head? "Given a function f mapping P(V) to P(V|do(X)), how does this constrain the graph G"?? This would be very helpful for interpretability research.
- It seems implausible that knowing the function f and all it's partial derivatives, etc. would yield *zero* information about G...But I would also be excited to find a proof of that being the case!
- I know that f will not always fully identify G (for example, if f is the adjustment formula, G could be any graph with those variables that satisfies the backdoor criteria).
- ^ But I think in this case it would certainly restrict G to satisfy the adjusted-backdoor criteria!

2/11
^Consider again the backdoor example. Is it even practical to identify that f is the adjustment formula, in practice? If not, the negative case could still hold.

2/12
Found better document templates, organized working space.

### Week of 2/13

2/13
Based on the feedback I've recieved from Elias, I think I should drop the 'bounding over PDAGs' proposal: he seems to think there's a lot of hidden complexity, and that it's too much for a semester. Since I believe he understands my idea there fairly well, I trust his judgement.

Furthermore, I just spent a few hours trying to make the 'structural learning from an identification formula' direction work, and it seems to have too many unresolved questions. I lean towards dropping it as a Bareinboim proposal, but perhaps it could still work as a Blei proposal?
At any rate, I don't think I'll reach out to the AIS people quite yet, since that would take up valuable time for preparing for the CI2 proposal deadline.

Which means that of my 3 proposals, the last one standing is "Incentivized unfairness". Although it's possible this one could split into 2 seperate proposals as well.

So, I will put all my effort into filling proposal(s) for "Incentivized unfairness".

2/14
So, creating a synthetic dataset based on domain-experts' causal model is fine for the project; as is coming up with my own causal assumptions given a real world dataset.

Started trying to piece together a self-contained description of the incentived work.


2/15
Put together a rough draft of what I envision for non-markovian incentivised fairness - and along the way came up with what I believe could be a mechanistic explanation of discrimination, to inform policy design. Basically, we use incenvized fairness as a way to evidence reward-dependancies for rational agents.
First draft for translating "observational incentives" into plain SCM language.

2/16
Drafted the first run through of my ideas on extending to non-markovian systems, identifying incentivized discrimination, incentives-based decomposition of discriminatory effects, and explaining descriminination via rewards.
Feeling kind of skeptical on the usefulness of extending to non-markovian systems, except as a means to do the later stuff.

2/17
Finished putting together a research plan, including 3 input-output problem statements: see doc/Proposal.

### Week of 2/20


2/22
Began working on the first step of proving the soundness direction of response incentives for the more general case of possible confounding (Non-markovian). Mostly, getting to know the original proof.

2/25
Identified more parts of the soundness proof that require Markovian assumptions; typed up more of the proof.

### Week of 2/27

2/27
Proved that ancestry is a necessary condition for a response incentive, and typed it up.
Was pretty disappointed to realize that "response incentives" involve 'existence' in either direction, so it's never concrete. This can be seen with the two sentences I boiled it down to:
"If G does not admit a response incentive on X, then it's always possible to be fair."
"If G admits a response incentive on X, then it's possible that it's impossible to be fair."

...JK! I just realized from poring over the proofs that (at least in the Markovian case) the soundness proof is constructive...with the result that:
"If G does not admit a response incentive on X, then every optimal policy can be made blind to X (by freezing the non-requisite parents) without sacrificing performance!"

So "does not admit a response incentive" is actually a really strong criteria! Especially if I can show it holds in the non-markovian case.

3/1
Today, I finished convincing myself that the original proof for soundness holds in the non-markovian case. 
I also explored the possibility that the absence of a response incentive implies that every optimal policy will ignore that sensitive attribute...but I'm skeptical of that direction. If I come back to it, I'll look first for a counterexample - it seems plausible that even though we know optimality isn't sacrificed when you freeze the attribute, that perhaps it still shifts probability mass around (in such a way that it cancels out when expecation is taken, in every case?).

There's 3 directions I would like to make progress on this week: 1. finding an example where Cft-effects aren't ID, but the incentivized versions are; 2. finding a concrete dataset which looks likely to demonstrate these phenomenon I'm talking about, and 3. start putting together a more thorough literature review.

3/2
I double checked that non endogenous parents doesn't break the validity of the non markov proofs.
Thought about possible counterexamples to the cool conjecture I'd thought of (that all optimal poicies are fair if not incentivized otherwise), and conjectured that spurious effects are always unincentivized.
Thought about examples in which Cft-DE and Cft-IE effects are ID in Gmin, but not in G. 

Read a pre-released version of "Why Fair Labels can yield Unfair Predictions", by Carolyn Ashurst, Ryan Carey, Silvia Chiappa, Tom Everitt. Had a lot of thoughts! The main takeaway is that it's sometimes better to feed a sensitive attribute to an algorithm! (If the labels were fair).

3/3
As far as datasets go, I currently expect to use the PyCid package to generate synthetic data https://github.com/causalincentives/pycid and use either the Adult or Berkely datasets (income, and collegiate admissions, respectively).

Read: A Complete Criterion for Value of Information in Soluble Influence Diagrams.
The homomorphisms seem useful for transforming CIDs.
I think I might be able to use them for proving which families of diagrams are incentive-ID, but not generally ID for Cft effects.
It cites r63 and r66! So those could be good directions to go.


### Week of 3/6

3/8
Started outlining a mid-semester report.

<!-- Project Switch, to the Cyclic Causality: D-separation -->

3/9
Elias said my proposal was unclear, and recommended switching proposals to "cyclic causal models and consider issues of equilibrium. For instance, how basic results (d-separation) break down and how to go around it. One reference you could check is White & Chalak - https://www.jmlr.org/papers/volume10/white09a/white09a.pdf , but there are others."
So I'm switching my proposal to that.
Reading "Settable Systems: An Extension of Pearl's Causal Model with Optimization, Equilibrium, and Learning"

3/10
Identified for myself why cycles, etc. are problematic in SCMs; if there's not a unique fixed point, then the potential response function isn't well-defined! Thought of a way to ensure a unique fixed point in a fairly general setting (over Lipschitz-continuous functions over a product space of metric spaces) by applying tools from my Master's thesis. 

I came up with several illustrative examples, ranging from full-deterministic to having a stochastic "prior" on V_0. 

Notably, I got a full proof sketch that my proposed class (of intrinsically-stable, aka Lipschitz-stable SCMs) will always have unique fixed points for all do(X=x) interventions. (I'm curious if I could prove the same thing for soft-interventions, and for counterfactuals generally?)

Met with Alexis (a postdoc in Elias' lab) and settled on reading 
 Stephan Bongers. Patrick Forré. Jonas Peters. Joris M. Mooij. "Foundations of structural causal models with cycles and latent variables." Ann. Statist. 49 (5) 2885 - 2915, October 2021. https://doi.org/10.1214/21-AOS2064 

Since it's so recent, and by one of the top names in cyclic causality (Mooij) it should be pretty comprehensive (as in, if I identify an open problem relative to this paper, it's probably still unsolved and relevant to the field). 

### Week of 3/13

3/16
Read "Foundations of structural causal models with cycles and latent variables."
I understand a lot more about what issues cycles bring up. In particular, each of the following conditions are lost in general:
- Closed under marginalization
- And the marginalization respects the latent projection
- Closed under perfect intervention
- Closed under the twin operation
- The observational distribution, interventional distribution, and counterfactual distributions all exist and are unique
- These distributions satisfy the general directed global Markov property (sigma-separation) relative to (each of the G’s)...
- ...which means that the solutions always satisfy the conditional independencies implied by σ -separation
- These distributions also satisfy d-separation if M satisfies at least one of the three conditions
- We can define the causal relationships for simple SCMs in terms of its graph
(the graph can be interpreted as having causal semantics)
Potential outcomes can be defined similarly to acyclic SCMs.

However, the class of "simple SCMs" have all these properties! A simple SCM is one which is "uniquely solvable" w.r.t. every subest of endogenous variables.

I keep having an itch that my work from my Master's research on "intrinsically-stable systems" should be able to extend some of these results: basically, intrinsically-stable systems are a class of nonlinear dynamical systems which still have the nice asymptotic properties of linear systems. (Kind of like how convex functions have all the nice properties of linear functions, relative to optimization.)

Ran some quick numerics to verify that the linear cyclic models presented in "LEARNING LINEAR CYCLIC CAUSAL MODELS WITH LATENT VARIABLES" by HYTTINEN, EBERHARDT AND HOYER https://www.jmlr.org/papers/volume13/hyttinen12a/hyttinen12a.pdf are not intrinsically stable: I would have been very surprised if they had.

Came up with quite a puzzle: consider the linear example of one node, with a self-dependece of 0.5 times its previous value. It should be uniquely solvable, I believe, but the presence of a self-loop means that it can't be. Quite puzzling.

3/17
Spent more time diving into whether that linear example counts as having self-loops (and hence is not uniquely solvable) but ultimately decided to put it off for now. I expect it will be uniquely solvable, and that there's just some measurable mapping I'm not seeing yet. All the answer will really change is whether intrinsically-stable SCMs are a subset of simple SCMs.

I came up with the following conjectures today:
- Conjecture 1: I can replace “linear” in Theorem 6.3 part 1(c) with “intrinsically stable”
Which means d-separation is guaranteed to hold in an broad non-linear, cyclic setting
- Conjecture 2: Every intrinsically-stable SCM is simple.
- Conjecture 2a: No intrinsically-stable SCM has self-cycles (by the definition provided in this paper).
- Conjecture 3: For all simple SCMs, there is an intrinsically-stable SCM equivalent to it.
So even though the set of “intrinsically stable SCMs” is contained within the set of “simple SCMs”, it’s broad enough to capture everything we care about.
(I expect this won’t hold in general, but I do expect it to hold for all continuous, smooth, or otherwise “nice” SCMs; aka all the ones practitioners would actually care about)
- Conjecture 4: Equivalence is preserved under isospectral transformation.
This means we can use any isospectral transformation to change the graph into one that is easier to analyze.

I think the usefulness of Conjectures 2 and 3 are dependent on the validity of Conjecture 1. Conjecture 4 seems less important than Conjecture 1, but may still have independent usefulness.

3/18
Parsed through some linear examples. Strategy thinking for next week:

TODOs for next week (decreasing in priority)

Main Objective: Get a good enough lit. review on cyclic causality that I stop being surprised by what I read

- Finish distilling "main highlights" from the 3 papers I already mentioned
    - Foundations of structural causal models with cycles and latent variables (Bongers, Forré, Peters, Mooij: October 2021)
    - Learning Linear Cyclic Causal Models with Latent Variables (Hyttinen, Eberhardt, Hoyer: 2012)
    - Settable Systems: An Extension of Pearl's Causal Model with Optimization, Equilibrium, and Learning (White and Chalak: 2009)

- Literature review specifically on any nonlinear, cyclic d-separation results
    - Since I think this is the topic I'll be narrowing on
    Read: Markov Properties for Graphical Models with Cycles and Latent Variables (Forré, Mooij: 2017)

- (Time permitting) Read: Causal Modeling of Dynamical Systems (Bongers, Blom, Mooij: original in 2018, updated Dec. 2021)

- (Time permitting) Switch to "breadth-first" search of literature, only reading abstracts/conclusions
    - with the goal of making sure I've identified all "landmark" papers on cyclic causality

### Week of 3/20

3/20
Gathered and ranked references on cyclic causality.
Wrote up a literature review of Foundations of structural causal models with cycles and latent variables (Bongers, Forré, Peters, Mooij: October 2021).

Tried to come up with examples of cyclic, nonlinear SCMs that I could test numerically whether d-separation is valid. This was surprisingly difficult: for one thing, testing the conditional independancies of a continuous-distribution is apparently hard to do. (I ended up using the fcit pacakge, although I don't have much background on if that's appropriate or not). I ran into multiple debugging issues.

Partly because I was stuck debugging, I came up with a different example which is intrinsically stable (but unknown if it's a counterexample to d-separation or not). When I ran it, both the intrinsic version, and the not-intrinsically stable version satisfied d-separation. I'm unsure whether to count that as weakly supportive of the intrinsic stability idea, but I lean against it.

3/21
Resolved the degugging issues of the counterexample from the literature! And the result is moderately convincing for me that this idea that intrinsicaly stable systems satisfy d-separation is legit: the counterexample works as expected when using a multivariate gaussian over an unbounded domain, but when only samples within a restricted domain are used (corresponding to the region over which the system is intrinsically stable) it starts respecting d-separation again!

Penciled out a thorough outline of a mid-semester report. Since I switched projects mid-semester, it will be bundling all of 1. problem statement, 2. literature review, and 3. preliminary results (for me, that's some sanity numerics and a proof sketch of my conjecture). It'll be a busy week, but I expect that by the end of the week I will be roughly back on track with where I should be for the semester pace.

3/22
Drafted multiple flow charts, plots from numerics, etc. demonstrating my research plan and progress. My current research direction is to demonstrate observational equivalence of an intrinsically stable (nonlinear, cylcic, continous-domained) SCM with a linear SCM which satisfies d-separation.

Met with Christian Kroer, as a possible game theory + causality collaborator. (Which is very relevant to cyclic causality, although not the specific direction I'm thinking of working on now).

3/23
Prepared a (nearly) full set of slides describing my research direction: The only thing missing is detailing how this research direction fits in with the previous literature.

In particular, I articulated what the problem statement is. I'll include the slides (and the mid-semester report I'm preparing) after the week is over (and update the abstract accordingly.)

3/24
Transferred over all the content from my slides to my report.
Investigated what the implications of my research would imply (in particular, to ensure it wouldn't violate the No Free Lunch principle).


### Week of 3/27

3/29
Worked out a clean simple example which highlights each of the following:
1. cyclic causality
2. existence of the observational distribution
3. cyclic d-separation
4. (failure of) directed global Markov property
5. numerics to verify conditional dependence
6. how to construct an intrinsically stable SCM instead
7. numerics demonstrating the intrisically stable SCM does satisfy d-seperation!

3/30
Created a presentation with these slides; uploaded to doc/weekly_updates.

Wrote out proof sketch for observational equivalence - the nonlinear's P(V) needs to be 'pushed back' through the inverse of the linear potential response, to get the linear's P(U).

### Week of 4/3

4/4
Proved my set-inclusion conjectures: that intrinsic SCMs subsume acyclic SCMs, and that they are contained in simple SCMs. 
Proved intervening on an intrinsic SCM yields another intrinsic SCM, and validated this with numerics (just in case I'd made a mistake).

4/5
Proved observational equivalence for intrinsic SCMs: that the linear SCM associated with the (nonlinear) intrinsic SCM generates the same observational distribution, and respects the same graph.

4/6
Read a nonpublic, submitted paper from Lewis Hammond (Oxford), et. al. titled "Reasoning about Causality in Games". Explored some potential expansion using intrinsic SCMs to extend it to a continious domain. I'd really like to move into game theory! That's why I find cyclic causality motivating.

4/7
Changed the name from "Intrinsic SCMs" to "Lipschitz SCMs", to better capture the meaning.
Made up an "bartering game" which converges to a supply-demand equilibrium and is a Lipschitz SCM
Discussed a possible game theory application with Christian Kroer. 

The "application proposal" we settled on that I'll keep working out the details for, basically amounts to extending the causal games paper for continous-domain, nonlinear games. I anticipate this will be done by showing that 1. for systems of [after fixing U=u, if there are multiple equilibria, each of these is in fact locally Lipschitz], we have that 2. d-separation holds, so that (regardless of which equilibrium we're at) we can evaluate causal queries from the graph. In this way it doesn't actually matter that there are multiple equilibria, because they all behave the same!

### Week of 4/10

4/10
Started thinking through how to do numerics for these results. I drafted up some psuedo-code but decided it'd be more clear how to do the numerics, after I finish working out the details of the interventional/counterfactual proofs, since a lot of my confusions overlap between them.

Proved the remaining portion of the observational equivalence case. I'm overall 90 percent confident in the validity of the proof: I wouldn't be surprised if there were minor mistakes, but very surprised if the overall result overturned. 

I fiddled with the definition of the Lipschitz matrix A a bit, to allow for infinities. I'm confident the math would go though, but I shy away from it for now, simply out of convienence: I'd need to reprove a lot of the original intrinsic stability results hold, but allowing for infinities in A. I'm confident they will, but for now it sufficies to just put in placeholder 1's where the derivative isn't bounded.

I proved that the interventional equivalence case boils down to just proving that the operations of (linearize via the Lipschitz matrix) and (do-operation) commute. I expect to prove that soon.

Now that I'm anticipating the counterfactual equivalence, I'm realizing I have some uncertainties/confusions about the twin network in "Foundations", Bongers et. al. I'll look into that soon.

A few other notes: I'm eager to do locally-lipschitz. I feel very confident all the results will still go through, which would be very exciting! (I'm also confident the graphs would remain the same across the equilibria, so long as there's no degeneracy, perhaps no vanishing of the structural functions. This would mean that you can use the same graph for multiple equilibria!! Crucial for game theory applications).

4/11
Finished proving interventional equivalence (the last piece was that order of intervening/linearizing didn't matter).

4/12
Proved a significant portion of counterfactual equivalence: intrinsic (Lipschitz) SCMs are still lipschitz afer the mirror transformation, and order doesn't matter.

4/13
met with Elias, Alexis. General feedback is to focus on building intuition, get clear examples.

4/17
Really liked this quote from "Learning linear cyclic causal models with latent variables", Hyttinen et al. 2012.
"Intuitively, the true causal structure is acyclic over time since a cause always precedes its effect: Demand of the previous time step affects supply of the ext time step. However, while the causally relevant time steps occur at the order of days or weeks, the measures of demand and supply are typically cumulative averages over much longer intervals, obscuring the faster interactions. A similar situation occurs in many biological systems, where the interactions occur on a much faster time-scale than the measurements. In these cases a cyclic model provides the natural representation, and one needs to make use of causal discovery procedures that do not rely on acyclicity (Richardson, 1996; Schmidt and Murphy, 2009; Itani et al., 2008)."

Example for causal discovery: In "Constraint-based Causal Discovery for Non-Linear Structural Causal Models with Cycles and Latent Confounders" Mooji and Forre generate SCMs randomly (as NNs with tanh activations) and record how many edges they were able to identify (ROC score so as to show progress w.r.t. both fewer false postivies and false negatives).

Wait up...I think I just disproved a foundational pillar of my ALL my results. Surveying now to see if any of the results are recoverable with a different formalism. I'll run some numerics soon if that helps claify if the phenomenon still holds (of d-separation working in the nonlinear, cyclic continous-domain case).

4/18
Summary of the current issue^: I thought the graph of the linearization matched the nonlinear, but unfortunately there's a tradeoff between getting the graphs to line up, and having the observational distributions match. (I can make the observational distributions match if I add unobserved confounding to all the variables - hardly the same graph)!

But the confounding's not really unobserved, I know it deterministically - so maybe I can leverage this knowledge to keep one implication direction?

4/19
Identified that the causal hierarchy is not yet proved for cyclic models. Did some reading on that. 

Identified that some of my attemps to explore the "asymptotic" graph really is just using sigma-separation, a weaker version of d-separation. The main thing that's stumping me right now, is that I have shown that the independence I want still holds, but for a larger graph which encompasses both the nonlinear and the linear model together (and how the linear model is derived from the nonlinear). But since part of this graph corresponds to nonlinear functions, I don't think d-separation is guarenteed to hold (at least not based on previous results). Maybe if I dig into the details of why d-separation holds in the linear case, I'll understand if I can claim it to hold in this case??

4/20
Generated all cyclic strongly connected graphs.

4/21
Provided unit tests for generating strongly connected graphs. Wrote code for evaluating arbitrary potential response functions, sampling neural networks compatible with a possibly-cyclic causal graph, sampling an observational distribution from a cyclic SCM, and generating all possible independance relations on n variables. Also, constructed unit tests for all of these.
David Blei pointed out that should these numerics prove successful, it would also be interesting to see numerics on whether intrinsic SCMs have better estimation properties as well.

I'm running up against a barrier: there's no implementations of cyclic d-separation online, it seems. I think I can reduce cyclic d-separation to a (combinatorial?) number of acylic d-separation evaluations, though I need to prove they're equivalent first. This would significantly reduce the likelihood of introducing my own errors into the code.

## Week of 4/24

4/25
On a loose investigation, it appears that the d-separation code for acyclic graphs from networkx SHOULD work without modificaiton on my cyclic graphs. Not 100% confident yet, but it's worth moving forward with the preliminary experiments.
Created functions for generating and validating all d-separations, as well as unit tests for these.

4/26
I was surprised to see the results of the numerics this morning: all the Lischitz neural networks seem to satisfy d-separation (not just the intrinsically-stable ones)! I would celebrate this finding more, but I wonder if there might be a bug in my code, since even product- examples seem to satisfy d-separation....(which I know shouldn't be true, for at least one graph, the counterexample). 
So I need to polish up my unit tests before I can draw confident conclusions.

4/27
Crafted examples highlighting the intuition of why dynamics are important: wrote up in slides. 

4/28
Considered defining L(M) as entirely a SCM acting on a distribution of distances - like, P_N(U) is the probability of the pairwise distance drawn from P_M(U). 
Worked out some kinks in my numerics that were preventing convergence of the potential response. Rerunning numerics now.

4/29-4/30
To avoid sampling bias (not getting data on systems that don't converge) I opted for finding the fixed points using root-finding methods, rather than iterating the system directly. In this way I was able to eliminate the non-convergence for both the intrinsic- and lipschitz-SCMs that were sampled. However, the root finding was too slow for the polynomial SCMs, so I dropped those for now.

## Week of 5/1

5/1
Numerics (excluding the polynomial SCMs) finished running this morning. With the updated root-finding approach for finding equilibria, all the intrinsic- and lipschitz-SCMs converged! Surprisingly, all of them satisfied the markov criteria of "d-separation validity": every independence of the graph was found in the observational data as well. This is significant, because I thought only the intrinsic SCMs would satisfy it, but not lipschitz SCMs in general (here, intrinsic SCMs are those lipschitz SCMs whose Lipschitz-linearization also converges, whereas "lipschitz SCMs" are just the ones which are lipschitz continuous).

The fact that the polynomial SCMs don't converge well to their fixed points weakly supports my conjecture that the salient issue at play regards the boundedness of the potential response function: I conjecture that d-separation fails when the potential response has a singularity for some inputs of the exogenous variables U. The reasoning is something like:
- Assume each U_i is independent
- if there exists some subset of U (still in the support of P(U)) for which V is unbounded, then any dependencies between the U's induced by restricting ourselves to that subset, will show up in the V's as well.
- (so even though each U_i is independent in general, they may not be independent over the subset over which V is unbounded. In the classic counterexample, the singularity occurs for V3 and V4 when U1 U2 == 1, clearly inducing a dependence between U1 and U2, which then is inhereited by V1 and V2)
- I believe if the structural equations are lipschitz continuous, then the potential response must be as well => no singularities.

It just occured to me that I missed some Lipschitz-continous SCMs: the ones with p(A)>1.

5/2
Extended the numerics to sample explicitly from Lipschitz-continous SCMs with p(A)>1. Surprisingly, they still seem to be respecting d-separation!

5/3
Worked on introduction, problem formulation, etc. of presentation. Identified which of my results are actualy ready to report as main results, vs. still preliminary.

I looked into the proof of why linear d-separation works for cyclic models. I'm intrigued by a recent idea of how to get the implication I need.

5/4
I think maybe stable-Lipschitz SCMs (my latest name for intrinsic SCMs) might not be closed under marginalization?? I only allowed myself an hour to look into it, because of the deadlines, but this would be very surprising since they are closed under interventions and counterfactuals.
But, if this ends up being the case, I think it would be straightforward to limit the space further with additional constraints, such that the resulting space is closed under marginalizations. (But then would it be closed under interventions still?) The mystery thickens.

Rewrote, refined the primary results I'll be reporting on: closed under interventions and the set-inclusion, but not observational d-separation :(
Latexed the primary results.


## Week of 5/8
5/9
Pepe said that one of times economists identify the parameters for a system of supply+demand is when there's a third variable correlated (for example) with demand, but not supply. He recommends I look into that....I'm thinking it should suffice for now to consider a causation between demand and that third variable for the purposes of my example. Although that's assuming even more...

5/10:
Finally proved that d-separation holds for stable-Lipschitz SCMs! Although I needed to assume additive, independent noise terms to do it. I expect that those will be able to be weakened though.

Basically, I followed the proof for why linear d-separation works for cyclic SCMs from Mooji et. al, "Markov Properties for Latent Cyclic Models".

5/11:
Worked through the supply + demand example more. Specifically, Pepe's suggestion to add a new correlated variable. I ran with a narrative of "the market is for a basket of goods (so the price is in fact just a measure of inflation). The Feds set the federal interest rate based on inflation. A lower interest rate encourages consumer spending but does not direct affect supply (economic output)."

5/12:
Finalized Elias' report, recorded presentation.

Met with Benjamin Webb from Brigham Young University, made sure that the way I've been using the "intrinsic stability results" is valid. And importantly, locally stable-Lipschitz is on!! Why this is important: it means that d-separation should work for multiple equilibria. Like, we don't need to know which equilibria we're at (aka how they do it in settable systems) to be able to do causal inference!

Also, Ben mentioned a really interesting point: I think isospectral reductions of a causal graph may improve estimation properties. Although I'm not entirely sure it transfers properly, it's an interesting applied direction to go.

5/13
Met with Pepe again. We discussed the use of structural equations models extensively, as they are used in economics. The final, synthesized lessons are:
- economists sample a new shock (aka. noise; exogenous variable) at each timestep.
- They either assume each timestep is a DAG, or use autogregressive relationships between each timestep and the next. 
- In contrast, Bongers et. al sample a noise term at the start, and then use that fixed term throughout the evolution of all the dynamics! 

So, while cyclic SCMs remove one huge modeling assumption for practical economics (the structural equations are recursive), they still hold onto another huge assumption: noise is infrequent. 

So, what are the modeling assumptions of cyclic SCMs? 
- 1. the sampling frequency is much lower than the feedback frequency.
- 2. the noise frequency is much lower than the feedback frequency.

Pepe agrees with me that assumption 1 seems valid for GDP measurements (the accounting process really slows things down), and that assumption 1 is not valid for high-frequency trading (where we have millisecond-level data, if not faster).

I'm going to review "10 questions from Pearl for Economists" for Pepe and tell him my thoughts next Friday, same time.