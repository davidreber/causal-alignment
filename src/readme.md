Numerics for testing the validity of d-separation in cyclic SCMs.

The bulk of the code is contained in helper_functions.py. Several highlights:
- sampling cyclic graphs (without isomorphic duplication)
- sampling neural networks compliant with a causal graph
- given a cyclic SCM, sampling the observational distribution (via root-finding for better convergence)
- enumerating and testing possible d-separation statements

unit_tests.py ensures the validity of the code in helper_functions.py.

The actual numerics are run via:
- main.ipynb (stable-Lipschitz and stable SCMs). Takes about 16 hours to run on my laptop.
- high_lipschitz.ipynb (non-stable SCMs). Takes about 8 hours to run on my laptop.

Then, the results of the numerics are aggregated in plotting.ipynb.

Lastly, supply_demand.ipynb represents my ongoing work to construct a Pepe-approved macroeconomic model which can showcase these new methods, but still represent a relevant real-world problem. For now, it's based on classic supply/demand examples.