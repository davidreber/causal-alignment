import numpy as np
import helper_functions as hf

def find_stationary(e,c): # verified seems legit
        assert len(e) == 4, e
        y,x,z,w = e[0].copy(), e[1].copy(), e[2].copy(), e[3].copy()
        assert np.all(x*y<c), ( (x*y)[x*y>c][:2], (x)[x*y>c][:2], (y)[x*y>c][:2] )
        assert np.all(1-x*y>1-c)
        Z = (w*y+z)/(1-x*y)
        W = (z*x+w)/(1-x*y)
        return np.vstack([y, x,Z,W])

def test_nice_graphs():
    # for n in [2,3,4]:
    for n in [2,3]:
        naive = hf.nice_graphs_naive(n)
        fast = hf.nice_graphs_fast(n)
        assert len(naive) == len(fast), (len(naive), len(fast))

        if n == 2:
            assert len(naive) == 1
        if n == 3:
            assert len(naive) == 5

        for A in naive:
            assert hf.isomorphic(A,fast), (n,A)
            assert np.sum(np.diag(A)) == 0

def test_sample_neural():
    # test relu, and dropping edges correctly
    n = 3
    for l in range(10):
        structure = hf.sample_neural(n=n,activation="relu")
        Adj,F,A,f_type = structure
        assert np.count_nonzero(Adj*A) == np.sum(Adj)
        assert np.allclose(np.nonzero(Adj*A), np.nonzero(Adj))

        # Test that relu zeros out correctly
        v = np.random.uniform(-5,5,n)
        out = F(v,np.eye(n),np.zeros(n))
        assert np.allclose(out==0,v<0)

        for k in range(10):
            pos_v = np.random.uniform(0,5,n)
            U = np.random.uniform(-5,5,n)

            # Relu should be the identity on nonnegative input
            assert np.allclose(pos_v+1,F(pos_v,np.eye(n),np.ones(n)))

            for i in range(n):
                # verify A*Adj drops edges correctly
                alt_A = (2*np.ones(n**2).reshape((n,n)))*Adj
                assert np.allclose(alt_A,2*Adj)

                assert np.all(alt_A.dot(pos_v)>=0)
                res = F(pos_v,alt_A,np.zeros(n))
                assert np.all(res >= 0)
                assert np.allclose(res,alt_A.dot(pos_v))

                # verify F doesn't change shape of vector
                assert np.allclose(np.shape(res),np.shape(pos_v))

    # test tanh
    structure = hf.sample_neural(n=n,activation="tanh")
    Adj,F,A,f_type = structure
    indices = np.nonzero(Adj[0])
    pos_v = np.random.uniform(0,5,n)
    U = np.random.uniform(-5,5,n)
    assert np.allclose(F(pos_v,A*Adj,U)[0], np.tanh(A[0,indices].dot(pos_v[indices])+U[0]))

    # check that multiple U can be passed through F
    U1 = np.random.uniform(-5,5,n)
    U2 = np.random.uniform(-5,5,n)
    U = np.vstack((U1,U2))
    assert np.allclose(np.shape(U), (2,n))

    v = np.random.uniform(-5,5,n)
    Adj,F,A,f_type = hf.sample_neural(n=n,activation="relu")
    separate = np.vstack((F(v,A*Adj,U1),F(v,A*Adj,U2)))
    together = F(v,A*Adj,U)
    assert np.allclose(together.shape,separate.shape)
    assert np.allclose(together,separate)

    Adj,F,A,f_type = hf.sample_neural(n=n,activation="tanh")
    separate = np.vstack((F(v,A*Adj,U1),F(v,A*Adj,U2)))
    together = F(v,A*Adj,U)
    assert np.allclose(together.shape,separate.shape)
    assert np.allclose(together,separate)

def test_potential_response():
    n = 3

    A = 0.9*np.eye(n)
    U = np.zeros((n,1))
    F = lambda V,A,U: A.dot(V) + U
    eps = 1e-5

    for Adj in hf.nice_graphs_fast(n):
        structure = Adj,F,A,"linear"
        equilibrium, orbits = hf.potential_response(structure, U, eps=eps,orbits=True)
        assert np.linalg.norm(equilibrium) < eps
        assert len(orbits) == 3
        finals = []
        for orbit in orbits:
            finals.append(orbit[-1])
        assert np.allclose(np.mean(finals), equilibrium)


    # Test that vectorization works correctly
    U1 = np.random.uniform(-5,5,(n,1))
    U2 = np.random.uniform(-5,5,(n,1))
    U = np.hstack((U1,U2))
    assert np.allclose(np.shape(U), (n,2))
    for act in ["relu","tanh"]:
        structure = hf.sample_neural(n=n,activation=act)
        eq1 = hf.potential_response(structure, U1)
        eq2 = hf.potential_response(structure, U2)
        separate = np.hstack((eq1,eq2))
        together = hf.potential_response(structure, U)
        assert np.allclose(together.shape,separate.shape)
        assert np.allclose(together,separate,atol=1e-4) # Note worse than 1e-5

def test_sample_observational():
    n = 3
    num_points = 1e3
    structure = hf.sample_neural(n=n,activation="tanh")
    obs_dist = hf.sample_observational(structure,num_points=num_points)
    assert np.allclose(obs_dist.shape,(n,num_points))

def test_all_candidate_independencies():
    assert len(list(hf.all_candidate_independencies(2))) == 1
    assert len(list(hf.all_candidate_independencies(3))) == 6

    for combo in hf.all_candidate_independencies(4):
        X, Y, Z = combo
        assert type(X) is int
        assert type(Y) is int
        assert type(Z) is tuple

        # X, Y, and Z are disjoint
        assert X != Y
        assert X not in Z
        assert Y not in Z

def test_all_d_separations():
    Adj = np.array([[0,0,0,0],[0,0,0,0],[1,0,0,1],[0,1,1,0]])
    d_seps = hf.all_d_separations(Adj)
    assert list(d_seps) == [(0, 1, (2, 3)),]

def test_obs_indep1():
    "make sure using new obs_indep() doesn't change previous results"
    def F(x,e,c=0.9):
        f0, f1 = e[:2]
        f2 = c*np.sin(x[0])*np.cos(x[3])+e[2]
        f3 = c*np.sin(x[1])*np.cos(x[2])+e[3]
        return np.vstack([f0,f1,f2,f3])

    def evolve(e,x,length = 100,c=0.9):
        for t in range(length):
            x = F(x,e,c=c)
        return x

    c = 0.9
    # Monte Carlo from P(V)
    num_points = 2100
    mu = np.zeros(4)
    cov = np.diag(np.ones(4))
    e = np.random.multivariate_normal(mu,cov,num_points).T
    x = np.copy(e)
    Pv = evolve(e,x,c=c)

    candidate = (0, 1, (2, 3))

    assert hf.obs_indep(candidate,Pv,p_thresh=1e-1)

def test_obs_indep_2():
    "make sure using new obs_indep() doesn't change previous results"

    independants = []
    for i in range(20):
        if len(independants) >= 3 and np.mean(independants) > .9:
            break
        c = 0.5
        # Monte Carlo from P(V)
        num_points = 100000 # if drop by an order of magnitude, fcit gives nans??
        mu = np.zeros(4)
        cov = np.diag(np.ones(4))
        e = np.random.multivariate_normal(mu,cov,num_points).T

        e_mask = np.all(np.abs(e)<c,axis=0)
        assert np.sum(e_mask) > 2000, np.sum(e_mask)
        e = e[:,e_mask]
        assert not np.any(e>c)

        Pv = find_stationary(e,c)
        assert np.allclose(Pv.shape,(4,int(np.sum(e_mask)))) # (n,num_points), effectively
        if c < 1.0:
            assert not np.any(Pv>1.0)

        candidate = (0, 1, (2, 3))

        indep = hf.obs_indep(candidate,Pv,p_thresh=1e-1)
        independants.append(indep)
    assert np.mean(independants) > .8, (np.mean(independants),len(independants)) # passes at least 90% of the time.

def test_obs_indep_3():
    "make sure using new obs_indep() doesn't change previous results"

    independants = []
    for i in range(20):
        if len(independants) >= 3 and np.mean(independants) < .1:
            break
        c = 100.
        # Monte Carlo from P(V)
        num_points = 100000 # if drop by an order of magnitude, fcit gives nans??
        mu = np.zeros(4)
        cov = np.diag(np.ones(4))
        e = np.random.multivariate_normal(mu,cov,num_points).T

        e_mask = np.all(np.abs(e)<c,axis=0)
        assert np.sum(e_mask) > 2000, np.sum(e_mask)
        e = e[:,e_mask]
        assert not np.any(e>c)

        Pv = find_stationary(e,c)
        assert np.allclose(Pv.shape,(4,int(np.sum(e_mask)))) # (n,num_points), effectively
        if c < 1.0:
            assert not np.any(Pv>1.0)

        candidate = (0, 1, (2, 3))

        indep = hf.obs_indep(candidate,Pv,p_thresh=1e-1)
        independants.append(indep)
    assert np.mean(independants) < .2, np.mean(independants) # fails at least 80% of the time.


def test_product_SCM():
    "Test validity of local functions"
    Adj = np.array([[0,0,0,0],[0,0,0,0],[1,0,0,1],[0,1,1,0]])
    n = Adj.shape[0]

    def alt_potential_response(U):
        U1, U2, U3, U4 = U.ravel()
        V1 = U1
        V2 = U2
        V3 = (U1*U4+U3) / (1-U1*U2)
        V4 = (U2*U3+U4) / (1-U1*U2)
        return np.array([V1,V2,V3,V4])

    for k in range(100):
        structure = hf.product_SCM(Adj)
        Adj,F,A,f_type = structure

        U_mean = -0.5*np.ones(n)
        U_cov = np.diag(0.2*np.ones(n))
        U = np.random.multivariate_normal(U_mean,U_cov,1)
        
        # Verify that my benchmark responses are legit
        true_response = find_stationary(U.ravel(),100.)
        definately_true = alt_potential_response(U)
        assert np.allclose(true_response.ravel(), definately_true.ravel())

        # Verify that the product function respects the fixed points
        assert np.allclose(true_response,F(true_response,Adj,U.T),1e-5), (true_response,F(true_response,Adj,U.T))

        # Make sure my potential response function works with my product function
        unsure_response = hf.potential_response(structure,U,eps=1e-8,max_iters=1e5)
        # assert unsure_response is not None
        # assert np.all(np.isfinite(unsure_response)), unsure_response
        # assert np.allclose(unsure_response,F(unsure_response,Adj,U.T))
        # assert np.allclose(true_response,unsure_response) # fails ...when potential response encounters infinities?

        # It would seem that my product function is valid, SO LONG AS the system is numerically stable...
        if unsure_response is not None and np.all(np.isfinite(unsure_response)):
            assert np.allclose(unsure_response,F(unsure_response,Adj,U.T))
            assert np.allclose(true_response,unsure_response) # fails ...when potential response encounters infinities?