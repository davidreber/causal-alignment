import numpy as np
import networkx as nx
import time
from itertools import chain, combinations
from collections import deque
from networkx.utils import not_implemented_for, UnionFind
from fcit import fcit
from scipy import optimize

def radius(A): # consistently fastest
    return np.max(np.abs(np.linalg.eigvals(A)))

##################### Naive version for graph generation ########################

def all_binary_matrices(n, m=None): # from stack exchange
    if m is None:
        m = n
    for i in range(2**(n*m)): 
        yield np.array([int(k) for k in "{0:b}".format(i).zfill(n*m)]).reshape(n,m)

def isomorphic(A,collection,is_graph=False):
    if is_graph:
        GA = A
    else:
        assert type(A) == type(np.array(1)), type(A)
        GA = nx.DiGraph(A.T)

    for B in collection:
        if is_graph:
            GB = B
        else:
            GB = nx.DiGraph(B.T)

        if nx.algorithms.is_isomorphic(GA,GB):
            return True
    return False
        
def nice_graphs_naive(n): # able to get up to n=4
    """Returns all cyclic matrices of size n, 
    corresponding to graphs which have:
    - no unobserved confounding
    - strictly cyclic (no acylic graphs are returned)
    - strongly connected (encompasses previous condition)
    - no self loops
    - no isomorphic repeats
    """
    collection = []
    for A in all_binary_matrices(n):
        # throw out any that don't meet the conditions we care about
        if np.sum(A) < n: # clearly acyclic
            continue
        if np.any(np.diag(A)): # has self-loops
            continue
        # check if A is irreducible
        DG = nx.DiGraph(A.T)
        if nx.algorithms.components.is_strongly_connected(DG):
            if not isomorphic(A,collection):
                collection.append(A)
    return collection


##################### (Slightly) faster version for graph generation ########################

def binary_combinations(l):
    for i in range(2**l,2**(l+1)):
        yield np.array([b for b in bin(i)[3:]],dtype=np.int8).tolist()

def all_triangles(n):
    for combo in binary_combinations((n*(n-1))//2):
        if np.sum(combo) == 0:
            continue
        triangle = np.zeros((n,n),dtype=np.int8)
        for d in range(1,n): # diagonal index
            l = n-d # how many entries on that diagonal
            diag = [combo.pop() for i in range(l)]
            triangle += np.diag(diag,d)
        yield triangle

def nice_graphs_fast(n): # ~3x speedup
    """Returns all cyclic matrices of size n, 
    corresponding to graphs which have:
    - no unobserved confounding
    - strictly cyclic (no acylic graphs are returned)
    - strongly connected (encompasses previous condition)
    - no self loops
    - no isomorphic repeats
    """
    A_collection = []
    G_collection = []
    for upper in all_triangles(n):
        for lower in all_triangles(n):
            A = upper + lower.T # no self-loops by construction
            if np.sum(A) >= n: # not obviously acyclic
                # check if A is irreducible
                DG = nx.DiGraph(A.T)
                if nx.algorithms.components.is_strongly_connected(DG):
                    if not isomorphic(DG,G_collection,is_graph=True):
                        G_collection.append(DG)
                        A_collection.append(A)
    return A_collection

def all_graphs_fast(n):
    """Returns all cyclic matrices of size n, 
    corresponding to graphs which have:
    - no unobserved confounding
    - strictly cyclic (no purely acylic graphs are returned)...TODO: verify via test!
    - no self loops
    - no isomorphic repeats
    """
    A_collection = []
    G_collection = []
    for upper in all_triangles(n):
        for lower in all_triangles(n):
            A = upper + lower.T # no self-loops by construction
            DG = nx.DiGraph(A.T)
            if not isomorphic(DG,G_collection,is_graph=True):
                G_collection.append(DG)
                A_collection.append(A)
    return A_collection

##################### Sampling SCMs ########################

def sample_neural(Adj=None,n=None,activation="relu",intrinsic=None,timeout=1):
    """Generates a structure-tuple corresponding to a random NN.
    Either the adjacency matrix Adj or a dimension n must be specified.
    intrinsic (bool or None) specifies whether the NN must be intrinsic (or not)
    """
    if n is None:
        n = Adj.shape[0]
    else:
        assert Adj is None
        adj_matrices = nice_graphs_fast(n)
        i = np.random.choice(len(adj_matrices))
        Adj = adj_matrices[i]

    if activation == "tanh":
        F = lambda V,A,U: np.tanh(A.dot(V) + U)
    elif activation == "relu":
        relu = lambda x: x * (x > 0) # TODO: adjust for scope?
        F = lambda V,A,U: relu(A.dot(V) + U)

    valid = False
    start = time.time()
    A_mean = 0.0*np.zeros(n**2)
    if intrinsic is None:
        A_cov = np.diag(0.3*np.ones(n**2)) # more variance than others
    else:
        # yields intrinsic 42% of the time, stable 81% of the time, for n=3
        A_cov = np.diag(0.2*np.ones(n**2))

    while not valid:
        if time.time() - start > timeout:
            raise RuntimeError("timeout of {} seconds exceeded".format(timeout))
        A = np.random.multivariate_normal(A_mean,A_cov).reshape(n,n)
        if intrinsic is None:
            valid = radius(A) > 1
        elif intrinsic:
            valid = radius(np.abs(A)) < 1
        else:
            valid = radius(A) < 1 and radius(np.abs(A)) > 1
    if intrinsic:
        assert radius(A) < 1, (radius(np.abs(A)), radius(A))

    structure = Adj,F,A,activation
    return structure

def product_SCM(Adj): # latest attempt
    A = np.ones_like(Adj)
    
    def F(V,A,U):
        n, num_points = np.shape(V)
        # assert num_points > n
        assert np.allclose(np.unique(A), (0,1)), np.unique(A)
        out_V = np.zeros_like(V)
        for i in range(n):
            mask = A[i,:].astype(bool)
            if np.sum(mask) > 0:
                out_V[i] = np.exp(np.sum(np.log(np.abs(V[mask,:])),axis=0))
                out_V[i] *= np.sign(np.product(V[mask,:],axis=0))
                out_V[i] += U[i,:]
            else:
                out_V[i] = U[i,:]
        return out_V
    
    structure = Adj,F,A,"product"
    return structure

def sample_market(Adj):
    A = np.ones_like(Adj)
    a = np.random.uniform(.5,2)
    b = np.random.uniform(.5,2)
    Fs = lambda P, Us: a*P**2 + Us
    Fd = lambda P, X, Ud: -b*X*P + Ud
    Fp = lambda P, S, D: P + D - S
    Fx = lambda P, Ux: P*Ux
    
    def F(data,Adj,U):
        "Accepts 1d data, and computes update"
        So,Do,Po,Xo = data.reshape((4,None))
        Us,Ud,Up,Ux = U
        S = Fs(P,Us)
        D = Fd(P,X,Ud)
        P = Fp(P,S,D)
        X = Fx(P,Ux)
        return np.hstack((S,D,P,X))
    
    structure = Adj,F,A,"market"
    return structure

##################### Potential Response ########################

def potential_response_orbits(structure,U,eps=1e-5,orbits=False,max_iters=1e3,bound=1e4): # From networkx
    """ structure is a tuple contianing everything necessary to construct
    the structural functions: (Adj, F, A)
    Evolves the system to equilibirum, with graph given by Adj.
    Assumes Adj is strongly connected, and that the system converges quickly (i.e. linearly).
    F(V,A,U) are the structural functions mapping V to V.
    A and U are the parameters to pass to F.
    Convergence is set as three random orbits coming within epsilon.
    The average of the final states is returned as the response.

    Returns:
        equilibrium, of shape (n,num_points)
    """
    Adj,F,A,f_type = structure
    n = Adj.shape[0]

    assert len(U.shape)==2
    if U.shape[0] != n:
        U = U.T
    
    # Verify Adj is a valid adjacency matrix
    assert np.all(np.unique(Adj) == [0,1])
    
    # Initialize 3 random orbits
    point1 = np.random.normal(0,1,U.shape) # was std 5
    point2 = np.random.normal(0,1,U.shape)
    point3 = np.random.normal(0,1,U.shape)
    
    if orbits:
        orbit1, orbit2, orbit3 = [point1,], [point2,], [point3,]
    
    distance12 = np.linalg.norm(point1 - point2)
    distance23 = np.linalg.norm(point2 - point3)

    num_iters = 0
    while distance12 + distance23 > eps and num_iters<max_iters:
        num_iters += 1
        point1 = F(point1,A*Adj,U)
        point2 = F(point2,A*Adj,U)
        point3 = F(point3,A*Adj,U)

        # Catch non-convergence early if possible
        for pnt in [point1,point2,point3]:
            if np.any(np.abs(pnt)>bound):
                return None
        
        distance12 = np.linalg.norm(point1 - point2)
        distance23 = np.linalg.norm(point2 - point3)
        
        if orbits:
            orbit1.append(point1)
            orbit2.append(point2)
            orbit3.append(point3)
    
    equilibrium = (point1 + point2 + point3)/3

    if num_iters == max_iters:
        return None

    if orbits:
        return equilibrium, (np.array(orbit1), np.array(orbit2), np.array(orbit3))
    else:
        return equilibrium

def potential_response(structure,U,eps=1e-5,orbits=False,max_iters=1e3,bound=1e4): # From networkx
    """ structure is a tuple contianing everything necessary to construct
    the structural functions: (Adj, F, A)
    Evolves the system to equilibirum, with graph given by Adj.
    Assumes Adj is strongly connected, and that the system converges quickly (i.e. linearly).
    F(V,A,U) are the structural functions mapping V to V.
    A and U are the parameters to pass to F.
    Convergence is set as three random orbits coming within epsilon.
    The average of the final states is returned as the response.

    Returns:
        equilibrium, of shape (n,num_points)
    """
    if orbits:
        raise NotImplementedError

    Adj,F,A,f_type = structure
    n = Adj.shape[0]

    assert len(U.shape)==2
    if U.shape[0] != n:
        U = U.T
    n2, num_points = U.shape
    assert n == n2
    
    # Verify Adj is a valid adjacency matrix
    assert np.all(np.unique(Adj) == [0,1])

    # # Compute Jacobian, if able, for speed-up
    # if f_type == "relu":
    #     def J(V_flat):
    #         "Allow to work on flattened input"
    #         V = V_flat.reshape((n,num_points))
    #         zero_mask = F(V,A*Adj,U) < 0
    #         A_broad = np.broadcast_to(A*Adj,(n,n,num_points))
    #         A_broad[zero_mask] == 0
    #         I_broad = np.broadcast_to(np.eye(n),(n,num_points))
    #         new_V = A_broad - I_broad
    #         return new_V.ravel()
    # elif f_type == "linear":
    #     def J(V_flat):
    #         "Allow to work on flattened input"
    #         V = V_flat.reshape((n,num_points))
    #         new_V = A*Adj - np.eye(n)
    #         return new_V.ravel()
    # else:
    #     J = None
    J = None
    

    # Find the fixed point (3 times, for comparison)
    def G(V_flat):
        "Allow to work on flattened input"
        V = V_flat.reshape((n,num_points))
        new_V = F(V,A*Adj,U) - V
        return new_V.ravel()

    fixed_points = []
    for k in range(1):
        initial = np.random.normal(0,1,U.shape) # was std 5
        if J is None:
            if f_type == "product":
                sol = optimize.root(G, initial, method='hybr', tol=eps)
            else:
                sol = optimize.root(G, initial, method='diagbroyden', tol=eps)
        else:
            sol = optimize.root(G, initial, method='hybr', tol=eps, jac=J)
        if sol.success:
            fixed_points.append(sol.x.reshape(n,num_points))

    # Check that each fixed point is valid, and compute equilibrium
    num_fixed = len(fixed_points)
    if num_fixed > 0:
        for fixed in fixed_points:
            assert np.allclose(fixed,F(fixed,A*Adj,U),atol=50*eps),np.max(np.abs(fixed-F(fixed,A*Adj,U)))
        if num_fixed > 1:
            # Check that there aren't multiple fixed points
            for (i,j) in combinations(range(num_fixed),2):
                assert np.allclose(fixed_points[i],fixed_points[j]) # could soften
            # If they're all equal, take their average (reduce numerical error)
            equilibrium = np.mean(fixed_points,axis=0)
            assert np.allclose(equilibrium.shape,(n,num_points))
        else:
            equilibrium = fixed_points[0]
        return equilibrium
    else:
        return None

##################### Sample Observational Distribution ########################

def sample_observational(structure,num_points=2100):
    Adj,F,A,f_type = structure
    n = Adj.shape[0]

    if f_type == "market":
        Us = np.random.uniform(0,100,num_points)
        Up = np.copy(Us)
        Ud = np.random.uniform(100,200,num_points)
        Ux = np.random.uniform(.5,1.7,num_points)
        U = np.vstack((Us,Ud,Up,Ux))
    else:
        U_mean = -0.5*np.ones(n)
        U_cov = np.diag(0.2*np.ones(n))
        U = np.random.multivariate_normal(U_mean,U_cov,int(num_points))
    return potential_response(structure,U)

##################### Enumerate all d-separations ########################

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def all_candidate_independencies(n):
    """For vertices in range(n), computes all possible combinations 
    of X, Y, and Z

    yields X (int), Y (int), Z (set) 
    """
    for X in range(n):
        for Y in range(X+1,n):
            V_minus_X_Y = set(range(n)) - set((X,Y))
            for Z in powerset(V_minus_X_Y):
                yield X, Y, Z


# If I understand correctly, the nx code for d-separation can be used without modificaiton?
# I looked over it, tried various d-separations with the classic counterexample, in sum for 20 minutes...
# ...and it passed at least that loose investigation.

def d_separated(G, x, y, z): #From NetworkX documentation
    """
    Return whether node sets ``x`` and ``y`` are d-separated by ``z``.

    Parameters
    ----------
    G : graph
        A NetworkX DAG.

    x : set
        First set of nodes in ``G``.

    y : set
        Second set of nodes in ``G``.

    z : set
        Set of conditioning nodes in ``G``. Can be empty set.

    Returns
    -------
    b : bool
        A boolean that is true if ``x`` is d-separated from ``y`` given ``z`` in ``G``.

    Raises
    ------
    NetworkXError
        The *d-separation* test is commonly used with directed
        graphical models which are acyclic.  Accordingly, the algorithm
        raises a :exc:`NetworkXError` if the input graph is not a DAG.

    NodeNotFound
        If any of the input nodes are not found in the graph,
        a :exc:`NodeNotFound` exception is raised.

    """

#     if not nx.is_directed_acyclic_graph(G):
#         raise nx.NetworkXError("graph should be directed acyclic")

    union_xyz = x.union(y).union(z)

    if any(n not in G.nodes for n in union_xyz):
        raise nx.NodeNotFound("one or more specified nodes not found in the graph")

    G_copy = G.copy()

    # transform the graph by removing leaves that are not in x | y | z
    # until no more leaves can be removed.
    leaves = deque([n for n in G_copy.nodes if G_copy.out_degree[n] == 0])
    while len(leaves) > 0:
        leaf = leaves.popleft()
        if leaf not in union_xyz:
            for p in G_copy.predecessors(leaf):
                if G_copy.out_degree[p] == 1:
                    leaves.append(p)
            G_copy.remove_node(leaf)

    # transform the graph by removing outgoing edges from the
    # conditioning set.
    edges_to_remove = list(G_copy.out_edges(z))
    G_copy.remove_edges_from(edges_to_remove)

    # use disjoint-set data structure to check if any node in `x`
    # occurs in the same weakly connected component as a node in `y`.
    disjoint_set = UnionFind(G_copy.nodes())
    for component in nx.weakly_connected_components(G_copy):
        disjoint_set.union(*component)
    disjoint_set.union(*x)
    disjoint_set.union(*y)

    if x and y and disjoint_set[next(iter(x))] == disjoint_set[next(iter(y))]:
        return False
    else:
        return True

def all_d_separations(Adj):
    """Generate all d-separations of Adj"""
    n = np.shape(Adj)[0]
    G = nx.DiGraph(Adj.T)
    for candidate in all_candidate_independencies(n):
        X, Y, Z = candidate
        if d_separated(G,{X},{Y},Z):
            yield candidate

##################### Check if an independence holds in P(V) ########################

def obs_indep(candidate,obs_data,p_thresh=1e-1,p_val=False):
    """ Checks if the candidate independence holds in obs_data, of shape (n,num_points).
    True if independent, False otherwise.
    """
    X, Y, Z = candidate
    n, num_points = np.shape(obs_data)
    assert num_points > n
    assert num_points > 2000, num_points

    x = obs_data[X,:].reshape((num_points,1))
    y = obs_data[Y,:].reshape((num_points,1))
    z = obs_data[tuple(Z),:].T
    pval = fcit.test(x, y, z)
    assert not np.isnan(pval)
    if p_val:
        return pval
    else:
        return pval > p_thresh


##################### Check if an independence holds in P(V) ########################

def check_d_separation(structure,d_seps,got_nanned):
    keep_sampling = True
    num_iters = 0
    pvals = []
    while keep_sampling and num_iters < 20:
        num_iters += 1
        obs_data = sample_observational(structure)
        if obs_data is None or not np.isfinite(obs_data).all():
            got_nanned += 1
            continue
        else:
            for candidate in d_seps:
                try:
                    pval = obs_indep(candidate,obs_data,p_val=True)
                    pvals.append((candidate,pval))
                except AssertionError as E:
                    print(E)
            keep_sampling = False
            break # repetative, just being sure...bad practice :)
    if num_iters == 20:
        print(structure)
        print("Could not find obs. distribution.")
    return pvals, got_nanned