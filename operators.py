import sys, warnings
from numpy import max, transpose, diag, dot, eye, max, min, mean, nan, zeros, argsort, multiply, \
    sum, sqrt, divide, isnan, ones, inf, any, logical_not, sign, hstack, vstack, exp, where, logical_and, logical_or
from numpy.linalg import svd, pinv, eig, norm, solve, cholesky
from scipy.sparse import issparse, eye as speye, csr_matrix
from scipy.sparse.linalg import spsolve, lsqr


def prox_l1(v, lam):
    return max(0, v - lam) - max(0, -v - lam)




def project_affine(v, A, b, C=None, d=None):
#  PROJECT_AFFINE    Project a point into an affine set.
# 
#    project_affine(v,A,b) is the projection of v onto
#    the affine set { x | Ax = b }.
# 
#    You can also call the function as
# 
#      [x C d] = project_affine(v,A,b)
# 
#    and then call it again with different argument v2 via
# 
#      x2 = project_affine(v2,A,b,C,d)
# 
#    If calling the function repeatedly with the same A and b,
#    all evaluations after the initial one will be much faster
#    when passing in the cached values C and d.

    if not C is None and not d is None:
        x = dot(C,v) + d
    else:
        pA = pinv(A)
        C = eye(v.shape[0]) - dot(pA,A)
        d = dot(pA,b)
        x = dot(C,v) + d

    return (x, C, d)

def project_box(v, l, u):
#  PROJECT_BOX    Project a point onto a box (hyper-rectangle).
# 
#    project_box(v,l,u) is the projection of v onto
#    the set { x | l <= x <= u }.

    return max(l, min(v, u))

def project_consensus(*args):
#  PROJECT_CONSENSUS    Bring a set of points into consensus.
# 
#    project_consensus(v1,v2,...,vn) returns the elementwise
#    average of v1, ..., vn, i.e., the projection of the vi
#    onto the consensus set.
    x = mean(args, axis=0)


def project_pos(v):
#  PROJECT_POS    Project a point onto the nonnegative orthant.
# 
#    project_pos(v) is the positive part of v.

    return max(v,0)

def project_semidef_cone(V):
#  PROJECT_SDC    Project a point onto the semidefinite cone.
# 
#    project_sdc(V), where V is a symmetric n x n matrix, is the
#    projection of V onto the semidefinite cone.

    eigvals,eigvecs = eig(V)
    return reduce(dot, [eigvecs, max(0,eigvals), transpose(eigvecs)])


def project_soc(v):
#  PROJECT_SOC    Project a point onto the second-order cone.
# 
#    Suppose v is an (n+1)-dimensional vector, so v = [tv0], where t is a
#    scalar and v0 is n-dimensional. Then project_soc(v) is the projection of
#    (t,v0) onto the second-order cone the result is also (n+1)-dimensional.
#
#  assume v = (t, v0)
    nv = norm(v[1:-1])
    x  = nan * zeros(v.shape)
    if nv <= -v[0]:
        x = zeros(v.shape)
    elif nv <= v[0]:
        x = v
    else:
        r = 0.5*(1 + v[0]/nv)
        x[0]     = r*nv
        x[1:-1] = r*v[1:-1]
    return x


def prox_l1_k_support(v, lam , k):
#  PROX_L1 The proximal operator of the l1 norm, with support bound by k
#  prox_l1(v, lambda ) is the proximal operator of the l1 norm with parameter lambda.
    v_shape = v.shape
    idx = argsort(v.flatten())[::-1]
    v_lt_xk = idx[k + 1:-1]
    v[v_lt_xk] = 0
    x = max(0, v - lam ) - max(0, -v - lam )
    x = x.reshape(v_shape)
    return x


def prox_l_21(v, lam):
#  PROX_L_21    The proximal operator of the l2,1 norm.
# 
#    prox_l_21(v,lambda) is the proximal operator of the l2,1 norm
#    with parameter lambda.

    return (max(0, norm(v,2) - lam) / norm(v,2)) * v




def generalized_prox_matrix(v, prox_u, prox_s, prox_v):

    [U,S,V] = svd(v, full_matrices=False, compute_uv=True)
    out = reduce(dot, [prox_u(U),
                     diag(prox_s(diag(S))),
                     prox_v(transpose(v))])

    return out



def prox_matrix(v, lam, prox_s):
#  PROX_MATRIX    The proximal operator of a matrix function.
# 
#    Suppose F is a orthogonally invariant matrix function such that
#    F(X) = f(s(X)), where s is the singular value map and f is some
#    absolutely symmetric function. Then
# 
#      X = prox_matrix(V,lambda,prox_f)
# 
#    evaluates the proximal operator of F via the proximal operator
#    of f. Here, it must be possible to evaluate prox_f as prox_f(v,lambda).
# 
#    For example,
# 
#      prox_matrix(V,lambda,@prox_l1)
# 
#    evaluates the proximal operator of the nuclear norm at V
#    (i.e., the singular value thresholding operator).

    identity = lambda tmp:tmp
    return generalized_prox_matrix(v, identity, prox_s, identity)


def prox_matrix_l_21(X, lam ):
    #  PROX_L_21    The proximal operator of the l2, 1 norm.
    # 
    #  prox_l_21(v, lambda ) is the proximal operator of the l2, 1 norm
    #  with parameter lambda.

    column_norms = sqrt(sum(multiply(X,X),0))
    mult = divide( prox_l1(column_norms, lam ), column_norms)
    return dot(X, diag(mult))

def prox_precompose(prox_f, t, b):
#  PROX_PRECOMPOSE    Proximal precomposition rule.
# 
#    Let g(x) = f(tx + b). Then
# 
#      prox_g = prox_precompose(prox_f,t,b).
# 
#    Here, prox_f only takes v and lambda as arguments.

    prox_g = lambda v, lam: ((1.0/t) * (prox_f(dot(v, t) + b, 1/(lam*(t**2)) - b)))
    return prox_g




def prox_quad(v, lam, A, b):

#  PROX_QUAD    The proximal operator of a quadratic.
# 
#    prox_quad(v,lam,A,b)

    rho = 1.0/lam
    n = A.shape
    if issparse(A):
        x = spsolve(A + rho*speye(n) , (rho*v - b))
    else:
        x = solve(A + rho*eye(n), (rho*v - b))


def prox_sum_square(v, lam):
#  PROX_SUM_SQUARE    Proximal operator of sum-of-squares.
# 
#    prox_sum_square(v,lambda) is the proximal operator of
#    (1/2)||.||_2^2 with parameter lambda.

    return (1.0/(1.0 + lam))*v



def prox_separable(v, fp, lam=None, l=None, u=None, x0=None, tol=None, MAX_ITER=None):
#  PROX_SEPARABLE   Evaluate the prox operator of a fully separable function.
#
#  Arguments:
#
#   v is the point at which to evaluate the operator.
#   fp is a subgradient oracle for the function.
#   lambda (optional) is the proximal parameter defaults to 1.
#   l (optional) is a lower bound for x defaults to -Inf.
#   u (optional) is an upper bound for x defaults to Inf.
#   x0 (optional) is a value at which to warm start the algorithm.
#   tol (optional) is a stopping tolerance.
#   MAX_ITER (optional) is the maximum number of iterations.
#
#  Examples:
#
#   v = randn(n,1)
#   x = prox_separable(v, 1, @(w) sign(w))
#   [x iter] = prox_separable(v, 1, @(w) sign(w))
#
#  This function can be called in vectorized form if fp is vectorized,
#  i.e., if fp works elementwise, then v, l, u, and x0 can be vectors.

    n = v.shape[0]

    arg_checker = lambda arg: not arg is None or isnan(arg) or arg.size()==0

    if arg_checker(lam):
        lam = 1

    rho = 1.0/lam

    if arg_checker(l):
        l = -inf*ones((n,1))

    if arg_checker(u):
        u = -inf*ones((n,1))

    if arg_checker(x0):
        x0 = zeros((n,1))

    if arg_checker(tol):
        tol = 1e-8

    if arg_checker(MAX_ITER):
        MAX_ITER = 500


    iter = 0
    x = max(l, min(x0, u))

    while any(u-l > tol) and iter < MAX_ITER:
        g = fp(x) + rho*(x - v)

        idx = (g > 0)
        l[idx] = max(l[idx], x[idx] - (g[idx]/rho))
        u[idx] = x[idx]

        idx = logical_not(idx)
        u[idx] = min(u[idx], x[idx] - g[idx]/rho)
        l[idx] = x[idx]

        x = (l + u)/2.0
        iter += 1

    if any(u-l > tol):
        sys.stderr.write('Warning: %d entries did not converge max interval size = %d.\n' % (sum(u-l > tol), max(u-l)))

    return (x, iter)


def prox_max(v, lam):
#  PROX_MAX    The proximal operator of the max function.
#
#    prox_max(v,lambda) is the proximal operator of the max
#    of the entries of v. This function is not vectorized.

    TOL      = 1e-8
    MAX_ITER = 100
    rho = 1.0/lam

    n = v.shape[0]
    tl = min(v) - 1.0/n
    tu = max(v)

    g = lambda t: (sum(max(0, rho*(v - t))) - 1)

    iter = 0
    while tu - tl > TOL and iter < MAX_ITER:
        t0 = (tl + tu)/2.0
        if sign(g(t0)) == sign(g(tl)):
            tl = t0
        else:
            tu = t0
        iter += 1

    x = min(t0,v)

    if tu - tl > TOL:
        warnings.warn('Algorithm did not converge.')

    return x



def project_graph(v, A, AA, L=None, D=None, P=None):
#  PROJECT_GRAPH    Project a point onto the graph of a linear operator.
#
#    If A is an m x n matrix and v = [v1v2] is a vector such that v1 is of
#    length n and v2 is of length m, then
#
#      z = project_graph(v,A)
#
#    gives the projection of v onto the graph { (x,y) | y = Ax } of A. The
#    vector z can then be sliced into its first n components and the remaining
#    m components.
#
#    If A is dense and m <= n, then the projection involves computing
#    the Cholesky factor L of I + A*A'. One can obtain A*A' and L by calling
#
#      [z AA L] = project_graph(v,A)
#
#    and can pass in AA and L to avoid them being recomputed via
#
#      z = project_graph(v,A,AA,L)
#
#    This allows for factorization caching when calling project_graph
#    multiple times for the same A.
#
#    This applies similarly when m >= n, only AA is A'*A and L is the
#    Cholesky factor of I + A'*A.
#
#    This also applies similarly when A is sparse, in which case we
#    use a permuted LDL factorization of some matrix. In this case,
#    one can obtain P, L, and D via
#
#      [z L D P] = project_graph(v,A)
#
#    and can pass these back in via
#
#      z = project_graph(v,A,[],L,D,P)
#
#    Note that L has different meanings depending on whether or not A is sparse.

    m, n = A.shape
    c = v[0:n]
    d = v[n+1:-1]

    if issparse(A):
        return NotImplementedError('need to find a good LDL implementation for python')
        # if P is None or P.size==0 or L is None or L.size==0 or D is None or D.size==0:
        #     K = vstack( hstack(speye(n), transpose(A)), hstack(A, logical_not(speye(m)).astype(float)) )
        #     [L,D,P] = ldl(K)
        #
        #
        # z = dot(P , solve(transpose(L) , solve(D , solve(L , (transpose(P) * csr_matrix(vstack( c + dot(transpose(A), d) , zeros(m,1) )))))))
        #
        # return (z, L, D, P)
    else:
        if m <= n:
            if AA is None or AA.size==0:
                AA = dot(A,transpose(A))

            if L is None or L.size == 0:
                L = cholesky(eye(m) + AA)

            y = solve(L, solve(transpose(L), (dot(A,c) + dot(AA,d))))
            x = c + dot(transpose(A),(d - y))
        else:
            if AA is None or AA.size==0:
                AA = dot(transpose(A),A)

            if L is None:
                L = cholesky(eye(n) + AA)

            x = solve(L, solve(transpose(L), (c + dot(transpose(A),d))))
            y = dot(A, x)

        return (vstack(x,y), AA, L)



def project_exp(v):
#  PROJECT_EXP    Project points onto the exponential cone.
#
#    When v is a 1 x 3 vector, project_exp(v) is the projection
#    of v onto the exponential cone. When v is an n x 3 vector,
#    project_exp(v) projects each row of v onto the cone
#    in a vectorized fashion.
#
#    For reference, the exponential cone and its dual are given by
#      Kexp   = { (x,y,z) | ye^(x/y) <= z, y > 0 }
#      Kexp^* = { (u,v,w) | u < 0, -ue^(v/u) <= ew } cup { (0,v,w) | v,w >= 0 }

    def hessg(w):
        r = w[0]
        s = w[1]

        h = exp(r/s)*[ [1/s,    -r/s^2,   0],
                       [-r/s^2, r^2/s^3,  0],
                       [0,      0,        0] ]
        return h

    r = v[:,0]
    s = v[:,1]
    t = v[:,2]
    x = nan*ones(v.shape)

    #  v in cl(Kexp)
    idx = ( (multiply(s, exp(divide(r,s))) <= t & s > 0) | (r <= 0 & s == 0 & t >= 0) )
    x[idx,:] = v[idx,:]

    # -v in Kexp^*
    idx =  (logical_and(-r < 0, multiply(r,exp(divide(s,r)))) <= logical_or(multiply(-exp(0),t), multi_logical_and(-r == 0, -s >= 0, -t >= 0)) )
    x[idx,:] = 0

    # special case with analytical solution
    idx = (r < 0 & s < 0)
    x[idx,:] = v[idx,:]
    x[idx,2] = max(x[idx,1],0)
    x[idx,3] = max(x[idx,2],0)

    # minimize ||x - v||^2 subject to se^{r/s} = t via primal-dual Newton method
    # these components are computed serially, so much slower
    idx = where(isnan(x[:,0]))

    g     = lambda w:   w[1]*exp(w[0]/w[1]) - w[2]
    gradg = lambda w:   vstack(exp(w[0]/w[1]), exp(w[0]/w[1])*(1 - w[0]/w[1]), -1 )

    alpha = 0.001
    beta = 0.5

    for i in range(0, idx.shape[0]):
        print('newton')
        u = transpose(v[idx[i],:])
        u[1] = max(u[1],1)
        u[2] = max(u[2],1)
        y = 1 # dual variable

        r = lambda w,z: vstack( w - transpose(v[idx[i],:]) + z*gradg(w), g(w) )

        for iter in range(0,100):
            KKT = vstack( hstack(eye(3)+y*hessg(u), gradg(u)) , hstack(transpose(gradg(u)), 0) )
            z = solve(KKT , -r(u,y)) #urgggh
            du = z[0:3]
            dy = z[3]

            # backtracking line search
            t = 1
            ustep = u + t*du
            ystep = y + t*dy
            while ustep[1] < 0 or (norm(r(ustep, ystep)) > (1 - alpha*t)*norm(r(u, y))):
                t = beta*t
                ustep = u + t*du
                ystep = y + t*dy


            u = ustep
            y = ystep

            if abs(g(u)) < 1e-8 and norm(r(u,y)) <= 1e-8:
                x[idx[i],:] = u
                break

    return x

def multi_logical_and(*args):
    out = args[0]
    for i in range(1,len(args)):
        out = logical_and(out,args[i])
    return out