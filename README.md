# Proximal operators

This small set of utilities contains sample implementations of various proximal operators in
Python.

The derivations and background for these operators can be found in
*[Proximal Algorithms](http://www.stanford.edu/~boyd/papers/prox_algs.html)* 
by Parikh and Boyd.

## Requirements

The only requirements are Numpy, and Scipy for sparse matrices.

## Examples


## Proximal operators

The operators module include the following proximal operators:

* Projection onto an affine set
* Projection onto a box
* Projection onto the consensus set (averaging)
* Projection onto the exponential cone
* Projection onto the nonnegative orthant
* Projection onto the second-order cone
* Projection onto the semidefinite cone
* Proximal operator of a generic function (via CVX)
* Proximal operator of the *l1* norm
* Proximal operator of the max function
* Proximal operator of a quadratic function
* Proximal operator of a generic scalar function (vectorized)
* Proximal operator of an orthogonally invariant matrix function
* Precomposition of a proximal operator

## Authors

* [Sanjay Bilakhia]

## License

This code is released under a BSD license; see the "LICENSE" file.
