function X_out = prox_matrix_l_21(X_in, lambda)
% PROX_L_21    The proximal operator of the l2,1 norm.
%
%   prox_l_21(v,lambda) is the proximal operator of the l2,1 norm
%   with parameter lambda.

    tmp = sqrt(sum(X_in.*X_in,1));
    mult = prox_l1(tmp,lambda)./tmp;
    X_out=X_in*diag(mult);
    
end
