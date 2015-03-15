function x = prox_l_21(v, lambda)
% PROX_L_21    The proximal operator of the l2,1 norm.
%
%   prox_l_21(v,lambda) is the proximal operator of the l2,1 norm
%   with parameter lambda.

    x = (max(0, norm(v,2) - lambda)/norm(v,2))*v;
end
