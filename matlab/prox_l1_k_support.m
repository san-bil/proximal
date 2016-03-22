function x = prox_l1_k_support(v, lambda, k)
% PROX_L1    The proximal operator of the l1 norm.
%
%   prox_l1(v,lambda) is the proximal operator of the l1 norm
%   with parameter lambda.
    [~,idx] = sort(v(:),'descend');
    v_lt_xk = idx(k+1:end);
    v(v_lt_xk) = 0;

    x = max(0, v - lambda) - max(0, -v - lambda);
    
end
