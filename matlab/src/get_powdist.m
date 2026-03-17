% [ powdmat ] = get_powdist( Y, X )
% Y     : ndims x M
% X     : ndims x N
function [ powdmat ] = get_powdist( Y, X )

[ndims M] = size(Y);
N         = size(X,2);
ynorm     = csum( Y .* Y );
xnorm     = csum( X .* X );
YTX       = Y' * X;
powdmat   = ynorm'*ones(1,N)+ones(M,1)*xnorm-2*YTX;


