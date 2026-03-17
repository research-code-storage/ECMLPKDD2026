%
%
function x = vec(X)

[nrows,ncols] = size(X);
x             = reshape(X,nrows*ncols,1);

