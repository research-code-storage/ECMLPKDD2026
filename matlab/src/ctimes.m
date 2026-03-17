function ret = ctimes( A, b )
%
%
%

b = b(:); 
[nrows,ncols] = size(A); 
tsassert( size(b) == [nrows,1] ); 
ret = A.*(b*ones(1,ncols)); 

