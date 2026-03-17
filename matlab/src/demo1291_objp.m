function objp1 = demo1291_objp( xrvec1, cd, avec1 )

ncons = numel(avec1); 
npts  = size(xrvec1,2); 
tsassert( size(xrvec1) == [ncons+1,npts] ); 
tsassert( size(cd) == [1,1] ); 
tsassert( size(avec1) == [ncons,1] ); 

xvec1 = xrvec1(1:ncons,:); 
r1    = xrvec1(end,:); 
term1 = 0.5.*csum(xvec1.^2); 
term2 = -avec1'*xvec1; 
term3 = 0.5.*cd.*r1.^2; 
objp1 = term1 + term2 + term3; 

