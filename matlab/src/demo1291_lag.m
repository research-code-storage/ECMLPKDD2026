function lag1 = demo1291_lag( xrvec1, tea1, cd, avec1 )
%
%
%
ncons = numel(avec1); 
npts  = size(xrvec1,2); 
tsassert( size(xrvec1) == [ncons+1,npts] ); 
tsassert( size(cd) == [1,1] ); 
tsassert( size(avec1) == [ncons,1] ); 
tsassert( size(tea1) == [1,npts] ); 

xvec1 = xrvec1(1:ncons,:); 
r1    = xrvec1(end,:); 
objp1 = demo1291_objp( xrvec1, cd, avec1 ); 
term2 = tea1.*(csum(xvec1)-r1); 
lag1  = objp1 + term2; 

