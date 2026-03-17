function [objd1,grad1_objd,xrvec1] = demo1291_objd( tea1, cd, avec1, sig1 )

ncons = numel(avec1); 
npts  = size(tea1,2); 
tsassert( size(cd) == [1,1] ); 
tsassert( size(avec1) == [ncons,1] ); 
tsassert( size(tea1) == [1,npts] ); 
xrvec1 = demo1291_rec( tea1, cd, avec1, sig1 ); 
objd1  = demo1291_lag( xrvec1, tea1, cd, avec1 ); 

xvec1 = xrvec1(1:ncons,:); 
r1    = xrvec1(end,:); 
grad1_objd = csum(xvec1)-r1; 


