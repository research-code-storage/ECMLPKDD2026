function xrvec1 = demo1291_rec( tea1, cd, avec1, sig1 )

ncons  = numel(sig1); 
npts   = numel(tea1); 
tsassert( size(tea1) == [1,npts] ); 
tsassert( size(cd) == [1,1] ); 
tsassert( size(avec1) == [ncons,1] ); 
tsassert( size(sig1) == [ncons,1] ); 

xvec1 = avec1*ones(1,npts)-ones(ncons,1)*tea1; 
xvec1 = ctimes( xvec1, sig1 ); 
xvec1 = max(0,xvec1); 
xvec1 = ctimes( xvec1, sig1 ); 
r1    = tea1./cd; 
xrvec1 = [xvec1;r1]; 


