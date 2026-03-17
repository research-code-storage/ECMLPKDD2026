function [sigbeta1,res1] = demo1344_proj( alph1_half, beta1_half, lam1, sigvec1, kernmat1_qq, kernmat1_qx, param1_proj )
%
% Return %\vsig\odot\vbeta$ in ohp240922sckmsdca1-kato.pdf
%
[ncons,ntras] = size(kernmat1_qx); 
tsassert( size(alph1_half)   == [ntras,1] ); 
tsassert( size(beta1_half)   == [ncons,1] ); 
tsassert( size(lam1) == [1,1] ); 
tsassert( size(sigvec1) == [ncons,1] ); 
tsassert( abs(sigvec1) == 1 ); 
tsassert( size(kernmat1_qq) == [ncons,ncons] ); 
tsassert( size(kernmat1_qx) == [ncons,ntras] ); 

lamn1   = lam1.*ntras; 
ilamn1  = 1./lamn1; 
ncons   = numel(sigvec1); 
if ncons == 0,
    sigbeta1 = []; res1 = []; return; 
end
c2 = 0; c1 = 1; 
if ncons >= 2, 
    c2 = kernmat1_qq(1,2); 
end
if ncons >= 1, 
    c1 = kernmat1_qq(1,1); 
end
if abs(c2) ~= 0, 
    tsassert_zro( (c1-c2).*eye(ncons) + c2.*ones(ncons) - kernmat1_qq ); 
    cd = c2./(c1-c2); 
    avec1 = - (ilamn1.*kernmat1_qx*alph1_half + kernmat1_qq*(sigvec1.*beta1_half))./(c1-c2);  
    [~,r_srt] = sort(avec1); 
    avec1_srt   = avec1(r_srt); 
    sigvec1_srt = sigvec1(r_srt); 
    sigbeta1_srt = demo1334_solve( cd, avec1_srt, sigvec1_srt, param1_proj); 
    sigbeta1 = zeros(ncons,1); sigbeta1(r_srt) = sigbeta1_srt; 

else
    tsassert_zro( c2 )
    c1    = mean(diag(kernmat1_qq)); 
    ic1   = 1./c1; 
    avec1 = - ic1.*ilamn1.*diag(sigvec1)*kernmat1_qx*alph1_half; 
    beta1 = max(avec1,beta1_half); 
    sigbeta1 = sigvec1.*(beta1-beta1_half);     
end
res1 = []; 
