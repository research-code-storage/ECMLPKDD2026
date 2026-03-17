function param1_sckmtwo =  demo1344_prepro_sckmtwo( param1_sckmtwo ) 

X_tra    = param1_sckmtwo.X_tra; 
y_tra    = param1_sckmtwo.y_tra; 
X_tst    = param1_sckmtwo.X_tst; 
fh_kern  = param1_sckmtwo.fh_kern; 
cvec_sgn = param1_sckmtwo.cvec_sgn; 

[nfeas,ntras] = size(X_tra); 
[~,ntsts] = size(X_tst); 
tsassert( size(X_tst) == [nfeas,ntsts] ); 
tsassert( size(y_tra) == [ntras,1] ); 
tsassert( abs(y_tra)  == 1 );

l_con   = cvec_sgn(:)' ~= 0; 
sigvec1 = cvec_sgn(l_con); sigvec1 = sigvec1(:); 
ncons   = sum(l_con); 

s1 = param1_sckmtwo.s1; 
s2 = param1_sckmtwo.s2; 
tsassert( size(s1) == size(s2) ); 
B1  = eye(ncons); 
B1  = kron( B1, s1);
Q_b = eye(nfeas); Q_b = Q_b(:,l_con);  
Q_b = kron(Q_b,s2);
K_bb = fh_kern(Q_b,Q_b); 
K_qq = B1*K_bb*B1'; 

if ncons >= 2,
    c1 = K_qq(1,1); c2 = K_qq(1,2);
    tsassert_zro( (c1-c2).*eye(ncons) + c2.*ones(ncons) - K_qq ); 
end
K_bx = fh_kern(Q_b,X_tra); 
K_qx = B1*K_bx; 

K_xx  = fh_kern(X_tra,X_tra); 
K_bx_ekm = fh_kern(Q_b,X_tst); 
K_qx_ekm = B1*K_bx_ekm;          
K_xx_ekm  = fh_kern(X_tra,X_tst); 
kernmat1_big = [ K_xx, K_qx'; K_qx, K_qq ]; 
tmp1 = ones(ntras+ncons,1); tmp1(1:ntras) = y_tra; 
kernmat1_big = diag(tmp1)*kernmat1_big*diag(tmp1); 

param1_sckmtwo.kernmat1_big = kernmat1_big; 
param1_sckmtwo.K_qx_ekm     = K_qx_ekm; 
param1_sckmtwo.K_xx_ekm     = K_xx_ekm; 
param1_sckmtwo.sigvec1      = sigvec1; 
