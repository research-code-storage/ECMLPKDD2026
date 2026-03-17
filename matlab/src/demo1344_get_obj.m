function [obj_p,obj_d,sco2s_q] = demo1344_get_obj( alph1, sigbeta1, param_pfgd1 )

kernmat1_big = param_pfgd1.kernmat1_big;
sigvec1 = param_pfgd1.sigvec1; 
ncons   = numel(sigvec1); 
ntras   = size(kernmat1_big,2)-ncons; 
lam1    = param_pfgd1.lam1;
lamn1   = lam1.*ntras; 
ilamn1  = 1./lamn1; 
r_tra   = 1:ntras; 
r_q     = (1:ncons)+ntras; 
kernmat1_tra = kernmat1_big(r_tra,r_tra); 
kernmat1_qx  = kernmat1_big(r_q,r_tra); 
kernmat1_qq  = kernmat1_big(r_q,r_q); 

fh_lr      = param_pfgd1.fh_lr; 
fh_loss_p  = param_pfgd1.fh_loss_p;
fh_grad_p  = param_pfgd1.fh_grad_p;
fh_loss_d  = param_pfgd1.fh_loss_d;
kkn_alph   = param_pfgd1.kkn_alph;

param1_proj.qpmeth1 = 'cqkp'; 
fh_proj1 = @(alph1_half,beta1_half) demo1344_proj( alph1_half, beta1_half, lam1, sigvec1, kernmat1_qq, kernmat1_qx, param1_proj); 
pnxs_rkhs = diag(kernmat1_big(r_tra,r_tra)); 

alphbar1_big = [ ilamn1.*alph1; sigbeta1 ]; 
sco2s_big = kernmat1_big*alphbar1_big; 
sco2s_tra = sco2s_big(1:ntras); 
sco2s_q   = sco2s_big(ntras+1:end); 

kernmat1_tra = kernmat1_big(r_tra,r_tra); 
kernmat1_qq  = kernmat1_big(r_q,r_q); 
ualph1     = -fh_grad_p(sco2s_tra); 
usigbeta1  = fh_proj1( ualph1, zeros(ncons,1) ); 
ubar1_big  = [ ilamn1.*ualph1; usigbeta1 ];
reg_p = 0.5.*lam1.*dot( ubar1_big, kernmat1_big*ubar1_big ); 
reg_d = -0.5.*lam1.*(dot( ualph1, kernmat1_tra*ualph1 ).*ilamn1.^2 - dot( usigbeta1, kernmat1_qq*usigbeta1 )); 
tsassert_zro( reg_p + reg_d , 1e-8); 
sco1s_big = kernmat1_big*ubar1_big; 
sco1s_tra = sco1s_big(r_tra); 
loss_p = mean( fh_loss_p( sco1s_tra ) );
loss_d = -mean( fh_loss_d( -ualph1 ) );
obj_p = reg_p + loss_p; 
obj_d = reg_d + loss_d; 
