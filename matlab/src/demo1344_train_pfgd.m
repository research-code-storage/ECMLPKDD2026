function res1 = demo1344_train_pfgd( param_pfgd1 )
%
% Projected full gradient descent in RKHS. 
% Orginated from demo1311_train_pfgd.m 
% This version uses the duality gap for termination condition. 
%
kernmat1_big = param_pfgd1.kernmat1_big;
sigvec1 = param_pfgd1.sigvec1(:); 
ncons   = numel(sigvec1); 
ntras   = size(kernmat1_big,2)-ncons; 
lam1    = param_pfgd1.lam1;
lamn1   = lam1.*ntras; 
ilamn1  = 1./lamn1; 
r_tra   = 1:ntras; 
r_q     = (1:ncons)+ntras; 
kernmat1_xx = kernmat1_big(r_tra,r_tra); 
kernmat1_qx = kernmat1_big(r_q,r_tra); 
kernmat1_qq = kernmat1_big(r_q,r_q); 

nepochs = 5000;
if isfield( param_pfgd1, 'nepochs' )
  nepochs = param_pfgd1.nepochs; 
end
tsassert( ~isfield( param_pfgd1, 'niters' ) );

thres_gap = 1e-3;
if isfield( param_pfgd1, 'thres_gap' )
  thres_gap = param_pfgd1.thres_gap;
end

mode_rec = 0;
if isfield( param_pfgd1, 'mode_rec' )
  mode_rec = param_pfgd1.mode_rec;
end

verbose = 0;
if isfield( param_pfgd1, 'verbose' )
  verbose = param_pfgd1.verbose;
end

alph1_new = zeros(ntras,1); 
if isfield( param_pfgd1, 'alph' )
  alph1_new = param_pfgd1.alph(:); 
end

beta1_new = zeros(ncons,1); 
if isfield( param_pfgd1, 'beta' )
  beta1_new = param_pfgd1.beta(:); 
end

iters_rec = ceil(logspace(0,log10(nepochs),100));
iters_rec = unique(iters_rec);
if isfield( param_pfgd1, 'iters_rec' )
  iters_rec = param_pfgd1.iters_rec;
end

obj1_stop = -1e+100; 
if isfield( param_pfgd1, 'obj1_stop' )
  obj1_stop = param_pfgd1.obj1_stop;
end

param1_proj.qpmeth1 = 'cqkp'; 
if isfield( param_pfgd1, 'qpmeth1' )
  param1_proj.qpmeth1 = param_pfgd1.qpmeth1;
end

fh_lr      = param_pfgd1.fh_lr; 
fh_loss_p  = param_pfgd1.fh_loss_p;
fh_grad_p  = param_pfgd1.fh_grad_p;

fh_proj1 = @(alph1_half,beta1_half) demo1344_proj( alph1_half, beta1_half, lam1, sigvec1, kernmat1_qq, kernmat1_qx, param1_proj ); 
pnxs_rkhs = diag(kernmat1_big(r_tra,r_tra)); 

niters_pfgd = nepochs; 
i_iter_rec  = 1;
eta1s   = fh_lr(1:niters_pfgd); 
obj1s_p = zeros(niters_pfgd,1); 
tms = zeros(1,numel(iters_rec)); tic; 
for tea1=1:niters_pfgd
    eta1 = eta1s(tea1); 
  alph1 = alph1_new(:); beta1 = beta1_new(:);
    sco3s_tra = ilamn1.*kernmat1_xx*alph1 + kernmat1_qx'*(sigvec1.*beta1); 
    sco3s_q   = ilamn1.*kernmat1_qx*alph1 + kernmat1_qq'*(sigvec1.*beta1); 
    pnw3 = ilamn1.*dot(alph1,sco3s_tra)+dot(sigvec1.*beta1,sco3s_q); 
    reg3_p    = 0.5.*lam1.*pnw3; 
    loss_p_each = fh_loss_p( sco3s_tra ); 
    loss_p = mean( loss_p_each );
    obj_p  = reg3_p + loss_p; 

    if i_iter_rec <= numel(iters_rec) & tea1 == iters_rec(i_iter_rec)
        obj1s_p( i_iter_rec ) = obj_p; 
        sigbeta1 = beta1.*sigvec1; 
        [obj2_p,obj2_d,sco2s_q] = demo1344_get_obj( alph1, sigbeta1, param_pfgd1 ); 
        gap1 = obj2_p - obj2_d; 
        tm1 = toc; tms( i_iter_rec ) = tm1;        
        if verbose > 0,
          fprintf('%d: obj_p=%g, obj2_p=%g, obj2_d=%g, gap=%g\n', tea1, obj_p, obj2_p, obj2_d, gap1 ); 
        end
        i_iter_rec = i_iter_rec + 1;        
        if obj_p <= obj1_stop, 
            fprintf('Observed obj_p <= obj1_stop\n'); 
            break; 
        end
        if gap1 < thres_gap,
            break; 
        end
        if i_iter_rec == numel(iters_rec)
          if verbose > 0
            fprintf('Final iteration!')
          end
        end
    end

    loss_p_each_grad_sco = fh_grad_p( sco3s_tra ); loss_p_each_grad_sco = loss_p_each_grad_sco(:); 
    alph1_nabla = (alph1 + loss_p_each_grad_sco).*lam1; 
    beta1_nabla = beta1.*lam1; 
    alph1_half = alph1 - eta1.*alph1_nabla; 
    beta1_half = beta1 - eta1.*beta1_nabla; 
    sigbeta6_delta = fh_proj1( alph1_half, beta1_half ); sigbeta6_delta = sigbeta6_delta(:);
    beta6_delta = sigbeta6_delta.*sigvec1; 
    alph1_new = alph1_half; 
    beta1_new = beta1_half + beta6_delta; 

end

res1.obj1s_p   = obj1s_p(1:i_iter_rec-1);
res1.iters_rec = iters_rec(1:i_iter_rec-1);
res1.tms   = tms(1:i_iter_rec-1);
res1.alph1     = alph1;
res1.sigbeta1  = beta1.*sigvec1; 
res1.fh_proj1  = fh_proj1; 

if isfield( param_pfgd1, 'K_qx_ekm' )
  y_tra    = param_pfgd1.y_tra; 
  yalph1   = (y_tra.*alph1); 
  K_xx_ekm = param_pfgd1.K_xx_ekm;
  term1 = ilamn1.*K_xx_ekm'*yalph1; 
  K_qx_ekm = param_pfgd1.K_qx_ekm;
  term2 = K_qx_ekm'*res1.sigbeta1; 
  sco1_tst = term1 + term2; 
  res1.sco_tst = sco1_tst; 
end
