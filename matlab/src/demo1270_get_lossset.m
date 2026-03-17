function ret = demo1270_get_lossset( typ_loss, param )

switch typ_loss
case {'lr'}
  fh_loss_p = @(z)logiloss(z); 
  fh_loss_d = @(a1)logiloss_ast(a1); 
  fh_grad_p = @(z)logiloss_grad(z); 
  kkn_alph   = [0,1]; 
  gam_sm    = 4.0; 
otherwise
  tsassert(0);
end

ret.fh_loss_p = fh_loss_p;
ret.fh_loss_d = fh_loss_d;
ret.fh_grad_p = fh_grad_p;
ret.kkn_alph  = kkn_alph;
ret.gam_sm    = gam_sm; 
