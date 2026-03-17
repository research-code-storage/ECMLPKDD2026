function [prob_active, param_out] = demo1334_exp_pfgd( param_dataset, param_exp_pfgd1 )

  function value = get_(param, name, default)
    if isfield( param, name )
      value = getfield(param, name); 
    else
      value = default;
    end
  end

  % --- Parameters ---
  rng_init =  get_(param_exp_pfgd1, 'rng_init',      0);
  bias_factor = get_(param_exp_pfgd1, 'bias_factor', 10.0);
  dbname1     = get_(param_exp_pfgd1, 'dbname', 'uci_adult');

  % Parameters of Kernel
  typ_kern   = get_(param_exp_pfgd1, 'typ_kern', 'lin');
  gam_rbf    = get_(param_exp_pfgd1, 'gam_rbf',  1.0/4.0);

  % Parameters of Sign constraints 
  mode_sgncon   = get_(param_exp_pfgd1, 'mode_sgncon', 'sc');
  b_prime       = get_(param_exp_pfgd1, 'b_prime',     1);
  s_prime       = get_(param_exp_pfgd1, 's_prime',     1);

  % Parameters of Loss / Optimization
  typ_loss   = get_(param_exp_pfgd1, 'typ_loss', 'lr');
  gam_sm     = get_(param_exp_pfgd1, 'gam_sm',   0.01);
  lamn       = get_(param_exp_pfgd1, 'lamn',     1.0);
  
  % Parameters of Experiment
  ncvs_rnd = get_(param_exp_pfgd1, 'ncvs_rnd', 200);
  ntras    = get_(param_exp_pfgd1, 'ntras',    100); 

  % --- Logic ---
  rng(rng_init);

  X_all = param_dataset.X_all;
  y_all = param_dataset.y_all;
  cvec_sgn = param_dataset.cvec_sgn;
  
  [~,npts] = size( X_all ); 
  X_all    = [X_all; bias_factor.*ones(1,npts)]; 
  cvec_sgn(end+1) = 0;
  
  [nfeas_total, ~] = size(X_all);
  lmat_tra = h34_gen_lmat_tra( y_all, ntras, ncvs_rnd );

  param1_sckmtwo.typ_kern = typ_kern; 
  param1_sckmtwo.gam_rbf  = gam_rbf; 
  fh_kern                 = demo1320_get_kern( param1_sckmtwo ); 

  count_active = zeros(nfeas_total, 1);
  
  for cv_rnd=1:ncvs_rnd
    rng(cv_rnd); 
    
    l_tra = lmat_tra(cv_rnd,:); 
    l_tst = ~l_tra; 
    X_tra = X_all(:,l_tra);     
    X_tst = X_all(:,l_tst); 
    y_tra = y_all(l_tra);       

    ntras_curr = numel(y_tra); 
    lam1  = lamn./ntras_curr; 
    
    param1_sckmtwo.X_tra    = X_tra;
    param1_sckmtwo.y_tra    = y_tra;
    param1_sckmtwo.X_tst    = X_tst;
    param1_sckmtwo.fh_kern  = fh_kern;
    
    if strcmp(mode_sgncon,'sf')
      param1_sckmtwo.cvec_sgn = zeros(1,numel(cvec_sgn));
    else
      param1_sckmtwo.cvec_sgn = cvec_sgn;
    end
      
    param1_sckmtwo.s1       = b_prime; 
    param1_sckmtwo.s2       = s_prime; 
    param1_sckmtwo          = demo1344_prepro_sckmtwo( param1_sckmtwo ); 

    l_con = cvec_sgn(:)' ~= 0; 
    
    param_loss.gam_sm = gam_sm;
    param_optim1      = demo1270_get_lossset( typ_loss, param_loss ); 

    param_optim1.kernmat1_big = param1_sckmtwo.kernmat1_big; 
    param_optim1.sigvec1      = param1_sckmtwo.sigvec1; 
    param_optim1.lam1         = lam1; 
    param_optim1.stochas      = 1; 
    param_optim1.y_tra        = param1_sckmtwo.y_tra; 
    param_optim1.K_qx_ekm     = param1_sckmtwo.K_qx_ekm; 
    param_optim1.K_xx_ekm     = param1_sckmtwo.K_xx_ekm; 
    param_optim1.mode_sgncon  = mode_sgncon; 
    param_optim1.verbose      = 0;

    param_sdca1 = param_optim1; 
    param_sdca1.lam = lam1; 
    thres_gap  = 1e-5; 

    K_xx_diag = diag(param1_sckmtwo.kernmat1_big(1:ntras_curr, 1:ntras_curr)); 
    R_mx_square = max(K_xx_diag);
    eta_L = (2 * gam_sm) / (2 * lam1 * gam_sm + R_mx_square);
    fh_lr = @(tea1) eta_L * ones(size(tea1));

    param_pfgd1 = param_optim1; 
    param_pfgd1.fh_lr   = fh_lr; 
    param_pfgd1.qpmeth1   = 'cqkp';
    res1_pfgd   = demo1344_train_pfgd( param_pfgd1 );

    alph1      = res1_pfgd.alph1; 
    sigbeta1   = res1_pfgd.sigbeta1; 
    sigvec1    = param1_sckmtwo.sigvec1;

    [~,~,sco2s_q] = demo1344_get_obj( alph1, sigbeta1, param_pfgd1 ); 
    active_thresh = 1e-3;
    
    is_active_reduced = (sco2s_q .* sigvec1) <= active_thresh;
    
    current_active_vec = zeros(nfeas_total, 1);
    current_active_vec(l_con) = is_active_reduced;
    
    count_active = count_active + current_active_vec;
  end

  prob_active = count_active ./ ncvs_rnd;
  
  param_out = param_exp_pfgd1;
end