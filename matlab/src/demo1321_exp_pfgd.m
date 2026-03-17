function [scos_cat, ytst_cat] = demo1321_exp_pfgd( param_dataset, param_exp_pfgd1 )

  function value = get_(param, name, default)
    if isfield( param, name )
      value = getfield(param, name); 
    else
      fprintf('Set default value [%s]: ', name);
      disp(default);
      value = default;
    end
  end

  %
  % Parameters
  %
  rng_init =  get_(param_exp_pfgd1, 'rng_init',      0);
  data_dir     = get_(param_exp_pfgd1, 'data_dir',      'dat');
  dbname1     = get_(param_exp_pfgd1, 'dbname',      'harbor-top');
  ntras       = get_(param_exp_pfgd1, 'ntras',       10);
  bias_factor = get_(param_exp_pfgd1, 'bias_factor', 10.0);

  % Parameters of Kernel
  typ_kern   = get_(param_exp_pfgd1, 'typ_kern', 'lin');
  gam_rbf    = get_(param_exp_pfgd1, 'gam_rbf',  1.0/4.0);

  % Parameters of Sign constraints 
  mode_sgncon   = get_(param_exp_pfgd1, 'mode_sgncon', 'sc');
  b_prime       = get_(param_exp_pfgd1, 'b_prime',     1);
  s_prime       = get_(param_exp_pfgd1, 's_prime',     1);

  % Parameters of Loss function
  typ_loss   = get_(param_exp_pfgd1, 'typ_loss', 'lr');
  gam_sm     = get_(param_exp_pfgd1, 'gam_sm', 0.01);

  % Parameters of FGD
  nepochs  = get_(param_exp_pfgd1, 'nepochs',  1000);
  lamn     = get_(param_exp_pfgd1, 'lamn',     1.0);
  ncvs_rnd = get_(param_exp_pfgd1, 'ncvs_rnd', 200);

  verbose = get_(param_exp_pfgd1, 'verbose', false);


  %
  % Logic
  %
  rng(rng_init);

  X_all = param_dataset.X_all;
  y_all = param_dataset.y_all;
  feanames_sel = param_dataset.feanames_sel;
  cvec_sgn = param_dataset.cvec_sgn;
  
  lmat_tra = h34_gen_lmat_tra( y_all, ntras, ncvs_rnd );
  
  [~,npts] = size( X_all ); 
  feanames_sel{end+1} = 'Bias'; 
  X_all    = [X_all; bias_factor.*ones(1,npts)]; 
  cvec_sgn(end+1) = 0; 
    
  % get kernel function
  param1_sckmtwo.typ_kern = typ_kern; 
  param1_sckmtwo.gam_rbf  = gam_rbf; 
  fh_kern                 = demo1320_get_kern( param1_sckmtwo ); 

  scos_cat = []; 
  ytst_cat = []; 
  for cv_rnd=1:ncvs_rnd
    rng(cv_rnd); 
    l_tra = lmat_tra(cv_rnd,:); l_tst = ~l_tra; % logical value
    X_tra = X_all(:,l_tra);     X_tst = X_all(:,l_tst); 
    y_tra = y_all(l_tra);       y_tst = y_all(l_tst); 

    ntras = numel(y_tra); 
    lam1  = lamn./ntras; 
    
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

    fh_lr = @(tea1)1./(lam1.*tea1); % Stepsize for gradient descent 
    param_pfgd1         = param_optim1; 
    param_pfgd1.fh_lr   = fh_lr; 
    param_pfgd1.nepochs = nepochs; 
    res1_pfgd = demo1344_train_pfgd( param_pfgd1 );
    sco1_sckm = res1_pfgd.sco_tst; 

    scos_cat  = [scos_cat;sco1_sckm]; 
    ytst_cat  = [ytst_cat;y_tst];
  end

end