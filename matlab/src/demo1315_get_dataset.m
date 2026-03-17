function param_dataset = demo1315_get_dataset( data_dir, dbname1, verbose )
  
file_in = strcat(data_dir, '/', dbname1, '.mat');
load(file_in, 'X_dat', 'y_dat', 'cvec_sgn') ;
if verbose
  fprintf('%s loaded.\n',file_in);
end
X_all = X_dat';
[nfeas,npts] = size( X_all ); 
y_all = y_dat';
feanames_sel = {};
for i=1:nfeas
  feanames_sel{i} = num2str(i);
end
cvec_sgn = cvec_sgn';

[nfeas,npts] = size( X_all ); 
tsassert( size(y_all) == [npts,1] );

if verbose
  n_pos = sum(y_all > 0);
  n_neg = sum(y_all < 0);
  n_zero = sum(y_all == 0);
  n_all = numel(y_all);
  fprintf('y_all balance: pos=%d (%.2f%%), neg=%d (%.2f%%), zero=%d (%.2f%%)\n', ...
    n_pos, 100*n_pos/n_all, n_neg, 100*n_neg/n_all, n_zero, 100*n_zero/n_all);
end

tsassert( size(cvec_sgn) == [nfeas,1] ); 
tsassert( numel(feanames_sel) == nfeas ); 

param_dataset.X_all         = X_all;
param_dataset.y_all         = y_all;
param_dataset.feanames_sel  = feanames_sel; 
param_dataset.cvec_sgn      = cvec_sgn;