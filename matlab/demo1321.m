1; clear; close all; 

ncvs_rnd = 200;
exp_name = 'exp_demo1321';
data_dir = 'dat';
meths     = { ...
'sf-lin','sf-rbf',... 
'sc-lin','sc-rbf'};
gam_rbf = 0.001; 
lamn = 1.0;
rng_init = 0;
bias_factor = 10.0;
b_prime     = [1,-1];
s_prime     = [1,0];
typ_loss    = 'lr';  
gam_sm      = 0.01;
nepochs = 1000;
verbose = false;
nmeths    = numel(meths);
v_ntras = unique([5:9,ceil(10.^(1.0:0.1:2.0))]); 
if strlength(exp_name) > 0
    exp_name = strcat('-', exp_name);
end
timestamp = string(datetime('now','TimeZone','local','Format','yyMMddHHmmss'));
output_exp_dir = strcat('out-demo1321/', timestamp, exp_name );
if not(isfolder(output_exp_dir))
    mkdir(output_exp_dir)
end
copyfile('demo1321.m', strcat(output_exp_dir, '/', 'demo1321.m'));

param_static.rng_init    = rng_init;
param_static.data_dir    = data_dir;
param_static.bias_factor = bias_factor;
param_static.b_prime     = b_prime;
param_static.s_prime     = s_prime;
param_static.typ_loss    = typ_loss;
param_static.gam_sm      = gam_sm;
param_static.nepochs     = nepochs;
param_static.ncvs_rnd    = ncvs_rnd;
param_static.verbose     = verbose;
param_exp_pfgd1         = param_static;
dbname1                 = 'uci_adult'; 
param_exp_pfgd1.dbname  = dbname1;
param_exp_pfgd1.lamn    = lamn;
param_exp_pfgd1.gam_rbf = gam_rbf;
param_dataset = demo1315_get_dataset(data_dir, dbname1 , verbose); 
cache_lin_metrics = cell(nmeths, numel(v_ntras));
cache_rbf_metrics = cell(nmeths, numel(v_ntras));
accmat_cv = zeros(nmeths,numel(v_ntras)); 

for i_ntras=1:numel(v_ntras)
    ntras = v_ntras(i_ntras); 
    param_exp_pfgd1.ntras = ntras;
    fprintf('.'); 

    for i_meth=1:numel(meths)

        meth1 = meths{i_meth}; 
        tsassert( meth1(3) == '-' ); 
        mode_sgncon = meth1(1:2);
        typ_kern    = meth1(4:end);
        param_exp_pfgd1.mode_sgncon = mode_sgncon;
        param_exp_pfgd1.typ_kern    = typ_kern;

        is_lin = contains(typ_kern, 'lin');
        is_rbf = strcmp(typ_kern, 'rbf');

        cached_vals = [];
        if is_lin && ~isempty(cache_lin_metrics{i_meth, i_ntras})
            cached_vals = cache_lin_metrics{i_meth, i_ntras};
        elseif is_rbf && ~isempty(cache_rbf_metrics{i_meth, i_ntras})
            cached_vals = cache_rbf_metrics{i_meth, i_ntras};
        end

        if ~isempty(cached_vals)
            accmat_cv(i_meth,i_ntras,:) = cached_vals;
        else
            [scos_cat, ytst_cat] = demo1321_exp_pfgd(param_dataset, param_exp_pfgd1);
            current_metrics = demo1321_get_acc( scos_cat, ytst_cat, 'acc' ); 

            accmat_cv(i_meth,i_ntras,:) = current_metrics;

            if is_lin
                cache_lin_metrics{i_meth, i_ntras} = current_metrics;
            elseif is_rbf
                cache_rbf_metrics{i_meth, i_ntras} = current_metrics;
            end
        end
    end
end
fprintf('\n'); 

output_result_dir = sprintf('%s/%s', output_exp_dir, dbname1); 
if not(isfolder(output_result_dir))
    mkdir(output_result_dir)
end

file_out = sprintf('%s/%s.mat', output_result_dir, dbname1); 
save( file_out, 'dbname1', 'accmat_cv', ...
'meths', 'ncvs_rnd', 'param_exp_pfgd1', ...
'v_ntras' ); 
if param_exp_pfgd1.verbose
    fprintf('%s written.\n',file_out); 
end

demo1320_plot_accroc(output_result_dir, file_out); 
close all; 

 