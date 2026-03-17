1; clear; close all; rng(0);

exp_name = 'exp_demo1334_scatter';
data_dir = 'dat';

ntras_fix = 20;
ncvs_rnd  = 200; 

gam_rbf = 0.001;
lamn    = 1.0;

b_prime = [1, -1];
s_prime = [1, 0];
gam_sm      = 0.01;
verbose     = false;
if strlength(exp_name) > 0
    exp_name = strcat('-', exp_name);
end
timestamp = string(datetime('now', 'TimeZone', 'local', 'Format', 'yyMMddHHmmss'));
output_exp_dir = strcat('out-demo1334/', timestamp, exp_name);
if ~isfolder(output_exp_dir)
    mkdir(output_exp_dir)
end
copyfile('demo1334.m', strcat(output_exp_dir, '/', 'demo1334.m'));
dbname1 = 'uci_aids_clinical_trials_group_study_175'; 
fprintf('Loading dataset: %s ...\n', dbname1);
param_dataset = demo1315_get_dataset(data_dir, dbname1, verbose);
X_all = param_dataset.X_all';
y_all = param_dataset.y_all;
cvec_sgn = param_dataset.cvec_sgn;
[npts, ~] = size(X_all);
bias_factor = 10.0;
X_all = [X_all, bias_factor * ones(npts, 1)];
cvec_sgn(end+1) = 0;
[~, nfeas_total] = size(X_all);
fprintf('Calculating SAC (corr * sign_constraint)...\n');
vec_sac = zeros(nfeas_total, 1);
for i = 1:nfeas_total
    if std(X_all(:, i)) > 1e-12
        vec_sac(i) = corr(X_all(:, i), y_all) * cvec_sgn(i);
    else
        vec_sac(i) = 0;
    end
end
param_exp.rng_init    = 0;
param_exp.data_dir    = data_dir;
param_exp.bias_factor = bias_factor;
param_exp.b_prime     = b_prime;
param_exp.s_prime     = s_prime;
param_exp.gam_sm      = gam_sm;
param_exp.ncvs_rnd    = ncvs_rnd;
param_exp.ntras       = ntras_fix;
param_exp.dbname      = dbname1;
param_exp.typ_loss    = 'lr';
param_exp.lamn        = lamn;
param_exp.gam_rbf     = gam_rbf;
meth1 = 'sc-rbf'; 
param_exp.mode_sgncon = meth1(1:2);
param_exp.typ_kern    = meth1(4:end);
[prob_active, ~] = demo1334_exp_pfgd(param_dataset, param_exp);
l_con = cvec_sgn(:) ~= 0;
if numel(prob_active) > numel(l_con)
    l_con(end+1:numel(prob_active)) = false;
end
indices_con = find(l_con);

probs_con = prob_active(indices_con);
corrs_con = vec_sac(indices_con);

[probs_sorted, sort_idx] = sort(probs_con, 'ascend');
corrs_sorted = corrs_con(sort_idx);

output_result_dir = sprintf('%s/%s', ...
    output_exp_dir, dbname1);
if ~isfolder(output_result_dir)
    mkdir(output_result_dir)
end

safe_dataset = sanitize_filename(dbname1);
safe_method = sanitize_filename(meth1);

mask_corr = ~isnan(corrs_sorted) & ~isnan(probs_sorted);
if sum(mask_corr) > 1
    r_corr = corr(corrs_sorted(mask_corr), probs_sorted(mask_corr));
else
    r_corr = NaN;
end

h_sc1 = figure('Name', sprintf('%s-%s-Scatter-SAC', dbname1, meth1), 'Visible', 'off');
set(h_sc1, 'Position', [100, 100, 500, 400]);
scatter(corrs_sorted, probs_sorted, 35, ...
    'MarkerFaceColor', [0.8392, 0.1529, 0.1569], ...
    'MarkerEdgeColor', 'none', ...
    'MarkerFaceAlpha', 0.7);
xlabel('SAC'); 
ylabel('Active Probability');
title(sprintf('%s (%s)', dbname1, meth1), 'FontWeight', 'bold', 'Interpreter','none');
ylim([-0.05, 1.05]);
grid on;
ax = gca;
ax.GridLineStyle = ':';
ax.GridAlpha = 0.6;

file_out_sc_corr = sprintf('%s/%s-%s-sac.scatter.png', ...
    output_result_dir, safe_dataset, safe_method);
saveas(h_sc1, file_out_sc_corr);
close(h_sc1);

fprintf('    Saved scatter png:\n      %s\n', file_out_sc_corr);

function out = sanitize_filename(name)
    out = regexprep(name, '[\(\)\s,\[\]\{\}]', '');
    out = strrep(out, '_', '-');
    out = regexprep(out, '-+', '-');
    out = regexprep(out, '^-|-$', '');
end
