1; clear; close all; 

exp_name = 'exp_demo1336';
data_dir = 'dat';

v_ntras = unique([ceil(10.^(0.7:0.1:2.3))]); 

gam_rbf = 0.001; 
b_prime = [1, -1];
s_prime = [1, 0];
ncvs_rnd    = 200; 
verbose     = false;
n_ntras = numel(v_ntras);
if strlength(exp_name) > 0
    exp_name = strcat('-', exp_name);
end
timestamp = string(datetime('now','TimeZone','local','Format','yyMMddHHmmss'));
output_exp_dir = strcat('out-demo1336/', timestamp, exp_name );
if not(isfolder(output_exp_dir))
    mkdir(output_exp_dir)
end
copyfile('demo1336.m', strcat(output_exp_dir, '/', 'demo1336.m'));
param_static.rng_init    = 0;
param_static.data_dir    = data_dir;
param_static.b_prime     = b_prime;
param_static.s_prime     = s_prime;
param_static.ncvs_rnd    = ncvs_rnd;
param_static.gam_rbf     = gam_rbf;
param_static.typ_loss    = 'lr';

dbname1 = 'uci_statlog_(german_credit_data)';
dbname1_title = dbname1;
if startsWith(dbname1_title, 'uci_')
    dbname1_title = extractAfter(dbname1_title, 4);
end
dbname1_title = strrep(dbname1_title, '_', ' ');
fprintf('Dataset: %s\n', dbname1);

param_dataset = demo1315_get_dataset(data_dir, dbname1 , verbose); 
cvec_sgn = param_dataset.cvec_sgn;
feanames = param_dataset.feanames_sel;
X_all    = param_dataset.X_all; 
y_all    = param_dataset.y_all; 

[nfeas, npts] = size(X_all);
vec_corr      = zeros(nfeas, 1);

for i = 1:nfeas
    feat_vec = X_all(i, :)'; 
    if std(feat_vec) > 1e-12
        vec_corr(i) = corr(feat_vec, y_all);
    else
        vec_corr(i) = 0;
    end
end

feanames{end+1} = 'Bias';
cvec_sgn(end+1) = 0;
vec_corr(end+1) = 0; 

l_con = cvec_sgn(:) ~= 0;
idx_con = find(l_con);
sacs_con = vec_corr(idx_con).*cvec_sgn(idx_con);

[~, sort_order] = sort(sacs_con, 'descend');
idx_con_sorted = idx_con(sort_order); 
n_con = numel(idx_con_sorted);

row_labels = cell(n_con, 1);
for k=1:n_con
    idx = idx_con_sorted(k);
    sgn_mark = '+';
    if cvec_sgn(idx) < 0, sgn_mark = '-'; end
    row_labels{k} = sprintf('%s (%s)', feanames{idx}, sgn_mark);
end

param_exp = param_static;
param_exp.dbname = dbname1;
meth1 = 'sc-rbf';   
mode_sgncon = meth1(1:2);
typ_kern    = meth1(4:end);
param_exp.mode_sgncon = mode_sgncon;
param_exp.typ_kern    = typ_kern;

heatmap_data = zeros(n_con, n_ntras);

for i_n=n_ntras:-1:1
    ntras = v_ntras(i_n);
    param_exp.ntras = ntras;
    
    [prob_active_vec, ~] = demo1334_exp_pfgd(param_dataset, param_exp);
    
    if numel(prob_active_vec) >= max(idx_con_sorted)
        heatmap_data(:, i_n) = prob_active_vec(idx_con_sorted);
    else
        heatmap_data(:, i_n) = NaN;
    end
    
    fprintf('.');
end
fprintf(' Done.\n');

% --- Plotting Heatmap ---
h_fig = figure('Name', sprintf('Heatmap %s - %s', dbname1_title, meth1), 'NumberTitle', 'off');
set(h_fig, 'Position', [100, 100, 1200, 600]); 

col_labels = string(v_ntras);

h = heatmap(col_labels, row_labels, heatmap_data);

h.Title = sprintf('Active Probability Heatmap [%s]', dbname1_title);
h.XLabel = 'Number of Training Samples';
h.YLabel = 'Constrained Features (SAC Descending)';

m = 256;
R_map = linspace(0, 1, m)';
G_map = zeros(m, 1); 
B_map = linspace(1, 0, m)';
custom_map = [R_map, G_map, B_map];

h.Colormap = custom_map;
h.ColorLimits = [0, 1]; 
h.CellLabelFormat = '%.2f'; 
h.GridVisible = 'off';

output_result_dir = sprintf('%s/%s', output_exp_dir, dbname1);
if not(isfolder(output_result_dir))
    mkdir(output_result_dir)
end

file_png = sprintf('%s/%s_%s_heatmap.png', output_result_dir, dbname1, meth1);
saveas(h_fig, file_png);
fprintf('Graph saved to: %s\n', file_png);
close all; 
