function demo1320_plot_accroc(output_dir, file_in)

load(file_in, 'dbname1', 'accmat_cv', ...
    'meths', 'ncvs_rnd', 'param_exp_pfgd1', ...
    'v_ntras' ); 

meths_all = {'sf-lin','sf-rbf','sc-lin','sc-rbf'};
stys_all = {'rs:','bv:','rs-','bv-'}; 
clrmat_all = [ 1, 0, 0; 0, 0, 1; 1, 0, 0; 0, 0, 1 ]; 

[~, idx] = ismember(meths_all, meths);
idx = idx(idx > 0);
meths = meths(idx);
[~, idx] = ismember(meths, meths_all);
stys = stys_all(idx);
clrmat = clrmat_all(idx,:);


names_old     = { ...
'sf-lin','sf-rbf',...
'sc-lin','sc-rbf' };  
names_new = {'Lin Kernel wo Sign Con', 'RBF Kernel wo Sign Con', ...
            'Lin Kernel w/ Sign Con', 'RBF Kernel w/ Sign Con' }; 
[meths_dsp,res] = rename_meths( meths, names_old, names_new );



nmeths = numel(meths); 
accmat1 = accmat_cv;
[gcf1,gca1] = tsfigure( [4.5,4] );
for i_meth=1:numel(meths)
    meth1 = meths{i_meth};
    plt1 = plot( log10(v_ntras), accmat1(i_meth,:), stys{i_meth} );
    set( plt1, 'Color', clrmat(i_meth,:) ); 
    if strcmp(meth1(1:2), 'sc'),
    set( plt1, 'MarkerFaceColor', clrmat(i_meth,:) ); 
    set( plt1, 'MarkerEdgeColor', 'none' ); 
    else
    set( plt1, 'MarkerFaceColor', [1,1,1] ); 
    set( plt1, 'MarkerEdgeColor', clrmat(i_meth,:) ); 
    end
end
set( gca1, 'XTick', log10(v_ntras) ); 
set( gca1, 'XTickLabel', v_ntras );
title(dbname1, 'Interpreter', 'none');
xlabel('# of training examples'); 
ylabel( 'Accuracy' ); 
legend( meths_dsp, 'Location', 'southeast' ); 

[~, filestem, ~] = fileparts(file_in);
filehint1 = sprintf('%s/%s.%s', output_dir, filestem, 'acc' ); 

%%% plot_eps_and png without printing
d_prt = '-depsc2'; fmt = 'eps'; 
file_out = sprintf('%s.%s',filehint1,fmt); 
print( d_prt, file_out ); 
d_prt = '-dpng'; fmt = 'png'; 
file_out = sprintf('%s.%s',filehint1,fmt); 
print( d_prt, file_out );
%%%
 