function fh_kern = demo1320_get_kern( param1_kern )

typ1_kern = param1_kern.typ_kern; 
switch typ1_kern
case 'lin'
    fh_kern   = @(X_tra,X_tst)X_tra'*X_tst; 
case 'rbf'
    gam1_rbf  = param1_kern.gam_rbf; 
    fh_kern   = @(X_tra,X_tst)ts_rbfkern(X_tra,X_tst,gam1_rbf); 
otherwise
    tsassert(0);
end   


