function kernmat1 = ts_rbfkern( X_a, X_b, gam1 )

powdmat1 = get_powdist( X_a, X_b ); 
kernmat1 = exp(-gam1.*powdmat1); 
