function [ret,res] = demo1321_get_acc( ghat_tst, y_tst, mode )

ghat_tst = full(ghat_tst(:));
y_tst = full(y_tst(:));
res   = []; 

if ( strcmp( mode, 'acc' ) )

  ntsts = sum( y_tst ~= 0 ); 
  ret   = sum(ghat_tst.*y_tst > 0) / ntsts; 

else
  tsassert(0); 
end