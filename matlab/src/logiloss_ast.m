function ret = logiloss_ast(a1)
%
%
%
l_lt = a1 < -1; 
l_lp = a1 == -1; 
l_in = -1 < a1 & a1 < 0; 
l_rp = a1 == 0; 
l_rr = a1 > 0; 

a1_in = a1(l_in); 
term1 = -a1_in.*log(-a1_in); 
term2 = (1+a1_in).*log(1+a1_in); 
ret   = zeros(size(a1)); 
ret(l_in) = term1 + term2; 
ret(l_lt|l_rr) = inf; 
ret(l_lp|l_rp) = 0.0; 
