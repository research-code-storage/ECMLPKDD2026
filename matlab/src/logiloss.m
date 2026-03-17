function ret = logiloss(z)
%
% Logistic loss. 
%

if 0,
  ret  = log(1+exp(-z));
end
l_p = z >= 0; l_n = ~l_p; 
z_p = z(l_p); z_n = z(l_n); 
term_p = log(1+exp(-z_p)); 
term_n = -z_n + log(1+exp(z_n)); 
ret = zeros(size(z)); 
ret(l_p) = term_p; 
ret(l_n) = term_n; 
