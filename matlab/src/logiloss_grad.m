function ret = logiloss_grad(z)
%
% Logistic loss. 
%

l_p = z >= 0; l_n = ~l_p; 
z_p = z(l_p); z_n = z(l_n); 
tmp1 = exp(-z_p); 
grad_p = - tmp1./(tmp1+1); 
grad_n = - 1./(exp(z_n)+1); 
ret = zeros(size(z)); 
ret(l_p) = grad_p; 
ret(l_n) = grad_n; 
