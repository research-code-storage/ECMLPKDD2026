function [xvec1_star,res1] = demo1334_solve( cd, avec1, sigvec1, param1_proj )
%
%
%

ncons = numel(avec1); 
tsassert( size(cd) == [1,1] ); 
tsassert( size(avec1) == [ncons,1] ); 
tsassert( size(sigvec1) == [ncons,1] ); 
tsassert_zro( abs(sigvec1) - 1.0 )
tsassert( issorted(avec1) ); 

advec1 = zeros(ncons+1,1); 
advec1(1) = avec1(1)-1.0; 
advec1(end) = avec1(end)+1.0; 
advec1(2:end-1) = 0.5.*(avec1(1:end-1)+avec1(2:end)); 

[~,grad1s_ep,~] = demo1291_objd( avec1', cd, avec1, sigvec1 ); 
l_neggrad1 = grad1s_ep <= 0; 
j1 = find(l_neggrad1);
if ( numel(j1) == 0 )
    j1 = ncons + 1; 
else
    j1 = j1(1); 
end
l1 = (avec1 - advec1(j1)).*sigvec1 >= 0; 
tea_star = dot(avec1,l1)./(sum(l1)+1.0./cd); 
[objd1_star,grad1_star,xrvec1_star] = demo1291_objd( tea_star, cd, avec1, sigvec1 ); 
xvec1_star = xrvec1_star(1:ncons); 
res1.objd1_star = objd1_star; 
res1.grad1_star = grad1_star; 
res1.tea_star = tea_star; 

if strcmp( param1_proj.qpmeth1, 'quadprog' )
    H_qp  = eye(ncons) + cd; 
    f_qp  = -avec1; 
    lb_qp = -100.*(sigvec1 == -1); 
    ub_qp = +100.*(sigvec1 == +1);
    opts_qp = optimoptions('quadprog', 'Display', 'off');
    xvec1_qp = quadprog(H_qp,f_qp,[],[],[],[],lb_qp,ub_qp,[],opts_qp); 
    tsassert_zro( norm(xvec1_star - xvec1_qp), 1e-3.*ncons ); 
end

