function [meths_dst,res] = rename_meths( meths_src, names_old, names_new )
%
%
%
tsassert( numel( names_old ) == numel( names_new ) ); 
nmeths    = numel( meths_src ); 
meths_dst = meths_src; 
l_renamed = false(1,nmeths); 
for i_meth=1:nmeths
  l1 = false; 
  for i1=1:numel( names_old )
    name_old = names_old{i1}; 
    name_new = names_new{i1}; 
    if strcmp( meths_dst{i_meth}, name_old )
      meths_dst{i_meth} = name_new; 
      l_renamed(i_meth) = 1; 
    end
  end
end

res.l_renamed = l_renamed; 
