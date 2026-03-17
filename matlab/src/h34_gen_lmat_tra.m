function lmat_tra = h34_gen_lmat_tra( y_all, ntras, ncvs_rnd )

  npts = numel(y_all); 
  tsassert( abs(y_all) == 1 ); 
  y_all(y_all==-1) = 2;
  npts_all = numel(y_all); 
  l_p       = y_all == 1;  
  nps_all   = sum(l_p);  
  nns_all   = npts - nps_all; 
  n2s = [nps_all,nns_all]; 
  ratio_tra = ntras./npts; 
  nps_ave   = nps_all.*ratio_tra; 
  nns_ave   = nns_all.*ratio_tra; 
  nps_floor = floor(nps_ave); 
  hasu1     = nps_ave- nps_floor; 
  n1mat = zeros(ncvs_rnd,2); 
  n1mat(:,1) = ( rand(1,ncvs_rnd) < hasu1 ) + nps_floor; 
  n1mat(:,2) = ntras - n1mat(:,1); 
  
  lmat_tra = false( ncvs_rnd, npts ); 
  for cv_rnd=1:ncvs_rnd
    for ct1=1:2
      r_ct = find( y_all' == ct1 ); 
      n1   = n1mat(cv_rnd,ct1); 
      n2   = n2s(ct1);
      r1   = randperm( n2, n1 ); 
      lmat_tra(cv_rnd,r_ct(r1)) = 1; 
    end
  end
