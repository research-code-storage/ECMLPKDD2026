function tsassert_zro(A, epsi, varargin)
%
%
%
if nargin <= 1
  epsi = 1e-12;
end

max_abs = max(abs(A(:)));
if max_abs < epsi
  return;
end

st = dbstack(1, '-completenames');
if isempty(st)
  loc = 'unknown location';
else
  loc = sprintf('%s:%d', st(1).file, st(1).line);
end

if nargin >= 3
  detail = varargin{1};
  if isstring(detail)
    detail = char(detail);
  end
  if ~ischar(detail)
    detail = mat2str(detail);
  end
  error('tsassert_zro:failed', ...
    'Zero-assertion failed at %s\nmax(abs(A(:)))=%.6g, epsi=%.6g\n%s', ...
    loc, max_abs, epsi, detail);
else
  error('tsassert_zro:failed', ...
    'Zero-assertion failed at %s\nmax(abs(A(:)))=%.6g, epsi=%.6g', ...
    loc, max_abs, epsi);
end