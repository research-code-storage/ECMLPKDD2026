%
%
%
function tsassert(x, varargin)

if all(vec(x))
  return;
end

st = dbstack(1, '-completenames');
if isempty(st)
  loc = 'unknown location';
else
  loc = sprintf('%s:%d', st(1).file, st(1).line);
end

if nargin >= 2
  detail = varargin{1};
  if isstring(detail)
    detail = char(detail);
  end
  if ~ischar(detail)
    detail = mat2str(detail);
  end
  error('tsassert:failed', 'Assertion failed at %s\n%s', loc, detail);
else
  error('tsassert:failed', 'Assertion failed at %s', loc);
end
end




