function [gam] = PIF_dGauss3(x, s)

%
% syntax: [gam] = PIF_dGauss3(x, s);
%
% where x is Oj - Oi and 
% s (short for sigma) is standard deviation of the Gaussian
%
% This Phase Interaction Function is the third derivative of a Gaussian.
% It is rotationally symmetric.  It has 2 humps on each side of 0 phase.  
% It has two stable and two unstable fixed points.
%
% Note: This is actually negative of derivative because I want 0 to be a
% stable fixed point.

gam = (1 ./ sqrt(2.*pi).*s.^7) .* exp( -x.^2 ./ (2.*s.^2) ) .* ...
                    x.*(x.^2 - 3.*s.^2);