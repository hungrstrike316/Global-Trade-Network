function [gam] = PIF_dGauss5(x, s)

%
% syntax: [gam] = PIF_dGauss5(x, s);
%
% where x is Oj - Oi and 
% s (short for sigma) is standard deviation of the Gaussian
%
% This Phase Interaction Function is the third derivative of a Gaussian.
% It is rotationally symmetric.  It has 3 humps oneach side of 0 phase.  
% It has three stable and three unstable fixed points.

gam = -(1 ./ sqrt(2.*pi).*s.^11) .* exp( -x.^2 ./ (2.*s.^2) ) .* ...
            (x.^5 - 10.*x.^3.*s.^2 + 15.*x.*s.^4);