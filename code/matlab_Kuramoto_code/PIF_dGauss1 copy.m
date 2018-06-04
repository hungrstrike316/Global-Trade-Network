function [gam] = PIF_dGauss1(x, s)

%
% syntax: [gam] = PIF_dGauss1(x, s);
%
% where x is Oj - Oi and 
% s (short for sigma) is standard deviation of the Gaussian
%
% This Phase Interaction Function is the first derivative of a Gaussian.
% It is rotationally symmetric.  It is kinda topologically the same as a
% sine wave, only it can be made to be more concentrated in phase (closer
% to 0 phase by increasing the s (sigma) parameter. It has one stable and
% one unstable fixed point.

gam = -(1 ./ sqrt(2.*pi).*s.^3) .* exp( -x.^2 ./ (2.*s.^2) ) .* (x);