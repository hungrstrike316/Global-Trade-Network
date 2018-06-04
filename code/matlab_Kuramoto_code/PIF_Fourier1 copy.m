function [gam] = PIF_Fourier1(x, a)

% syntax: [gam] = PIF_Fourier1(x, a);
%
% The function takes in phases of two oscillators and calculates the
% interaction between them using the phase interaction function in the
% Original Kuramoto Model (true when a = 0).  a is the phase shift imposed 
% by this PIF. If not specified, it is 0.
%
% Gamma_{ij}(x) = sin(x) ... where x = Oj - Oi.
%

if ~exist('a','var')
    a=0;
end

gam = sin(x + a);