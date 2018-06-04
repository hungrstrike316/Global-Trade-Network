function [gam] = PIF_Fourier2(x, a, R)

% syntax: [gam] = PIF_Fourier2(x, a, R);
%
% The function takes in phases of two oscillators and calculates the
% interaction between them using an extension of the phase interaction 
% function in the used in the Original Kuramoto Model.  The original used
% only one fourier component (sin(x)) and only generated synchronizing 
% forces. This one uses 2 fourier components (sin(x) & sin(2x)) and the 2nd
% component provides a desynchronizing force to balance the synchronizing
% force of the first term.
%
% Gamma_{ij}(x) = sin(x + a) - R*sin(2x)  ... where x = Oj - Oi.
%
% a is preferred phase offset between oscillators and between 2 fourier
% modes. R is the relative strength of the 1st synchronizing mode vs the
% 2nd desynchronizing mode.  If R = 0, this becomes the PIF_Fourier1.
%

if ~exist('a','var')
    a=0;
end

if ~exist('R','var')
    R=0;
end

gam = sin(x + a) - R*sin(2*x);