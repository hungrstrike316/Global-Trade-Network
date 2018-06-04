function [w] = build_w1(netparams, kurparams)


% Distribution of Resonant Frequencies for All Oscillators. (Gaussian)
% Centered about zero. Unimodal with infinite tail.   

%% Unpack variable values from parameter & flag structures
muW = kurparams.muW;
sigW = kurparams.sigW;
N = netparams.N;


%% Randomly choose natural frequencies - constrained to be positive.
w = random('Normal',muW,sigW,1,N); % Random natural frequencies


% Assume: Natural frequencies need to be positive - (represent spike rate kinda)
w = abs(w);
% Note: This disturbs shape of distribution. Can think of other ways to 
%       constrain natural frequencies to be positive.

% ToDo: Make natural frequencies based on similarity correlated with 
% coupling K (both derived from pixel values) - maybe not.