function [gam] = pick_PIF(x, PIFlg, params)

%
% syntax: [gam] = pick_PIF(x, PIFlg, params)
%
% This function is just a clean and modular way to choose which Phase 
% Interaction Function to use in the Kuramoto simulation.  It will call one
% of the underlying functions.  It is called inside the Kuramoto.m function
%

switch PIFlg
    case 'Fourier1'
        gam = PIF_Fourier1(x, params(1));
    case 'Fourier2'
        gam = PIF_Fourier2(x, params(1), params(2));
    case 'dGauss1'
        gam = PIF_dGauss1(x, params(1));
    case 'dGauss3'
        gam = PIF_dGauss3(x, params(1));    
    case 'dGauss5'
        gam = PIF_dGauss5(x, params(1));        
    case 'dGauss7'
        gam = PIF_dGauss7(x, params(1));   
end