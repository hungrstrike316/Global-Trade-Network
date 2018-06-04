%% Structure of parameters used on Kuramoto simulation.
     
if ~exist('kurParams','var')
    
    % Parameters to build vector of natural frequency of oscillators
    kurParams.muW = 60;                  % mean of oscillator resonant phase distribution (Hz)   
    kurParams.sigW = (1/30)*kurParams.muW; % STD of oscillator resonant phase distribution (but all run thru abs function after)
    kurParams.Kscale = 100;
    %
    % Parameters to run Kuramoto simulation
    % kurParams.runs = 1;                                      % Number of times to run simulation (stochastic & yields different results each time) 
    kurParams.Tsec = 1/2;                                      % User specifies time duration of simulation (in sec) - [[could also do in # oscillation periods]]
    kurParams.spp = 10;                                         % User specifies # of time steps per period (mean period of muW)
    kurParams.tau = 1/(kurParams.spp*kurParams.muW);           % Time step size (spp timesteps per period)    
    kurParams.T = kurParams.Tsec*kurParams.muW*kurParams.spp;  % Total number of timesteps 

    %
    % Setting Up the Phase Interaction Function (many options)
    kurParams.PIFlg = 'Fourier1';    % Options:  Fourier1, Fourier2, dGauss1, dGauss3, dGauss5, dGauss7
    kurParams.PIFparams = [0];       % Look at each Phase Interaction Function to determine its Params
                                     % Params go into one vector in proper order 
    %
    % Numerical Integration Scheme
    kurParams.numInt = 'Vanilla';    % Options: 'RK4' is not implemented yet.
    
else
    
    %disp('Not creating: kurParams already exists.')
    %kurParams

end