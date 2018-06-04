%% Flags & Input Parameters For Kuramoto Simulation
%  What plots and movies to produce.

% Flags for plotting (from inside Kuramoto - may not need all)
if ~exist('kurflags','var')
    
    kurflags.PIFplt = 1;         % Plot the Phase Interaction Function between oscillators (in viz_PIF).
    % kurflags.TwoPiXing = 0;      % Plot 2pi crossings of oscillators at beginning and end of simulation
    
    kurflags.phasePrecess = 1;   % Plot phase precession of oscillators w.r.t. to the muW oscillation period and Separation Metric.


    % Make Movies (really nice visualizations)
    kurflags.OscPhaseImg_Mov = 0;  % A movie of oscillators alligning in phase as simulation evolves.

    
    % kurflags.dataFileChk = 1;     % COMMENTED OUT BECAUSE BEING DONE IN LOOP_IMGSEG CODE.
    kurflags.savemetaCluster = 0; %1; % Save results of meta clustering analysis mat file from each run.

else
    
    %disp('Not creating: kurflags already exists.')
    %kurflags
    
end