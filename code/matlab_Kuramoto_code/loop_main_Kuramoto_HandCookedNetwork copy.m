function loop_main_Kuramoto_HandCookedNetwork(loop_Gnd_Truth, loop_weak)





%% Set up parameters I want to loop over - These I am setting in Sbatch file and feeding in as input to Function
%  I set them here just in case they are not input into the function.  They will be looped over within.
if ~exist('loop_Gnd_Truth','var')
                  
    % N = 48
    loop_Gnd_Truth = {{[ones(1,24), 2*ones(1,24)]}, ... % C = 2
                      {[ones(1,16), 2*ones(1,16), 3*ones(1,16)]}, ... % C = 3
                      {[ones(1,12), 2*ones(1,12), 3*ones(1,12), 4*ones(1,12)]}, ... % C = 4
                      {[ones(1,8), 2*ones(1,8), 3*ones(1,8), 4*ones(1,8), 5*ones(1,8), 6*ones(1,8)]}, ... % C = 6
                      {[ones(1,6), 2*ones(1,6), 3*ones(1,6), 4*ones(1,6), 5*ones(1,6), 6*ones(1,6), 7*ones(1,6), 8*ones(1,6)]} }; %, ... % C = 8
%                       {[ones(1,4), 2*ones(1,4), 3*ones(1,4), 4*ones(1,4), 5*ones(1,4), 6*ones(1,4), 7*ones(1,4), 8*ones(1,4), 9*ones(1,4), 10*ones(1,4), 11*ones(1,4), 12*ones(1,4)]}, ... % C = 12
%                       {[ones(1,3), 2*ones(1,3), 3*ones(1,3), 4*ones(1,3), 5*ones(1,3), 6*ones(1,3), 7*ones(1,3), 8*ones(1,3), 9*ones(1,3), 10*ones(1,3), 11*ones(1,3), 12*ones(1,3), 13*ones(1,3), 14*ones(1,3), 15*ones(1,3), 16*ones(1,3)]}, ... % C = 16
%                       {[ones(1,2), 2*ones(1,2), 3*ones(1,2), 4*ones(1,2), 5*ones(1,2), 6*ones(1,2), 7*ones(1,2), 8*ones(1,2), 9*ones(1,2), 10*ones(1,2), 11*ones(1,2), 12*ones(1,2), 13*ones(1,2), 14*ones(1,2), 15*ones(1,2), 16*ones(1,2), 17*ones(1,2), 18*ones(1,2), 19*ones(1,2), 20*ones(1,2), 21*ones(1,2), 22*ones(1,2), 23*ones(1,2), 24*ones(1,2)]} ... % C = 24
%                       };
                  

end
%
if ~exist('loop_weak','var')
    loop_weak = [-1:-1:-10]; %[-100 -50 -20 -10 -1 -0.1 0.1 1];
end





%% Stuff I am also looping over that I am not setting in sbatch file.              
kurParams.muW = 60;         % mean of oscillator resonant phase distribution (Hz) 
%
loop_strng = [1:10]; %[5 10 20 50 100];
loop_sigW = [0 0.005*kurParams.muW 0.010*kurParams.muW 0.015*kurParams.muW 0.020*kurParams.muW]; % ...
% loop_sigW = 0.02*kurParams.muW; %[0.11*kurParams.muW 0.13*kurParams.muW 0.15*kurParams.muW 0.17*kurParams.muW 0.19*kurParams.muW]; %[0.001*kurParams.muW 0.01*kurParams.muW 0.1*kurParams.muW];  
loop_Rmax = [1, 4, inf]; %[1 2 3 4 7 10 inf]; % inf
loop_Pconnfar = 0; %[0 0.05]; % Probability of a connection out beyond Rmax (set to zero for hard threshold at Rmax)




%% Flags for plotting (from inside Kuramoto - may not need all)
kurflags.PIFplt = 0;         % Plot the Phase Interaction Function between oscillators (in viz_PIF).
kurflags.coherence = 0;      % Plot radius of coherence for different combinations of oscillators (all, within clusters, across pairs of clusters)
kurflags.TwoPiXing = 0;      % Plot 2pi crossings of oscillators at beginning and end of simulation
kurflags.phasePrecess = 0;   % Plot phase precession of oscillators w.r.t. to the muW oscillation period and Separation Metric.


% Make Movies (really nice visualizations)
kurflags.OscPhaseImg_Mov = 0;  % A movie of oscillators alligning in phase as simulation evolves.

% Flag to save metaCluster data and to check if it is already there before running to save time.
kurflags.dataFileChk = 0;     % When running looped simulation, dont rerun of output data file exists already (useful when cluster crashes after running half of files)
kurflags.savemetaCluster = 1; % Save results of meta clustering analysis from each run.


% I could alternatively use kurflags_setup here.

%% Setting up a structure of parameters used on this simulation for saved data files  

% Unused (as of now) Parameters to build Coupling Matrix K1     
netParams.sigSg = 0; %0.001;    % Std of Coupling Weights within Clusters. 
netParams.sigWk = 0; %0.01;    % Std of Coupling Weights across Clusters.  
  

% Parameters to run Kuramoto simulation
kurParams.runs = 1;              % Number of times to run simulation (stochastic & yields different results each time) 
kurParams.Tsec = 1;              % User specifies time duration of simulation (in sec) - [[could also do in # oscillation periods]]
kurParams.spp = 10;              % User specifies # of time steps per period (mean period of muW)
kurParams.tau = 1/(kurParams.spp*kurParams.muW);     % Time step size (spp timesteps per period) not sure why 2* is necessary   
kurParams.T = kurParams.Tsec*kurParams.muW*kurParams.spp;                          % Total number of timesteps 

%
% Setting Up the Phase Interaction Function (many options)
kurParams.PIFlg = 'Fourier1';    % Options:  Fourier1, Fourier2, dGauss1, dGauss3, dGauss5, dGauss7
kurParams.PIFparams = [0];       % Look at each Phase Interaction Function to determine its Params
                                 % Params go into one vector in proper order 
%
% Numerical Integration Scheme
kurParams.numInt = 'Vanilla';    % Options: 'RK4' is not implemented yet.


saveEigMCdata = 1;  % if you wanna save a data structure containing Meta Cluster Results across all these different parameter settings. 
                    % Can combine the results later using MetaClusterVisualize
EigCntr = 0;        % counter for data vector that will contain Dominant Eigenvector segmentation stats.


netflags.netflags = 'HandCookedNetwork';






%% Loop over parameters and run Kuramoto simulation

for NN = 1:numel(loop_Pconnfar)

    netParams.PconnFar = loop_Pconnfar(NN);       % Probability of a connection beyond rmax (could make distance dependent too). 
    
    for MM = 1:numel(loop_Rmax)

        netParams.Rmax = loop_Rmax(MM);    % Distance each oscillator sees out from itself for connections
        
        for LL = 1:numel(loop_Gnd_Truth)  % Number of cluster options. 

            netParams.gndTruth = loop_Gnd_Truth{LL};  
            netParams.C = numel(unique(netParams.gndTruth{1}));       % Number of clusters.    
            netParams.Ndims = size(netParams.gndTruth{1});            % Number of oscillators in x, y (,and maybe z) dimension.
            netParams.N = prod(netParams.Ndims); % Number of oscillators is product of number in each dimension.                                
            
            for jj = 1:numel(loop_weak) % Mean Coupling weights across Clusters.

                netParams.Weak = loop_weak(jj); 

                for ii = 1:numel(loop_strng) % Mean Coupling Weights within Clusters.

                    netParams.Strng = loop_strng(ii);  
                    
                    
                    % segment using Eigenvector of coupling matrix
                    % [Note: Does not need to happen inside sigW loop.]
                    if(1) % Do Eigenvector
                        kurParams.sigW = 0;
                        
                        setup_Kuramoto_HandCookedNetwork
                        
                        % Build K here so that EvecSeg and KurSeg will have same Coupling Matrix.
                        if ~exist('K','var')
                            K = build_K2(netParams); % Build coupling matrix for hand crafted network using Win, Wout, C, Rmax, etc in kurParams.
                        end
                        
                        [Vdom] = main_EvecSeg(K);
                        
                        [MC] = metaClusterAnalysis(Vdom, netParams, kurParams, 0);
                     
                        

                        if(saveEigMCdata)
                        
                            EigCntr = EigCntr+1;
                            EigMCdata.Vdom(:,EigCntr) = Vdom;
                            EigMCdata.Win(EigCntr) = netParams.Strng;
                            EigMCdata.Wout(EigCntr) = netParams.Weak;
                            EigMCdata.N(EigCntr) = netParams.N;
                            EigMCdata.C(EigCntr) = netParams.C;
                            EigMCdata.Rmax(EigCntr) = netParams.Rmax;
                            EigMCdata.Pfar(EigCntr) = netParams.PconnFar;
                            EigMCdata.gndTruth(:,EigCntr) = netParams.gndTruth;
                            %
                            EigMCdata.phase.meanCSep(EigCntr) = MC.phase.meanCSep;
                            EigMCdata.phase.stdCSep(EigCntr) = MC.phase.stdCSep;
                            EigMCdata.phase.minCSep(EigCntr) = MC.phase.minCSep;
                            EigMCdata.phase.minCSepID(EigCntr,:) = MC.phase.minCSepID;
                            EigMCdata.phase.meanCDist(EigCntr) = MC.phase.meanCDist;
                            EigMCdata.phase.stdCDist(EigCntr) = MC.phase.stdCDist;
                            %
                            EigMCdata.location.meanCSep(EigCntr) = MC.location.meanCSep;
                            EigMCdata.location.stdCSep(EigCntr) = MC.location.stdCSep;
                            EigMCdata.location.minCSep(EigCntr) = MC.location.minCSep;
                            EigMCdata.location.minCSepID(EigCntr,:) = MC.location.minCSepID;
                            EigMCdata.location.meanCDist(EigCntr) = MC.location.meanCDist;
                            EigMCdata.location.stdCDist(EigCntr) = MC.location.stdCDist;
                            %
                            EigMCdata.meanCExt(EigCntr) = MC.meanCExt;
                            EigMCdata.stdCExt(EigCntr) = MC.stdCExt;
                            
                        end
                        
                    end
                    
                        


                    if(1) % Do Kuramoto

                        for kk = 1:numel(loop_sigW) % STD of oscillator resonant phase distribution (but all run thru abs function after)

                            kurParams.sigW = loop_sigW(kk); 

                            tic

                            disp([num2str(NN),'/',num2str(numel(loop_Pconnfar)),'   :   PconnFar = ',num2str(netParams.PconnFar)])
                            disp([num2str(MM),'/',num2str(numel(loop_Rmax)),'   :   Rmax = ',num2str(netParams.Rmax)])
                            disp([num2str(LL),'/',num2str(numel(loop_Gnd_Truth)),'   :   C = ',num2str(netParams.C)])
                            disp([num2str(jj),'/',num2str(numel(loop_weak)),'   :   Weak = ',num2str(netParams.Weak)])
                            disp([num2str(ii),'/',num2str(numel(loop_strng)),'   :   Strng = ',num2str(netParams.Strng)])
                            disp([num2str(kk),'/',num2str(numel(loop_sigW)),'   :   sigW = ',num2str(kurParams.sigW)])


                            % segment using Kuramoto Oscillator System Dynamics
                            setup_Kuramoto_HandCookedNetwork
                            
                            if exist('K','var')
                                main_Kuramoto(netParams, kurParams, netflags, kurflags, K);
                            else
                                main_Kuramoto(netParams, kurParams, netflags, kurflags);
                            end

                            toc

                        end % loop over sigW
                    
                    else
                        
                        disp([num2str(NN),'/',num2str(numel(loop_Pconnfar)),'   :   PconnFar = ',num2str(netParams.PconnFar)])
                        disp([num2str(MM),'/',num2str(numel(loop_Rmax)),'   :   Rmax = ',num2str(netParams.rmax)])
                        disp([num2str(LL),'/',num2str(numel(loop_Gnd_Truth)),'   :   C = ',num2str(netParams.C)])
                        disp([num2str(jj),'/',num2str(numel(loop_weak)),'   :   Weak = ',num2str(netParams.Weak)])
                        disp([num2str(ii),'/',num2str(numel(loop_strng)),'   :   Strng = ',num2str(netParams.Strng)])
                        
                    end
                    
                    
                end % loop over Win
            end % loop over Wout
        end % loop over Ground truth (clustering configurations)
    end % loop over Rmax
end % loop of Pfar


if(saveEigMCdata)
    dirPre = onCluster;
    save([dirPre,'output/Kuramoto/HandCookedNetwork/data/EigMC_ParamSearch_data'], 'EigMCdata') 
end