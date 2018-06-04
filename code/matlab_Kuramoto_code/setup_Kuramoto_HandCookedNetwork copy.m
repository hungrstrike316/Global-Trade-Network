%% Make Structures with parameters for network setup, run flags and Kuramoto Simulation.

kurflags_setup  % runflags from the kuramoto simulation code (what plots/movies to make, etc.)

kurParams_setup % parameters from kuramoto simulation (phase interaction function, time, runs, NF distribution, etc.)

netParams_setup_HCN % parameters from network/graph generation (Win, Wout, Rmax, C, etc.)

netflags.netflags = 'HandCookedNetwork';


                                            
                                            
%% add directory names into kurflags to save output data and images.
[kurflags]= kuramoto_output_dirs_setup_HCN(netParams, kurParams, netflags, kurflags);





%%

% % Tag plots with parameters used just for easier analysis
% paramsTag = {['PIF = ',PIFlg],['N = ',num2str(Ndims(1)),'x',num2str(Ndims(2)),' - C = ',num2str(C)],...
%     ['Rmax = ',num2str(Rmax),' - pfar = ',num2str(PconnFar)],['NF = ',num2str(muW),' , ',num2str(sigW)],['Win = ',num2str(Strng),...
%     ' , ', num2str(sigSg)],['Wout = ',num2str(Weak),' , ',num2str(sigWk)]};
% 
% 
% disp(['Parameters:: PIF ',PIFlg,' N',num2str(Ndims(1)),'x',num2str(Ndims(2)),' - C',num2str(C),...
%       ' - Rmax',num2str(Rmax),' - pfar = ',num2str(PconnFar),' - NF ',num2str(muW),' ',num2str(sigW),...
%       ' - Win ',num2str(Strng),' ', num2str(sigSg),' - Wout ',num2str(Weak),' ',num2str(sigWk)]);
% 
