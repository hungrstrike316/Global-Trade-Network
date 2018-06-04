%% Make Structures with parameters for network setup, run flags and Kuramoto Simulation.

netflags_setup_IMG
netParams_setup_IMG


kurflags_setup  % runflags from the kuramoto simulation code (what plots/movies to make, etc.)
kurParams_setup % parameters from kuramoto simulation (phase interaction function, time, runs, NF distribution, etc.)







%% Build vector of Oscillator Natural Frequencies
w = build_w1(netParams,kurParams);          % note: maybe can also make w a function of the input image.
                                            % maybe move this inside run loop to change natural frequency on each run.



                                            
%% add directory names into kurflags to save output data and images.
[kurflags] = kuramoto_output_dirs_setup_IMG(netParams, kurParams, netflags, kurflags);





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
