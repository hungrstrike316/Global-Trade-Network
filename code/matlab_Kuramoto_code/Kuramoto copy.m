function [theta] = Kuramoto(K, w, netParams, kurParams, kurflags, saveDir) 


% (K, w, tau, T, PIFlg, PIFparams, dynMov, PIFplt, saveDir) % ,theta_unwr

% syntax: Kuramoto(K, w, tau, T, B, R, dynMov);
%
% Input:
%
%   K = NxN matrix of couplings where N is number of oscillators in
%      network. Couplings are heterogeneous and not necessarily all-to-all
%      (K can have zero entries).  Also, there are no self-couplings so
%      diagonal of K is zero.
%
%   w = N-element vector of the natural frequencies of each oscillator
%
%   tau = time step (tau -> 0 approaches simulation of continuous time)
%
%   T = total number of timesteps in simulation
%
%   B = de-phasing or phase offset between synchronizing 1st order
%       interaction and 2nd order desynchronizing interaction.
%
%   R = Relative weighting between synchronizing 1st order and
%       desynchronizing 2nd order interactions.
%
%   dynMov = flag to make and save a movie of oscillators rotating around
%      unit circle.
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function is taken from Munkhammar and editted.
%
% Simulation of Kuramoto's model (Chapter 6, Box A)
% Written by Joakim Munkhammar 
%
% http://www.collective-behavior.com/Site/Synchronization.html
% from: http://www.collective-behavior.com/Simulations/Ch6BoxA.m

% % Flags for plotting
% resFreq = 0;
% phaseDist = 1;
% coherence = 0;
% dynMov = 0;

%% Unpack variable values from parameter & flag structures.
% N = params.N;
% T = params.T;
% tau = params.tau;
% PIFlg = params.PIFlg;
% PIFparams = params.PIFparams;
% unitCircMov = flags.unitCircMov;
% PIFplt = flags.PIFplt;


% Extract data from kurflags
vars = fieldnames(kurflags);
for i=1:numel(vars)
    eval([vars{i},'=kurflags.',vars{i},';'])
end

% Extract data from kurParams
vars = fieldnames(kurParams);
for i=1:numel(vars)
    eval([vars{i},'=kurParams.',vars{i},';'])
end

% Extract data from netParams
vars = fieldnames(netParams);
for i=1:numel(vars)
    eval([vars{i},'=netParams.',vars{i},';'])
end




% %% Preallocate arrays for data and movie structure.
% if(unitCircMov)
%     writerObj = VideoWriter([saveDir,'K-phase_',KurParamsTag(1:end-6),'.avi']);
%     writerObj.FrameRate = 2;
%     open(writerObj);
%     
%     colors = 'kbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmy';
%     
% end


%% Compute Normalization for the Phase Interaction Function
x = linspace(-pi, pi, 100);            
gam = pick_PIF(x, PIFlg, PIFparams);            % choose PIF here.
normalization = max(abs(gam));                  % compute normalization 
gam = gam ./normalization;                      % normalize PIF

% if(PIFplt)
%     viz_PIF(x, gam, PIFlg, PIFparams, saveDir); % visualize PIF.
% end

%% Need to create a separate distribution for initial conditions and for resonant frequency

% Preallocate mamory for theta data
theta = zeros(T,N);

% Initial conditions for theta
phase_init

M = sum(K~=0); % Number of "Neighbors" to a node.  Number of interactions. Row Sums.

tic

for t=2:T % Loop through timesteps

    % "Matrixify" (tyler) this shit.
    DT = repmat(theta(t-1,:)',1,N) - repmat(theta(t-1,:),N,1);   % Phase Difference between pairs of oscillators
    gam = pick_PIF(DT, PIFlg, PIFparams);                        % Note: I am not normalizing gamma anymore i dont think.
    gam = gam ./ normalization;                                  % Normalized PIF
    theta(t,:) = theta(t-1,:) + 2*pi*tau*( w + sum(K.*gam)./M ); % Update phase of oscillator.


%     % Create movie of polar phase plot of oscillators
%     if(unitCircMov)
%         pp = figure; hold on
%         for i = 1:numel(unique(gndTruth{1}))
%             ind = find(gndTruth{1} == i);
%             p = polar(theta(t,ind),ones(1,numel(ind)),[colors(i),'.']);
%         end
%         
%         axis square
%         set(gca,'XtickLabel',[],'YTickLabel',[])
%         delete(findall(ancestor(p,'figure'),'HandleVisibility','off','type','line','-or','type','text')); 
%         set(gca,'nextplot','replacechildren');
%         title(['Oscillator Phase (t = ',num2str(t),'/',num2str(T),')'],'FontSize',20,'FontWeight','Bold');
%         mov = getframe(pp);
%         writeVideo(writerObj,mov);
%         if(mod(t,20))
%            close all % to keep figures from overloading memory of compu.
%         end
%     end

end

theta = mod(theta, 2*pi); 

toc
    

% %% Create AVI movie of phase oscillators around circle
% if(unitCircMov)
%     close(writerObj);
% end