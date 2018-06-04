function [theta] = KuramotoB(K, w, netParams, kurParams, kurflags, saveDir) 


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








compareBatch2Full = 0; % flag to compare solving for theta in batches to solving all at once in one big NxN matrix.

batchSize = 1000; % ceil(N/2); % number of oscillators to calculate theta for at one time (necessary for large matrices)

disp(['Running inside KuramotoB Simulation for ',num2str(N),' oscillators in batches of ',num2str(batchSize),'.'])






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
x = linspace(0, 2*pi, 100);            
gamx = pick_PIF(x, PIFlg, PIFparams);            % choose PIF here.
normalization = max(abs(gamx));                  % compute normalization 
gamx = gamx ./normalization;                     % normalize PIF

% if(PIFplt)
%     viz_PIF(x, gam, PIFlg, PIFparams, saveDir); % visualize PIF.
% end

%% Need to create a separate distribution for initial conditions and for resonant frequency

% Preallocate mamory for theta data
theta = zeros(T,N);


% Initial conditions for theta
phase_init

M = sum(K~=0); % Number of "Neighbors" to a node.  Number of interactions. Row Sums.

if(compareBatch2Full)
    thetaB = theta;
end

tic

for t=2:T % Loop through timesteps
    
    disp(['Simulation Time Step: ',num2str(t),' / ',num2str(T)])

    % "Matrixify" (tyler lee) this shit.
    if(compareBatch2Full)
        DTB = repmat(thetaB(t-1,:)',1,N) - repmat(thetaB(t-1,:),N,1);   % Phase Difference between pairs of oscillators
        gamB = pick_PIF(DTB, PIFlg, PIFparams);                         % Note: I am not normalizing gamma anymore i dont think.
        gamB = gamB ./ normalization;                                   % Normalized PIF
        thetaB(t,:) = thetaB(t-1,:) + 2*pi*tau*( w + sum(K.*gamB)./M ); % Update phase of oscillator.
    end
    
    
    % Unmatrixify because matrices can not handle large image patches.
    % Loop through each oscillator (for loop) and calculate distance to all other oscillators (vector)
    % I cant calculate DB for entire NxN matrix at same time but can do for (N)x(procBatchSize) I think.
    
    batchBeg = [1:batchSize:N];
    batchEnd = [batchSize:batchSize:N];
    if ( numel(batchEnd) < numel(batchBeg) )
        batchEnd = [batchEnd,N];
    end
    %
    disp(['Batch # out of ',num2str(numel(batchBeg))])
    for i = 1:numel(batchBeg)
        
        fprintf('%s',[num2str(i),' '])
        
        st = batchBeg(i);
        nd = batchEnd(i);
        num = nd-st+1;
        
        DT = repmat(theta(t-1,:)',1,num) - repmat(theta(t-1,st:nd),N,1);  % Phase Difference between pairs of oscillators
        gam = pick_PIF(DT, PIFlg, PIFparams);                             % Note: I am not normalizing gamma anymore i dont think.
        gam = gam ./ normalization;                                       % Normalized PIF
        theta(t,st:nd) = theta(t-1,st:nd) + 2*pi*tau*( w(st:nd) + sum(K(:,st:nd).*gam)./M(st:nd) ); % Update phase of oscillator  

    end
    fprintf('\n')
    
    
    if ( 0 & ~mod(t-1,100) )
        
       x = repmat([1:N],t-1,1);
       y = mod(theta(1:(t-1),:),2*pi);
        
       figure, ylim([0,2*pi]), hold on,
       scatter(x(:),y(:)), 
       scatter([1:N],mod(theta(t-1,:),2*pi),'rx')
       keyboard
        
    end
    
    
    % some plots to be sure that computing thetaB in batches is the same as
    % computing theta in one big chunk.  Can only do for smaller image
    % patches. It looks fine as of 2/10/15. CW.
    if(compareBatch2Full)
        figure, 
        subplot(3,3,1),imagesc(DTB(:,st:nd)-DT), title('Diff')
        subplot(3,3,2),imagesc(DTB(:,st:nd)), title('DT')
        subplot(3,3,3),imagesc(DT), title('DTBatch')
        subplot(3,3,[4:6]),hold on, plot(theta(t,:)),  plot(thetaB(t,:),'r'),title(['t = ',num2str(t)]),legend(['batches',num2str(batchSize)],'whole')
        subplot(3,3,[7:9]),hold on, plot(theta(t,:) - thetaB(t,:),'g'),title('Phase Difference')

        keyboard
    end


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

% Error checking.. Plot each oscillator's phase as a function of simulation time (NICE PLOT!)
if(0)
    figure, imagesc(theta'), zlim([0, 2*pi]), colormap('hsv'),colorbar
    xlabel('simulation time')
    ylabel('oscillator number')
    
    keyboard
end

% Plot theta for error checking.  Reshape is for Alex's Style Transfer Starwars Images.
% Moving this analysis out to main_Kuramoto.
if(0)
    times_to_plot = [1:20,180]


    for t = times_to_plot

        figure, imagesc(reshape(theta(t,:),90,160)), caxis([0 2*pi]), colormap(hsv), colorbar
        title(['Iteration #',num2str(t)])

    end
    
    keyboard

end


disp(['Memory Check in KuramotoB Simulation : '])
%whos
check_memory_usage
    

% %% Create AVI movie of phase oscillators around circle
% if(unitCircMov)
%     close(writerObj);
% end