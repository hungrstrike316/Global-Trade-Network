function [MC, metaCluster] = main_Kuramoto(netParams, kurParams, netflags, kurflags, K, doMCA, ksBracket)
         

disp(['In Main_Kuramoto Function : '])

runs = 1;

%% Extract Variables from kur & net Params & flags data structures
%
% Extract data from netParams
vars = fieldnames(netParams);
for i=1:numel(vars)
    eval([vars{i},'=netParams.',vars{i},';'])  
end
%
% Extract data from kurParams
vars = fieldnames(kurParams);
for i=1:numel(vars)
    eval([vars{i},'=kurParams.',vars{i},';']) 
end
%
% Extract data from netflags
vars = fieldnames(netflags);
for i=1:numel(vars)
    eval([vars{i},'=netflags.',vars{i},';'])
end
%
% Extract data from kurflags
vars = fieldnames(kurflags);
for i=1:numel(vars)
    eval([vars{i},'=kurflags.',vars{i},';']) 
end

% % For file naming.
% TiScale_Tag = num2str(TiScale);
% TiScale_Tag(TiScale_Tag=='.')='p';


%% If output file already exist (ie. this parameter combination has already been run), exit now.
if( dataFileChk )
    
    brk_flg = 0;

    if( savemetaCluster & exist([dataKurDir,'KurMC_',fname,KurParamsTag,'_ks',ksBracket,'.mat'],'file') ) % ,'_kscale',num2str(Kscale),'_tscale',TiScale_Tag,'_runs',num2str(runs),
        disp('Kur Data File already exists:')
        [dataKurDir,'KurMC_',fname,KurParamsTag,'_ks',ksBracket,'.mat'] % ,'_kscale',num2str(Kscale),'_tscale',TiScale_Tag,'_runs',num2str(runs)
        disp('Next...')
        brk_flg=1;
    end
    %
    if(brk_flg)
        MC=0;
        metaCluster = 0;
        return
    end
    
end




%% Build Coupling Matrix and vector of Oscillator Natural Frequencies

if ~exist('K','var')
    K = build_K2(netParams); % Build coupling matrix for hand crafted network using Win, Wout, C, Rmax, etc in kurParams.
end





[dirPre,sizeGoodIm] = onCluster;            % I use sizeGoodIm below when saving figures.
                                            

%% Formatting for plots
% for i = 1:N
%     OscLeg{i} = [num2str(i),' - w=',num2str(w(i),3)];
% end
%

% Note:  This will work ok up to 7 clusters (cludge)
colors = ['bgrkcmybgrkcmybgrkcmybgrkcmybgrkcmybgrkcmybgrkcmybgrkcmybgrkcmybgrkcmybgrkcmybgrkcmy',...
        'bgrkcmybgrkcmybgrkcmybgrkcmybgrkcmybgrkcmybgrkcmybgrkcmybgrkcmybgrkcmybgrkcmybgrkcmybgrkcmybgrkcmybgrkcmy'];
stylee = cell(1,numel(gT{1}));
for i = 1:numel(gT{1})
    stylee{i} = colors(gT{1}(i)); % This not doing the right thing, but I dont worry about it for now.
end

% Create clusters colormap that matches 'colors' & 'stylee' used in imagesc plot
cl_map(1,:) = [0 0 1]; % blue  ('bgrkcmybgrkcmy'....)
cl_map(2,:) = [0 1 0]; % green 
cl_map(3,:) = [1 0 0]; % red
cl_map(4,:) = [0 0 0]; % black
cl_map(5,:) = [0 1 1]; % cyan
cl_map(6,:) = [1 0 1]; % magenta
cl_map(7,:) = [1 1 0]; % yellow
cl_map(8,:) = [0 0 0]; % black
cl_map(9,:) = [0 0 1]; % blue 
cl_map(10,:) = [0 1 0]; % green 
cl_map(11,:) = [1 0 0]; % red
cl_map(12,:) = [0 1 1]; % cyan
cl_map(13,:) = [1 0 1]; % magenta
cl_map(14,:) = [1 1 0]; % yellow ...

% A Picture of a colorwheel I will use as a colorbar for oscillator phase
colorwheel = imread([dirPre,'images/HSV_colorwheel.jpeg']);

% Colormaps to use in plots        
cmapRWB = rd_plotColorbar('redwhiteblue',256);
cmapRW = rd_plotColorbar('whitered',128);
            
% Visualize Phase Interaction Function (PIF)
if(kurflags.PIFplt)
    viz_PIF(PIFlg, PIFparams, [saveDir,'imgs/']);     % visualize PIF.
end
    


% run metaClustering analysis on StrawMan Model ( see how far apart image
% pixel values are separated). Is analogous to binary segmentation of image
% based on mean pixel value. If this performs well, then image patch was
% easy to segment.  Want to compare this performance with that of Kuramoto
% system
%
% disp('For Pixel Only Strawman Model : ')
% [MCsm] =  metaClusterAnalysisB(reshape(im,numel(im),1), netParams, kurParams, 0); % NEED TO IMPROVE THIS TO HANDLE LARGER IMAGE PATCHES !
% NOTE: No longer doing metaClusteringAnalysis on strawman because that is the initial phase embedding so it is happening anyway.

%% Run Kuramoto Simulation & Analyze Results
for r = 1:runs
    
    
    w = build_w1(netParams,kurParams);      % note: maybe can also make w a function of the input image.
                                            % move this inside run loop to change natural frequency on each run.
   

    % (1). Run Kuramoto Simulation - The main loop to simulate Kuramoto dynamics through timesteps
    disp(['Run #',num2str(r),'/',num2str(runs),' -- Kuramoto Simulation Running for ',num2str(T),' time steps.'])
    [theta] = KuramotoB(K, w, netParams, kurParams, kurflags, imgsKurDir);
    

    % (2). MetaCluster Analysis:  Creates a "Separation" measure of how well oscillators are 
    %      clumped within a cluster and how well clusters are separated in phase space.
    if(doMCA)
        
        
        % USE SPP TO SPLIT 180 TIME STEPS DOWN TO 18 + Initialization.
        % 
        % BUT ALSO, HOW TO MAKE MOVIES OF BEGINNING OF SIMULATION BECAUSE
        % THATS WHEN THE 'MAGIC' HAPPENS.
        

        % want to understand if phaseAtClk and theta are similar or if not why not for StarWars Images.
        
        
        X = visKurPhase_inHSV(im, reshape(theta(end,:),Ndims));        
        [MC.F] = compute_Spatial_Gradient(X, 1);
        
        
    else
        disp('WARNING:  NOT DOING METACLUSTER ANALYSIS ON KURAMOTO!! DID YOU FORGET ABOUT THIS?? ')
        MC = 'not doing';
    end
    
    
    
    % (3). phaseAtClk is phase of each oscillator demodulated by muW. 
    %      For visualization purposes, dont want plots osc phase plots to wrap around because it is distracting
    %      Set a clock signal
    
    
    % NOTE: THIS MAY ALSO HAVE TO BE DONE IN BATCHES FOR LARGE IMAGES (FOR FULL IMAGE 400x300 and 300ms, DL is 120,000 x 180) - BASICALLY
    % WHATEVER WE DECIDE TO DO IN METACLUSTERANALYSISB, WE SHOULD DO THE SAME HERE...
    t = tau*[1:T];                                                         % Building up a vector of time points based on user specifications.
    dl = diff(theta,1);
    jumpind = [dl<-pi];                                                    % now if jumpind(i) = true, we know that the point
    %                                                                      %   [lat(i) lon(i)] is the first after a jump
    for i=1:N                                                              % loop through all N oscillators
        blockEnds = find(jumpind(:,i));  
        blockEnds = [1;blockEnds];                                         % include initial phase distribution too.
        %
        for j=1:numel(blockEnds)                                           % loop over what?
            ind = blockEnds(j);
            %
            a = theta(ind,i);                                              % phase just before wrap around
            b = theta(ind+1,i);                                            % phase just after wrap around
            ta = t(ind);                                                   % time stamp just before wrap around
            tb = t(ind+1);                                                 % time stamp just after wrap around
            %
            t_2pi{i}(j) = ta + (2*pi - a)/(2*pi + b - a)*tau;              % time of 2pi crossing
            t_clk(j) = (j-1)/muW;
        end
    end
    
    
    % Find number of 2pi crossings that osc with least amount has.
    min_2piX = 1e12;
    for i = 1:N
        minCurr = numel(t_2pi{i});
        min_2piX = min(min_2piX,minCurr);
    end
    min_2piX = min(min_2piX,T/spp);
    %
    for i = 1:N
        phaseAtClk(i,:) = 2*pi*mod(t_2pi{i}(1:min_2piX),1/muW)/(1/muW);
        tt = 0:min_2piX;
    end

    % Include phase initialization explicitly in phaseAtClk
    phaseAtClk = [theta(1,:)',phaseAtClk];
    
    
    % Add these two things to the meta cluster structure.
    metaCluster(r).phaseAtClk = phaseAtClk;
    metaCluster(r).t_2pi = t_2pi;
   

    % Set up and save a movie of time evolution of coupled oscillator system
    if(kurflags.OscPhaseImg_Mov)

        writerObj = VideoWriter([imgsKurDir,'Mov_OscPhaseImg_',KurParamsTag,'_kscale',num2str(kurParams.Kscale),'_tscale',TiScale_Tag,'_run',num2str(r),'.avi']);
        writerObj.FrameRate = 5;
        open(writerObj);

        disp('Making a movie of oscilllator phases at 2\pi clock intervals.')
        
        movieFrames = [1:10, (2*spp):spp:(spp*(T/spp-2)),  size(theta,1)-10:size(theta,1)];
        RateDistStd = [0 0.01 0.1 0.3 0.5 0.7 0.9];
        
        
        i=0;
        for P = movieFrames
            i=i+1;
            
            kur = reshape( theta(P,:), netParams.Ndims);
            [mnCluster.kur,stdCluster.kur] = calc_ClusterMnNVars(gT,kur,1);
            %
            dp = compute_SensitivityIndex(mnCluster.kur,stdCluster.kur,sizeCluster,1,RateDistStd);
            DPtot(:,:,i) = dp;
            
            
            DPmn(:,i) = mean(dp); % average across different ground truths
            DPstd(:,i) = std(dp); % std across different ground truths
             
        end

        %
        
        for i = 1:numel(movieFrames)
        
            disp([num2str(i),'/',num2str(numel(movieFrames))])

            phase = reshape(theta(movieFrames(i),:),netParams.Ndims); 

            pp = figure;
            set(gcf,'units','normalized','position',[0 0 1 1])
            
            % Plot Phase of oscillators at this instant in time
            subplot(2,3,1), imagesc(phase)
            caxis([0 2*pi]), colormap('HSV')
            title(['Oscillator Phase'],'FontSize',20,'FontWeight','Bold');
            xlabel(['(t = ',num2str(tau*spp*i,2),')'],'FontSize',18,'FontWeight','Bold')
            set(gca, 'FontSize',16, 'FontWeight','Bold','XTick',[],'YTick',[])
            freezeColors
            colorbar, cbfreeze
            
            % Histogram of phase of oscillators.
            for j = 1:numel(unique(gT{1}))
                ind = find(gT{1} == j);
                [hC(:,j), xC(:,j)] = hist(theta(movieFrames(i),ind),100);
            end
            subplot(4,3,5), bar(xC,hC), xlim([0 2*pi]) 
            
            % Plot weight distribution of K matrix
            %subplot(4,6,[11,12]), hold on,
            subplot(4,3,2), hold on
            x = repmat(reshape(gT{1},1,numel(gT{1})), numel(gT{1}), 1 );
            y = repmat(reshape(gT{1},1,numel(gT{1}))', 1, numel(gT{1}) );
            clusterTruth = (x==y); 
            [histWin2, histWin2X] = hist(K(find(clusterTruth.*Wdist)),100);      % When 2 pixels are within a cluster (Rmax taken into consideration)
            [histWout2, histWout2X] = hist(K(find(~clusterTruth.*Wdist)),100);   % When 2 pixels are different a cluster 

            plot(histWin2X,histWin2./sum(histWin2),'m','LineWidth',3)
            plot(histWout2X,histWout2./sum(histWout2),'c','LineWidth',3)
            title(['Graph Weight Distribution : Rmax = ',num2str(Rmax)],'FontSize',18,'FontWeight','Bold')
            legend('Within Cluster','Across Cluster')
            set(gca,'FontSize',16,'FontWeight','Bold')

            
            % Plot input image
            subplot(2,3,3), hold on,
            imagesc(im), colormap('bone')
            axis off ij
            title(['Input Img'],'FontSize',18,'FontWeight','Bold')
            freezeColors

            % Plot d'-metric change with simulation time (Plot GTs separated)
            subplot(2,5,[6:9]), hold on,
            colors = 'rgbkcmy';

            for i = 1:numel(RateDistStd)
                plot(squeeze(DPtot(:,i,:))',colors(i),'LineWidth',2);
                h(i) = plot(squeeze(DPtot(1,i,:))',colors(i),'LineWidth',2); 
                RD_leg{i} = num2str(RateDistStd(i));
            end
            %
            for j = 1:numel(gT)
               text(numel(movieFrames), DPtot(j,1,end),['\color{red}{gT#,',num2str(j),'}']) 
            end
            hleg = legend(h,RD_leg,'Location','SouthEast');
            set(get(hleg,'title'),'string','Rate Distortion \sigma')
            set(gca,'XTick',1:numel(movieFrames),'XTickLabel',movieFrames.*(Tsec/T))
            xlabel('Simulation Time (sec)')
            ylabel('d''-metric')
            rotateXLabels(gca,40)
            
            % Plot Ground truths
            for i = 1:numel(gT)
                subplot(2*numel(gT), 5, 5*numel(gT)+5+5*(i-1)),
                imagesc(gT{i}), colormap(cl_map)
                caxis([1 size(cl_map,1)])
                set(gca,'Xtick',[],'Ytick',[])
                axis ij square
                ylabel(['gT#',num2str(i)],'FontSize',18,'FontWeight','Bold')
                freezeColors
                
            end
            
            % Save Frame as part of the movie file
            mov = getframe(pp);

            writeVideo(writerObj,mov);
            close(pp)
        end
        
        close(writerObj);

    end
        
    % clear variables that may change sizes from run to run
    clear phaseAtClk

end % end looping over runs.



%% Run Diagnostics averaged over all runs & Save Result


Ks = num2str(kurParams.Kscale);
Ks(Ks=='.')='p';

if(kurflags.savemetaCluster)
    disp(['Saving Mat File: KurMC_',kurflags.fname,KurParamsTag,'_ks',ksBracket]) % ,'_kscale',num2str(kurParams.Kscale),'_tscale',TiScale_Tag,'_runs',num2str(runs)
    save([dataKurDir,'KurMC_',kurflags.fname,KurParamsTag,'_ks',ksBracket],...    % ,'_kscale',num2str(kurParams.Kscale),'_tscale',TiScale_Tag,'_runs',num2str(runs)
        'netParams','netflags','kurParams','kurflags','metaCluster','MC');
end

