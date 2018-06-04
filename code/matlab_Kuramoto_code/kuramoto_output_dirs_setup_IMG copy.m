function [kurflags] = kuramoto_output_dirs_setup_IMG(netParams, kurParams, netflags, kurflags, fname)









%% Extract Variables from  data structures
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

% Extract data from netflags
vars = fieldnames(netflags);
for i=1:numel(vars)
    eval([vars{i},'=netflags.',vars{i},';'])
end



%% Make directory to save output data files and images produced.
dirPre = onCluster;

saveDir = [dirPre,'output/Kuramoto/NetsFromImgs/',imageIn,'/'];

method(method==' ')='_'; % spaces in directory names dont go over well in some architectures.

% (1). Directory to store images generated in Kuramoto code
imgsKurDir = [saveDir,'imgs/Kur_PIF_',PIFlg,'/',method,'/']; %,'_rM',num2str(Rmax),'_sD',sD,'_sP',sP,'/'];
%
if ( ~exist(imgsKurDir,'dir') & (kurflags.phasePrecess | kurflags.OscPhaseImg_Mov) )
    mkdir(imgsKurDir)
end


% (2). Directory to store mat data file containing all the meta clustering results for a set of network topologies (N,C,R)
dataKurDir = [saveDir,'data/Kur_PIF_',PIFlg,'/',method,'/']; %,'_rM',num2str(Rmax),'_sD',sD,'_sP',sP,'/'];     
%
if ( ~exist(dataKurDir,'dir') & kurflags.savemetaCluster )
    mkdir(dataKurDir)
end
%
% Change a . to a p in sigW because file naming doesnt work.
sigWp = num2str(sigW);
sigWp(sigWp=='.')='p';
%

if strcmp(method,'IsoDiff')
    KurParamsTag = ['_rM',rM,'_NF_',num2str(muW),'_',sigWp]; % for file names
    KurTitleTag = ['rM',rM,' NF ',num2str(muW),' ',num2str(sigW)]; % for plots
else
    KurParamsTag = ['_rM',rM,'_sD',sD,'_sP',sP,'_NF_',num2str(muW),'_',sigWp]; % for file names
    KurTitleTag = ['rM',rM,' sD',sD,' sP',sP,' NF ',num2str(muW),' ',num2str(sigW)]; % for plots
end




% Put directory and tag names into kurflags data structure
kurflags.imgsKurDir = imgsKurDir;
kurflags.dataKurDir = dataKurDir;
kurflags.KurParamsTag = KurParamsTag;
kurflags.KurTitleTag = KurTitleTag;
%
kurflags.fname = [fname];