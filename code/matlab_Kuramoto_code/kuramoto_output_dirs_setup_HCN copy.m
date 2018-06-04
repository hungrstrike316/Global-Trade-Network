function [kurflags] = kuramoto_output_dirs_setup_HCN(netParams, kurParams, netflags, kurflags)









%% Extract Variables from data structures
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





%% Make directory to save output data files and images produced.
dirPre = onCluster;
saveDir = [dirPre,'output/Kuramoto/HandCookedNetworkBB/'];


% (1). Directory to store images generated in Kuramoto code
imgsKurDir = [saveDir,'imgs/PIF_',PIFlg,'_N',num2str(Ndims(1)),'x',num2str(Ndims(2)),'_C',num2str(C),'_Rmax',num2str(Rmax),'/']
%
if ( ~exist(imgsKurDir,'dir') & (kurflags.coherence | kurflags.TwoPiXing | kurflags.phasePrecess | kurflags.unitCircMov | kurflags.CosDist_2piX_Mov) )
    mkdir(imgsKurDir)
end


% (2). Directory to store mat data file containing all the meta clustering results for a set of network topologies (N,C,R)
dataKurDir = [saveDir,'data/MetaCluster/PIF_',PIFlg,'_N',num2str(Ndims(1)),'x',num2str(Ndims(2)),'_C',num2str(C),'_Rmax',num2str(Rmax),'/'];       
%
if ( ~exist(dataKurDir,'dir') & kurflags.savemetaCluster )
    mkdir(dataKurDir)
end
%
KurParamsTag = ['NF_',num2str(muW),'_',num2str(sigW),'_Win_',num2str(Strng),'_',num2str(sigSg),'_Wout_',num2str(Weak),'_',num2str(sigWk),'_runs',num2str(runs)];
KurParamsTag(KurParamsTag=='.')='p';


% Put directory and tag names into kurflags data structure
kurflags.imgsKurDir = imgsKurDir;
kurflags.dataKurDir = dataKurDir;
kurflags.KurParamsTag = KurParamsTag;