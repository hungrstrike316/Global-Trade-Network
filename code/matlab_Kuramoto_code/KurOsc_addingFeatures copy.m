% Not ready yet.  Coming to this soon.


f_osc = [zeros(10,1);ones(10,1)]';    % features of the oscillators
imagin = f_osc;


%  parameter values and flags for running functions below.
sigpix = 0.1;
sigdist = 1;
maskFlg = 1;
distBin = 1;
rmax  = 1;
plt = 1;

ximg = size(imagin,1);
yimg = size(imagin,2);
maxEnt = 0;
maxIter = 1000;
N = numel(imagin);







[W, Wconn, Wunc, Mask] = calc_weights(imagin,sigpix,sigdist,maskFlg,distBin,rmax,plt);



NoSelfLoopsMask = ones(N) - eye(N);
W = W .* NoSelfLoopsMask;




A = compute_AvgAssociation(W,0); % Average Association




L = compute_Laplacian(W,0,1); % Negative Graph Laplacian
Ln = compute_Laplacian(W,1,1); % Neg Normalized GL




% [Q,iter] = compute_Modularity(W,Wconn,Wunc,Mask,ximg,yimg,normlze,maxEnt,maxIter,topo,maskFlg)
%[Q,iter] = compute_Modularity(W,Wconn,Wunc,Mask,ximg,yimg,0,maxEnt,maxIter,0,1); % Modularity (Non-topo)
[Qt1,iter] = compute_Modularity(W,Wconn,Wunc,Mask,ximg,yimg,0,maxEnt,maxIter,1,0); % Modularity Topographic
%[Qt2,iter] = compute_Modularity(W,Wconn,Wunc,Mask,ximg,yimg,0,maxEnt,maxIter,1,1); % Modularity Topographic


figure,
% subplot(231), imagesc(Q),colorbar,title('Q')
subplot(232), imagesc(Qt1),colorbar,title('Topographic Q')
%subplot(233), imagesc(Qt2),colorbar,title('Topographic Q')
subplot(234), imagesc(A),colorbar,title('A')
subplot(235), imagesc(L),colorbar,title('-L')
subplot(236), imagesc(Ln),colorbar,title('Normalized -L')
