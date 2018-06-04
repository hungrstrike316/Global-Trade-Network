function K = build_K2(params)

% Syntax: [K, K_true] = build_K2(params,flags)
%
%
% The coupling matrix K has weak all-to-all connectivity with clusters of
% nodes having strong connectivity within cluster. Also, there are no self
% connections so diagonal of K is zero.
%
% This function constructs a coupling matrix K describing the connectivity
%   for N nodes separated into C equal sized clusters. N must be divisible 
%   by C.  
%
% Within a cluster, the weights are drawn from a normal distribution with 
%   mean Strng and standard deviation sigStrng. 
%
% Across clusters, the weights are drawn from a normal distribution with 
%   mean Weak and standard deviation sigWeak.
%
% Output is the Coupling matrix K.

%% Unpack necessary variables from data structures
vars = fieldnames(params);
for i=1:numel(vars)
    eval([vars{i},'=params.',vars{i},';'])
end



%% Calculate Kdist - coupling dependence based on Euclidian Distance between oscillators
Nx = Ndims(1);
Ny = Ndims(2);
%
y_vec = [1:Ny]';
y_mat = repmat(y_vec,1,Nx);
y_unwrapt = reshape(y_mat,1,N);
y_un_mat = repmat(y_unwrapt,numel(y_unwrapt),1);
%
x_vec = [1:Nx];
x_mat = repmat(x_vec,Ny,1);
x_unwrapt = reshape(x_mat,1,N);
x_un_mat = repmat(x_unwrapt,numel(x_unwrapt),1);
%
x_dist = abs(x_un_mat - x_un_mat');  %
y_dist = abs(y_un_mat - y_un_mat');  %
r = sqrt(x_dist.^2 + y_dist.^2);     % Euclidian or L2 distance between two pixels.
%
Kdist = (r <= Rmax);                 % a binary mask allowing oscillators closer than Rmax to have connection





%% Build Ksim - Coupling matrix based on Similarity (or clusters)
Ksim = Weak + sigWk.*randn(N); % put in weak connections between all oscillators
%
for i = 1:C % loop thru clusters.
    ind = find(gndTruth{1}==i);
    Ksim(ind,ind) = Strng + sigSg.*randn(numel(ind)); % put in strong connections within clusters (Block Diagonal)
end
%


%% Make possible for distant oscillators to be connected with some probability.
th = params.PconnFar;          % Probability of a connection between distant oscillators
K_rand = rand(N);        % random matrix size of K
K_far = (K_rand > (1 - th));  
K_far = logical( triu(K_far) + triu(K_far)' ); % make symmetric


%% Set diagonal to zero - No Self Loops.
NoSelfLoopsMask = ones(N) - eye(N);

%% Add everything together.
K = Ksim .* logical(Kdist + K_far) .* NoSelfLoopsMask;
