function [K] = build_K1(params,flags) %N, C, Strng, Weak, sigStrng, sigWeak, negWts)

% Syntax: [K] = build_K1(N, C, strng, weak, sigStrng, sigWeak, negWts)
%
% The coupling matrix K has weak all-to-all connectivity with clusters of
% nodes having strong all-to-all connectivity within cluster. Also, there 
% are no self connections so diagonal of K is zero.
%
% N nodes separated into C equal sized clusters. N must be divisible by C.  
%
% Within a cluster, the weights are drawn from a normal distribution with 
%   mean Strng and standard deviation sigStrng. 
%
% Across clusters, the weights are drawn from a normal distribution with 
%   mean Weak and standard deviation sigWeak.
%
% Output is the Coupling matrix K.

%% Unpack necessary variables from data structures
N = params.N;
C = params.C;
Strng = params.Strng;
Weak = params.Weak;
sigStrng = params.sigSg;
sigWeak = params.sigWk;
negWts = params.negWts;

%% Error Checks.

if mod(N/C,1)~=0
    error('Error:  N (number of oscillators) must be evenly divisible by C (number of clusters)')
end

if (N/C <= 1)
    error('Error:  must be more than 1 oscillator per clusters)')
end



% %% Fill in missing variables for sigmas if they arent there.
% if ~exist('sigStrng','var')
%     sigStrng = 0;   % single value for strong connections if not specified.
% end
% %
% if ~exist('sigWeak','var')
%     sigWeak = 0;    % single value for weak connections if not specified.
% end
% %
% if ~exist('negWts','var')
%     negWts = 1;     % allow negative weights if not specified.
% end


%% Build up Coupling Matrix K

% note: for now this does not allow for sigmas to be nonzero.
%       they arent even used.


K = Weak + sigWeak.*randn(N); % put in weak connections between all oscillators
%
for i = 1:C % loop thru clusters.
    indSt = 1+(N/C)*(i-1);
    indNd = i*(N/C);
    K(indSt:indNd,indSt:indNd) = Strng + sigStrng.*randn(N/C); % put in strong connections within clusters (Block Diagonal)
end
%
NoSelfLoopsMask = ones(N) - eye(N);
K = K .* NoSelfLoopsMask;

