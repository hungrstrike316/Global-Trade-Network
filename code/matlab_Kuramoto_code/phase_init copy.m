

% This function will initialize the phase of all oscillators based on
% filter activations (on pixel intensity) not randomly.  Using this, the
% whole simulation should be deterministic and we should not have to run
% multiple simulations.


% Can have a switch case statement using this...
% init_type = 'random'; % 'PIV'


% theta_scale = TiScale;
% theta(1,:) = pi*theta_scale*reshape(im,1,N); % initialize phase proportional to pixel intensity.

% theta(1,:) = pi*theta_scale*reshape(gT{1},1,N); % start with correct answer (cheating).





%% The 3 reasonable options right now.

% theta(1,:) = pi/2*rand(1,N); % tighter random initial conditions. maybe randn.

theta(1,:) = 2*pi*rand(1,N); % random initial conditions.

% theta(1,:) = zeros(1,N); % start all at zero phase.