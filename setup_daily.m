%creates cell with objects needed for estimation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%General Setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning('off')

% Notes by SB Sim
% I change the order of some parameters to insert the monetary shocks
% into the law of motion of trend.

% Parameter vector
% [sig_pi sig_tau rho_g sig_g con_month sig_month con_day sig_day con_be sig_be
% theta_g_0 theta_tau_0 sig_monetary_shock theta_g_1~59 theta_tau_1~59]'

% See Trend Behavior.pdf for the detail.




setup.length_param_vector=131;


%now picking prior distributions
% parameters 5 and 6 are the parameters of the monthly meas. eq.

% NEW CODE_start_SBS
setup.index_gamma=[1 2 4 6 8 10 13]'; % Add 6 (stdv of CPI), 12->13 (stdv of monetary shock)
setup.index_beta=[3];
setup.beta_prior_shape1=4;
setup.beta_prior_shape2=4;
setup.gamma_prior_shape=ones(7,1); % 6->7 (add stdv of CPI)
setup.gamma_prior_scale=0.5*ones(7,1); % 6->7 (add stdv of CPI)


setup.index_uniform=[];
setup.index_normal=[5 7 9 11 12 14:131]; % Remove 6, Add 12 and 13~71->14~131

%now pick prior parameters
setup.normal_prior_means=[0 0 0 zeros(1,120)]'; 
setup.normal_prior_std=[.0001 5 5 0.25*ones(1,120)]';
exponent=0:1:58;
scaling_factor=0.95;
scaling=scaling_factor.^exponent;

setup.normal_prior_std(6:64)=setup.normal_prior_std(6:64).*scaling';
setup.normal_prior_std(65:end)=setup.normal_prior_std(65:end).*scaling';
% NEW CODE_end_SBS


%storage and display options

%display every disp iter draw
setup.disp_iter=500;
% keep every keep_draw th draw
setup.keep_draw=1;


setup.TVKF=1;





%parameter transformations for proposal (we want proposal to be unrestricted)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Keep this block unchanged for now
setup.length_logit_general=0;
setup.index_logit_general=[];
 setup.logit_general_lb=[]';
setup.logit_general_ub=[]';


setup.length_log=0;
setup.index_log=[];

setup.length_logit=0;
setup.index_logit=[]; 
setup.transform=0;
setup.state_size=94;

setup.initial_provided=1;
setup.state_initial=[zeros(94,1)];
setup.state_initial(3)=1; %constant
setup.cov_initial=10*eye(94);
setup.cov_initial(3,3)=eps; %taking care of constant
setup.cov_initial(4,4)=eps; %lagged monetary shock
setup.cov_initial(36:end,36:end)=eps*eye(59); %lagged monetary shocks (we should later use actual data to set the initial states here!)




%should additional matrices be stored
setup.add_matrices=0;
%dimension of those variables
setup.dim_add_matrices=[];
%proposal=1 ->standard RW
%adaptive MH not implemented in this code
setup.proposal=1;
%log likelihood computation
%likelihood=1 -> SS KF (only uses the SS covariance matrix, initial mean
%still has to be provided)
%likelihood=2 ->  KF
%likelihood=3 ->  non-KF likelihood
setup.likelihood=2;
%name of function that evaluates log prior
setup.prior_function='prior'; %right now prior function has to be called prior.m!
%initial value for the state and covariance of the state
setup.wrapper='system_matrices';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


setup.number_blocks=2;
setup.index_block{1}=[1:4 7:131]'; 
setup.index_block{2}=[5:6]';






%%%%%%
%SMC OPTIONS

setup.N_phi=200;%100
setup.initial_SMC_scaling=.5;
setup.N_particles=15000;
setup.N_MH=5;
setup.number_of_draws=setup.N_MH;
setup.lambda=2;
setup.store_all_draws=0;
setup.par_pools = 26; % number of workers to be used in parallel mode

load('data');
setup.data =  data_for_estimation_final;
setup.sample_size=size(setup.data,2); 

