clear
%% init
result_folder = 'results/icse1/real/';
% load data
data_Adiac_aws_gpu_micro = csvread('../../experiments/deepxplore/aws-gpu-micro/exp/exp_Adiac.csv',1,0);
data_Adiac_azure = csvread('../../experiments/deepxplore/azure-22feb/exp/exp_Adiac.csv',1,0);
data_Coffee_aws_gpu1_theano_micro = csvread('../../experiments/deepxplore/aws-gpu1-theano-micro/exp/exp_Coffee.csv',1,0);
data_Adiac_aws_gpu1_theano_micro = csvread('../../experiments/deepxplore/aws-gpu1-theano-micro/exp/exp_Adiac.csv',1,0);

% loading configurations we have measured
configuration = csvread('../../experiments/deepxplore/configurations_exp.csv',0,0);

%------------------------------------
% select source and target
exp_name = 'deepxplore';
source_name = 'Adiac_azure';
target_name = 'Adiac_aws_gpu_micro';

source = data_Adiac_azure(:,3);
target = data_Adiac_aws_gpu_micro(:,3);
% source = data_Adiac_aws_gpu1_theano_micro(:,3);
% target = data_Coffee_aws_gpu1_theano_micro(:,3);
% source = data_Adiac_azure(:,3);
% target = data_Coffee_aws_gpu1_theano_micro(:,3);

%------------------------------------
% config options
exec_l2s = 1; %
exec_l2s_seams = 1;
exec_seams1 = 1;
exec_seams2 = 0;
exec_modelshift = 1;
exec_random = 1;
exec_l2s_uncertainty_reduction = 1;
exec_hyp_auto = 0;

ndim = size(configuration, 2);
n = 1000; % total random samples to select samples for training CART
uuid = char(java.util.UUID.randomUUID);
zt = zeros(1, ndim); % zero tensor
budget = 70; % number of samples in the target
init_tr_size = 2; % initial sample size, after this we start training a model
rnd_size = 2; % number of random samples at each iterations (to select one)
epsilon = 0.1; % adjust exploration-exploitation
N = 1000; % discretization level for KL
syn = 0; % 1 if this is running a synthetic model, 0 if running real system
rs = RandStream('mt19937ar','Seed',0);
pvalue_threshold = 0.05;
%------------------------------------
% test set
xTest = configuration;
yTest = target;
yTest_source = source;

%------------------------------------
% remove invalid configs
idx_invalid_source = (yTest_source == 0);
idx_invalid_target = (yTest == 0);
idx_invalid = idx_invalid_source | idx_invalid_target;

% response lower than 1
idx_lower_one = (abs(yTest) < 1 & abs(yTest) > 0);

% prunning out invalid and ill-conditioned samples
xTest = xTest(~idx_invalid & ~idx_lower_one,:);
yTest_source = yTest_source(~idx_invalid & ~idx_lower_one);
yTest = yTest(~idx_invalid & ~idx_lower_one);
actual = yTest;

%% create T based on source, L2S transfers this to the target
T = ff2n(ndim);
T = [T zeros(length(T),1)];
L = zeros(1,ndim+1);
% learning a model to extract knowledge about interactions
mdl = stepwiselm(xTest, yTest_source, L, 'upper', T);

%% extract dimensions and interactions
terms_valid = [];
idx_valid_terms = 1; % the first index of terms is for intercept so always valid
idx_uninfluential_options = [];
terms = mdl.Formula.Terms;
num_terms = size(terms,1);

% excluding uninfluential terms that appeared in the stepwise regression
% process
for i = 1:num_terms
    if mdl.Coefficients.pValue(i)>pvalue_threshold && length(find(terms(i,:)))==1 && any(terms(i,:)>0)
        idx_uninfluential_options = [idx_uninfluential_options; i];
    end
end

for i = 1:num_terms
    if mdl.Coefficients.pValue(i)<=pvalue_threshold && any(terms(i,:)>0)
        idx_valid_terms = [idx_valid_terms;i];
    end
end

% prunning valid terms that include options from the uninfluential ones
idx_invalid = [];
for i = 1:length(idx_valid_terms)
    idx_invalid(i) = 0;
    if length(find(terms(idx_valid_terms(i),:))) > 1
        for j = 1:length(idx_uninfluential_options)
            if ~isempty(find(terms(idx_valid_terms(i),:) & terms(idx_uninfluential_options(j),:),1))
                idx_invalid(i) = 1;
            end   
        end
    end
end

terms_valid = terms(idx_valid_terms(~idx_invalid),1:ndim);

% if we want to sort them to create a prioritised list
terms_coeff = mdl.Coefficients.Estimate(idx_valid_terms);
[~,idx_terms_sorted] = sort(abs(terms_coeff),'descend');

%% construct the sample set

% create Ts based on source, L2S transfers this information to the target
% this structure in Ts not only allows for starting from a reasonable
% model for learning the target, but more importantly also derive the
% samples we generate
Ts = terms_valid;

% T for the target
Tt = [];
idx1 = nchoosek(1:1:ndim,1);
idx2 = nchoosek(1:1:ndim,2);
idx3 = nchoosek(1:1:ndim,3);
Tt = [Tt; zt];

for i = 1:size(idx1,1)
    it = zt;
    it(1, idx1(i,:)) = 1;
    Tt = [Tt; it];
end

for i = 1:size(idx2,1)
    it = zt;
    it(1, idx2(i,:)) = 1;
    Tt = [Tt; it];
end

for i = 1:size(idx3,1)
    it = zt;
    it(1, idx3(i,:)) = 1;
    Tt = [Tt; it];
end

% Adding extra zero to cover response in the structure;  t-by-(p + 1)
Ts = [Ts, zeros(size(Ts,1), 1)];
Tt = [Tt, zeros(size(Tt,1), 1)];

%% Start designing active space (prioritised list of configurations)

% [~, idx_imp_o] = sort(abs(source_model.o(source_model.o~=0)),'descend');
% [~, idx_imp_i] = sort(abs(source_model.i),'descend');

conf = [];
conf = Ts(:, 1:ndim);
model_terms = Ts(2:end, 1:ndim);
no_influential_terms = size(model_terms, 1);
nzo = ~all(model_terms == 0);

idx_rnd = randperm(no_influential_terms);

it = zt;
it(nzo) = 1; % adding all 1s
conf = [it; conf];

% Create subspace samples from existing interactions o1o2o3 -> we
% already did 111, now we want to do 110, 101, 011, etc
for i = 1:no_influential_terms
    idx = [];
    idx = find(model_terms(i,:)~=0);
    if length(idx) > 2
        idx2 = nchoosek(1:1:length(idx),2);
        for j = 1:size(idx2,1)
            it = zt;
            it(1, idx(idx2(j, :))) = 1;
            conf = [conf; it];
        end
    end
end

% create superspace samples from existing dimensions and interactions
idx2 = nchoosek(1:1:no_influential_terms,2);

for i = 1:size(idx2,1)
    it = double(model_terms(idx_rnd(idx2(i,1)),:) | model_terms(idx_rnd(idx2(i,2)),:));
    conf = [conf; it];
end

% only consider unique configurations and preserving their order
conf = unique(conf,'rows','stable');

%------------------------------------
% initializations before start training
num_active = size(conf, 1);
exploit_idx = 0;
Xtrain = [];
ytrain = [];
ee = [];
mdl_glm = struct([]);
v = randperm(n);
idx_rand = v(1:budget);
Xrand = xTest(idx_rand, :);
yrand = yTest(idx_rand);

%% iterative sampling to add one samples to the training set until the
% budget has exhausted

for idx_tr = 1:budget
    
    %------------------------------------
    % determining next sample
    r=rand; % get 1 uniform random number
    x=sum(r>=cumsum([0, 1-epsilon, epsilon])); % check it to be in which probability area
    
    idx_next = [];
    % choose either explore or exploit
    if x == 1   % exploit        
        ee(idx_tr) = 0;
        exploit_idx = exploit_idx + 1;
        if exploit_idx <= num_active
            next_sample = conf(exploit_idx,:);
        else % if we do not have more samples to exploit then we randomely sample
            next_sample = randi([0,1], 1, ndim);
        end        
    else  % explore (sample from the whole configurations that make the distribution closer, therefore more diverse)       
        ee(idx_tr) = 1;
        % when we try to reduce uncertainty for exploration instead of random samples
        if exec_l2s_uncertainty_reduction && idx_tr > init_tr_size + 1
            [~,idx_next] = max(std_t);
            next_sample = xTest(idx_next,:);
        else % random samples (default)
            rand_samples = randi([0,1], rnd_size, ndim);
            if ~isempty(mdl_glm)
                y_rand_samples = predict(mdl_glm, rand_samples);
                kl = [];
                for i = 1:rnd_size
                    data_source = yTest_source;
                    data_target = [ytrain;y_rand_samples(i)];
                    [~,bins_source] = discretize(data_source,N);
                    [~,bins_target] = discretize(data_target,N);
                    pd = fitdist(data_source,'Kernel');
                    pd_source = pdf(pd,bins_source);
                    pd = fitdist(data_target,'Kernel');
                    pd_target = pdf(pd,bins_target);
                    kl(i) = KLDiv(pd_source,pd_target);
                end
                [~,idx_sorted] = sort(kl);
                next_sample = rand_samples(idx_sorted(1),:); % select the points that make the target distribution closer to the source                
            else % if we do not have a model yet (i.e., first iteration), we select the first random sample
                next_sample = rand_samples(1,:);
            end
        end        
    end
    
    idx_next = find(ismember(configuration,next_sample, 'rows'));
    y_next_sample = target(idx_next);
    
    %------------------------------------
    % adding selected sample to the sample set
    Xtrain = [Xtrain; next_sample];
    ytrain = [ytrain; y_next_sample];
    
    %------------------------------------
    % only start training the models after collecting certain number of samples
    
    if idx_tr > init_tr_size
        % Transfer happens when we start off the target model with the
        % structure of the source
        % We also consider an upper bound based on our observations of
        % configurable systems
        
        %------------------------------------
        % GLM
        mdl_glm = stepwiseglm(Xtrain, ytrain, Ts, 'upper', Tt);
        
        % calculating accuracy
        predicted_t = predict(mdl_glm, xTest);
        scoreNMSE_glm(idx_tr) = nmse(actual, predicted_t)
        scoreAE = ae(actual, predicted_t);
        scoreAP = abs(scoreAE./(actual)*100); % absolute percentage error
        scoreMAP_glm(idx_tr) = mean(scoreAP)
        
        %------------------------------------
        % GP by handset hyper-parameters based on intuitions
        % Initialize length scales of the kernel function at 10 and signal
        % and noise standard deviations at the standard deviation of the response.
        sigma0 = std(ytrain);
        sigmaF0 = sigma0;
        sigmaM0 = 10*ones(ndim,1);
        mdl_gp = fitrgp(Xtrain, ytrain,'KernelFunction','ardsquaredexponential','Verbose',0, ...
            'Optimizer','lbfgs', 'KernelParameters',[sigmaM0;sigmaF0],'Sigma',sigma0,'InitialStepSize','auto');
        
        % calculating accuracy
        [predicted_t, std_t] = predict(mdl_gp, xTest);
        scoreNMSE_gp(idx_tr) = nmse(actual, predicted_t)
        scoreAE = ae(actual, predicted_t);
        scoreAP = abs(scoreAE./(actual)*100); % absolute percentage error
        scoreMAP_gp(idx_tr) = mean(scoreAP)
        scorestd_gp(idx_tr) = sum(std_t)
        
        %------------------------------------
        % GP with auto-optimized hyper-parameters
        if exec_hyp_auto
        mdl_gp_auto = fitrgp(Xtrain, ytrain,'KernelFunction','squaredexponential',...
            'OptimizeHyperparameters','auto','Verbose',0,'HyperparameterOptimizationOptions',...
            struct('AcquisitionFunctionName','expected-improvement-plus', 'ShowPlots', false, 'Verbose',1));
        
        % calculating accuracy
        predicted_t = predict(mdl_gp_auto, xTest);
        scoreNMSE_gp_auto(idx_tr) = nmse(actual, predicted_t)
        scoreAE = ae(actual, predicted_t);
        scoreAP = abs(scoreAE./(actual)*100); % absolute percentage error
        scoreMAP_gp_auto(idx_tr) = mean(scoreAP)
        end
        
        %------------------------------------
        % taking random samples with the same size of L2S
        n_train = size(Xtrain, 1);
        Xtrainrnd_t = Xrand(1:n_train,:);
        ytrainrnd_t = yrand(1:n_train);

        %-----------------------------------
        % CART
        mdl_cart = fitrtree(Xtrainrnd_t, ytrainrnd_t,'PredictorSelection','curvature','Surrogate','on');
        
        % calculating accuracy
        predicted_t = predict(mdl_cart, xTest);
        scoreNMSE_cart(idx_tr) = nmse(actual, predicted_t)
        scoreAE = ae(actual, predicted_t);
        scoreAP = abs(scoreAE./(actual)*100); % absolute percentage error
        scoreMAP_cart(idx_tr) = mean(scoreAP)
        
        %-----------------------------------
        % CART with auto-optimized hyper-parameters
        if exec_hyp_auto
        mdl_cart_auto = fitrtree(Xtrainrnd_t, ytrainrnd_t,'OptimizeHyperparameters','auto', ...
            'PredictorSelection','curvature','Surrogate','on', 'HyperparameterOptimizationOptions',...
            struct('AcquisitionFunctionName','expected-improvement-plus', 'ShowPlots', false, 'Verbose',1));
        
        % calculating accuracy
        predicted_t = predict(mdl_cart_auto, xTest);
        scoreNMSE_cart_auto(idx_tr) = nmse(actual, predicted_t)
        scoreAE = ae(actual, predicted_t);
        scoreAP = abs(scoreAE./(actual)*100); % absolute percentage error
        scoreMAP_cart_auto(idx_tr) = mean(scoreAP)
        end
        %------------------------------------
        % GP with random samples
        sigma0 = std(ytrainrnd_t);
        sigmaF0 = sigma0;
        sigmaM0 = 10*ones(ndim,1);
        mdl_gp_rnd = fitrgp(Xtrainrnd_t, ytrainrnd_t,'KernelFunction','ardsquaredexponential','Verbose',0, ...
            'Optimizer','lbfgs', 'KernelParameters',[sigmaM0;sigmaF0],'Sigma',sigma0,'InitialStepSize','auto');
        
        % calculating accuracy
        [predicted_rnd_t, std_rnd_t] = predict(mdl_gp_rnd, xTest);
        scoreNMSE_gp_rnd(idx_tr) = nmse(actual, predicted_rnd_t)
        scoreAE = ae(actual, predicted_rnd_t);
        scoreAP = abs(scoreAE./(actual)*100); % absolute percentage error
        scoreMAP_gp_rnd(idx_tr) = mean(scoreAP)
        scorestd_gp_rnd(idx_tr) = sum(std_rnd_t)
        
    end
end

%% saving results
save([result_folder exp_name '-' num2str(source_name) '-' num2str(target_name) '-' uuid '.mat'], 'score*', 'xTest', 'Xtrain', 'ee', 'conf')