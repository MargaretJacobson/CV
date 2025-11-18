clear;
clc;
close all;


% add path with subfolders
addpath(genpath('functions'))
addpath(genpath('data'))
coder_compilation;
tic

rng(100)

% Initialization
setup_daily;

% SMC
    delete(gcp('nocreate'))
    [draws, acc_rate, log_posteriors, statedraws, individual_post_kernels, total_draws] = sampling_SMC_parallel( setup );

    
results.timeSMC = toc/60;
disp(['SMC took ', num2str(results.timeSMC), ' minutes']);

save daily_smc results setup draws
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

upper=84;
lower=16;
med=50;

inflation_upper=prctile(statedraws(5,:,:),upper,3);
inflation_lower=prctile(statedraws(5,:,:),lower,3);
inflation_med=prctile(statedraws(5,:,:),med,3);
load data_total




figure;
plot(total_data_tt.ddate,total_data_tt.daily_pi,'LineWidth',2)
hold on
plot(cpi_rdate(2:end),monthly_pi,'LineWidth',2)
hold on
p1=plot(data_for_estimation.ddate,inflation_upper(1,end-length(data_for_estimation.ddate)+1:end),'r--','LineWidth',2)
p1.Color(4) = 0.5;
hold on
p2=plot(data_for_estimation.ddate,inflation_lower(end-length(data_for_estimation.ddate)+1:end),'r--','LineWidth',2)
p2.Color(4) = 0.5;
hold on
plot(data_for_estimation.ddate,inflation_med(end-length(data_for_estimation.ddate)+1:end),'r','LineWidth',2)

legend('daily','monthly')
grid on

print -depsc
savefig('states')


for dd=1:size(draws,2)
    
   irf(:,dd) = irf_computation(draws(:,dd),setup,60) ;
    
end

irf_lower=prctile(irf,lower,2);
irf_med=prctile(irf,med,2);
irf_upper=prctile(irf,upper,2);

figure
plot(1:size(irf,1),irf_lower,'-b','LineWidth',2)
hold on
plot(1:size(irf,1),irf_med,'-r','LineWidth',2)
hold on
plot(1:size(irf,1),irf_upper,'-b','LineWidth',2)
grid on

print -depsc
savefig('irfs')




for dd=1:size(draws,2)
    
   irf(:,dd) = irf_computation_trend(draws(:,dd),setup,60) ;
    
end

irf_lower=prctile(irf,lower,2);
irf_med=prctile(irf,med,2);
irf_upper=prctile(irf,upper,2);

figure
plot(1:size(irf,1),irf_lower,'-b','LineWidth',2)
hold on
plot(1:size(irf,1),irf_med,'-r','LineWidth',2)
hold on
plot(1:size(irf,1),irf_upper,'-b','LineWidth',2)
grid on

print -depsc
savefig('irfs_trend')



for dd=1:size(draws,2)
    
   irf(:,dd) = irf_computation_g(draws(:,dd),setup,60) ;
    
end

irf_lower=prctile(irf,lower,2);
irf_med=prctile(irf,med,2);
irf_upper=prctile(irf,upper,2);

figure
plot(1:size(irf,1),irf_lower,'-b','LineWidth',2)
hold on
plot(1:size(irf,1),irf_med,'-r','LineWidth',2)
hold on
plot(1:size(irf,1),irf_upper,'-b','LineWidth',2)
grid on

print -depsc
savefig('irfs_g')


