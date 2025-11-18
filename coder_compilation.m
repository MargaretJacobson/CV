addpath(genpath('functions'))
setup_daily;
data = setup.data;

%get parameter matrices to set size

wrapper_func=str2func(setup.wrapper);
temp_param=rand(setup.length_param_vector,1);
[A, B, C ,D ,add_matrices]=wrapper_func(temp_param,setup,data);

%[ llk, xest] = TVKF(A,B,C,D,x10,S10,Y)

% coder types - SMC part
AType = coder.typeof(A, [size(A,1) size(A,2)  size(A,3)], [false false false]);
BType = coder.typeof(B, [size(B,1) size(B,2)  size(B,3)], [false false false]);
CType = coder.typeof(C, [size(C,1) size(C,2)  size(C,3)], [false false false]);
DType = coder.typeof(D, [size(D,1) size(D,2)  size(D,3)], [false false false]);
x10Type = coder.typeof(setup.state_initial, [size(setup.state_initial,1) size(setup.state_initial,2)], [false false]);
S10Type = coder.typeof(setup.cov_initial, [size(setup.cov_initial,1) size(setup.cov_initial,2)], [false false]);
dataType = coder.typeof(data, [size(data,1) size(data,2)], [false false]);

setupType = coder.typeof(setup, [size(setup,1) size(setup,2)], [false false]);
paramsType = coder.typeof(temp_param, [size(temp_param,1) size(temp_param,2)], [false false]);




cd functions/estimation_code_SMC
clear mex
codegen TVKF_org -args {AType,BType,CType,DType,x10Type,S10Type,dataType} -o TVKF  -report
codegen system_matrices -args {paramsType,setupType,dataType} -o system_matrices_coder  -report


% 
tic
for jj=1:100
    
   [ llk, xest] = TVKF_org(A,B,C,D,setup.state_initial,setup.cov_initial,data);
    
end

toc

tic
for jj=1:100
    
   [ llk2, xest2] = TVKF(A,B,C,D,setup.state_initial,setup.cov_initial,data);

    
end
toc


tic
for jj=1:100
    
   [A, B, C ,D ,add_matrices]=system_matrices(temp_param,setup,data);
    
end

toc

tic
for jj=1:100
    
      [A, B, C ,D ,add_matrices]=system_matrices_coder(temp_param,setup,data);

    
end
toc


