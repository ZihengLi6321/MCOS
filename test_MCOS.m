function [NMAE RMSE L weighted] = test_MCOS(para,lambda)%  (M,m,n,Omega,k,num)

% parameters

M = para.X;
W = para.W;
r = para.r;
size = para.size;
m = size(1);
n = size(2);
Omega = para.Omega;
data = para.data;
Test_ind = para.test.Ind;
Test_values = para.test.values;
dif = para.dif;

% parameters for algorithm
% test algorithms
[out] = MCOS(M,W,r,500,lambda);
L = out.U*out.V'+out.b*ones(1,n);
num = para.out_num;
L = L(:,1:n-num);
obj = out.re;
weighted = out.weight;
%% TEST
Rvec = L(Test_ind)-Test_values;
NMAE = mean(abs(Rvec))/dif;
RMSE = sqrt(mean(Rvec.*Rvec));
