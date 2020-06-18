%% Test MC on generated data
clc
clear all
m0 = 500;
n0 = 500;
r = 5;

% mxn data matrix M with rank r
U0 = rand(m0,r);
V0 = rand(r,n0);

%ground truth 
M0 = U0*V0; %+ones(m,n);
M = M0;

% adding column outiers

add_outliers = 1;
if add_outliers ==1
    ratio = 0.2;
    num = ceil(n0*ratio);
    O = randn(m0,num)*2;
    W_O = (randn(m0,num)<0.6);
    O = O.*W_O;
    M = [M O];
else
    num = 0;
end
[m,n]=size(M);

% Training data
SR   = 0.7;                             % Sampling ratio
M_train = random_sampling(M, SR);
Omega  = find(M_train);   % 
data = M_train(Omega); 

para.out_num = num;
if num>0   % have outliers, test only inliers
    temp = M - M_train;
    Test_ind = find(temp(:,1:n-num));
    para.test.Ind = Test_ind;
    Tdata = M(:,1:n-num);
    para.test.values = Tdata(Test_ind);   % Test values
else
    Test_ind = find(M - M_train);      % Test ID
    para.test.Ind = Test_ind;
    para.test.values = M(Test_ind);     % Test values
end

[I,J] = ind2sub([m,n],Omega); 
W = sparse(I,J,ones(length(Omega),1),m,n,length(Omega));

% add sparse noise
add_sparse = 0;
if add_sparse ==1
G = double(rand(m,n) > 0.9);
G = G.*W;
M_train = M_train + G;
end

para.Omega = Omega;
para.size = [m,n];
para.data = data;
para.X = M_train;
para.r = r;
para.W = W;
para.M0 = M0;
para.dif = max(data)-min(data);


lambda = 1.5;%0.12 is the value of lambda, if there is no sparse noise, a large lambda will be better, like 1.5
[~, ~, L] = test_MCOS(para,lambda);    
L = L(:,1:n0);
E_re = M0 - L;
re = norm(E_re,'fro')/norm(M0,'fro');






















