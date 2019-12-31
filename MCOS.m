% RPCA  % 2019-03-01
% object function min ||X-E-b1^T-UV||2,1+\lambda||E||1
% RPCA with outliers and sparse noise. 
function out = MCOS(X,N,r,INITER,lambda)
fprintf('Test MCOS \n')
%% Input: 
%   X: data matrix
%   N: observed matrix
%   r : the rank of disered matrix
%   NITER: the number of iteration 
[m,n] = size(X);
z = find(N ~= 0);
nz = find(N == 0);
X(nz) = 0;
% lambda = 0.1;
data = X(z); % the value of known entries 
%%   Start
E = zeros(m,n);
d = ones(n,1);
[U, S, V] = svds(X,r);
V = S*V';
V = V';
re = [];
b = zeros(m,1);
tic
for k = 1:INITER
    C = X-E;
    % update D
    temp_d = sqrt(d);
    D = spdiags(temp_d,0,n,n);
    inD = spdiags(1./temp_d,0,n,n);
    %    
    V_hat = V'*D;
    C_hat = C*D;
    d_s = ones(n,1)'*D; 
   % update b
     b = C_hat * d_s'/(d_s*d_s');           
   % update U 
    temp = C_hat - b*d_s;
    temp1 = temp*V_hat';
    [Us,sigma,Ud] = svd(temp1,'econ');
    U = Us*Ud';
    % update V_deta
    V_hat = U'*(C_hat - b*d_s);
    % update V
    V = V_hat*inD;
    V = V';
    
    % update known data
    X = U*V'+ b*ones(1,n)+E;
    X(z)= data;
       
    % update E
     tempX = X-b*ones(1,n)-U*V';
    % E(z) = max(temp(z)-lambda/2, 0) + min(temp(z)+lambda/2, 0);
    for i = 1:n
     EX(:,i) = max(tempX(:,i)-lambda/(2*d(i))*ones(m,1), 0) + min(tempX(:,i)+lambda/(2*d(i))*ones(m,1), 0);   
    end
    E(z) = EX(z);
    E(nz) = 0;
    
    % update D
    tempE = (X-U*V'-b*ones(1,n)-E);
    Bi = sqrt(sum(tempE.*tempE,1)+eps)';
    d = 0.5./(Bi);  
    
    obj =lambda*sum(sum(abs(E)))+sum(sqrt(sum(tempE.*tempE,1)));
    
    if k>2
    if abs(obj - re(k-1))/re(k-1) < 1e-8
       break;
    end
    end
    re(k) = obj;
    %display
    if mod(k,50)==0
    display(strcat('In the ',num2str(k), '-th iteration'));
    end
end

t1 = toc;
display(strcat('the time of iteration is£º',num2str(t1),'s'));
out.matrix = X;
out.E = E;
out.U = U;
out.V = V;
out.b = b;
out.re = re;
[~,index] = sort(Bi);
out.index = index;
out.weight = d;

