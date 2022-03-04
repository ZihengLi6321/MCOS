function Msub = random_sampling(M, SR);
%
% Generating data for training
% For details, see: 
% Kim-Chuan Toh and Sangwoon Yun. 
% An accelerated proximal gradient algorithm for nuclear norm regularized least squares problems,
% Pacific J. Optimization, 6 (2010), pp. 615-640.

normM = sum(M.*M);
idx = find(normM == 0);
M = M(:,setdiff([1:size(M,2)],idx));  %%% removed zero columns.
fprintf('\n***** removed %2.1d zero columns in M *****\n',length(idx));
Mt = M';
[nr, nc] = size(M);
nnzM = nnz(M);
NumElement = 10;

if (NumElement > 0)
    rr = zeros(nnzM,1);
    cc = zeros(nnzM,1);
    vv = zeros(nnzM,1);
    count = 0;
    for i=1:nr
        rand('state',i);
        Mrow = full(Mt(:,i))';
        truerow = find( abs(Mrow) > 1e-13);
        len = length(truerow);
        rp = randperm(len);
        NumPerRow = min(max(NumElement,floor(len*SR)),len);
        chrow = truerow(rp(1:NumPerRow));
        idx = [1:NumPerRow];
        rr(count+idx) = i*ones(NumPerRow,1);
        cc(count+idx) = chrow;
        vv(count+idx) = Mrow(chrow);
        count = count + NumPerRow;
    end
    rr = rr(1:count); cc = cc(1:count); vv = vv(1:count);
    Msub = spconvert([rr,cc,vv; nr,nc,0]);
    colnorm = sum(Msub.*Msub);
    colnormidx = find(colnorm == 0);
        
    if ~isempty(colnormidx)
        rr = zeros(nnzM,1);
        cc = zeros(nnzM,1);
        vv = zeros(nnzM,1);
        count = 0;
        for j = 1:length(colnormidx)
            rand('state',j);
            jcol = colnormidx(j);
            Mcol = full(M(:,jcol));
            truecol = find( abs(Mcol) > 1e-13 );
            len = length(truecol);
            rp = randperm(len);
            NumPerCol = min(max(NumElement,floor(len*SR)),len);
            chcol = truecol(rp(1: NumPerCol));
            idx = [1: NumPerCol];
            rr(count+idx) = chcol;
            cc(count+idx) = jcol*ones(NumPerCol,1);
            vv(count+idx) = Mcol(chcol);
            count = count + NumPerCol;
        end
        brr = rr(1:count); cc = cc(1:count); vv = vv(1:count);
        Msub = Msub + spconvert([brr,cc,vv; nr,nc,0]);
    end
end
end

