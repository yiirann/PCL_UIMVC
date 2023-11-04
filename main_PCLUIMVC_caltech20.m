clear;
clc

Dataname = 'Caltech101-20';
per=0.1;

lambda1  = 10;
lambda2  = 1;
lambda3 = 1e-6;
% load multi-view dataset
load(Dataname);
truthF = Y;
X = X';
clear categories cateset datasetname feanames lenSmp Y
num_cluster = length(unique(truthF));
num_instance=length(truthF);
num_view = length(X);
for i = 1:num_view
    X{i} = X{i}';
end
%average missing rate
%per=0.4;
%the missing rate of each view
perDel=[0.25*per 0.5*per 0.75*per per 1.5*per 2*per];
for i = 1:num_view
    folds(:,i) = randsrc(num_instance,1,[0 1; perDel(i) (1-perDel(i))]);
end
ind_folds = folds;
sum_miu=sum(sum(folds));


for iv = 1:num_view
    X1 = X{iv}';%X1 nxm
    X1 = NormalizeFea(X1,1);
    X2 = X{iv};%X2 mXn
    X2 = NormalizeFea(X2,0);
    
    ind_0 = find(ind_folds(:,iv) == 0);
    X2(:,ind_0) = 0 ;%X2是缺失的列为0 mxn
    XX{iv} = X2;
    
    X1(ind_0,:) = [];
    Y{iv} = X1'; %Y是去掉缺失样本的 mxne
    %补全图A用的M
    %WW{iv} = diag(ind_folds(:,iv));
    W1 = eye(size(ind_folds,1));
    W1(ind_0,:) = [];
    M{iv} = W1;
    %构造缺失部分EW
    Win1 = eye(num_instance);
    ind_1 = find(ind_folds(:,iv) == 1);
    Win1(ind_1,:) = [];%去除样本存在的那一行
    Win{iv} = Win1;
    Ind_ms{iv} = ind_0;%每个视图缺失的样本序号
end
clear sum_miu
clear X X1 W1 X2
X = Y;
XX2 = XX; 
clear Y XX
    
for iv = 1:num_view
        options = [];
        options.NeighborMode = 'KNN';
        options.k = 7;
        options.WeightMode = 'Binary';
        A1 = constructW(X{iv}',options);
        %Initialize A
        A_ini{iv} = full(A1);
        clear A1;
 
end
for iv = 1:num_view
    WA{iv} = (abs(M{iv}'*A_ini{iv}*M{iv})+abs(M{iv}'*A_ini{iv}'*M{iv}))/2;
    DA{iv} = diag(sum(WA{iv},2));
    LM{iv} = DA{iv}-WA{iv};
end
    
    neighbor = 7;
  for iv = 1:length(XX2)
        options = [];
        options.NeighborMode = 'KNN';
        options.k = neighbor;
        options.WeightMode = 'Binary';      % Binary  HeatKernel
        Z1 = full(constructW(XX2{iv},options));%X是m*n
        Z1 = (Z1+Z1')/2;
        Lg{iv} = diag(sum(Z1,2))-Z1;
        clear Z1;
  end
    
fid = fopen('result/PCLUIMVC_caltech20.txt','a');
replic = 1;

% 指标
AC_ = zeros(1, replic);
NMI_ = zeros(1, replic);
purity_ = zeros(1, replic);
Fscore_ = zeros(1, replic);
Precision_ = zeros(1, replic);
Recall_ = zeros(1, replic);
AR_ = zeros(1, replic);

for i = 1: replic      
    max_iter = 40;
    dim = num_cluster;
    [P,obj] = update_PCLUIMVC(XX2,Win,Lg,LM,Ind_ms,lambda1,lambda2,lambda3,max_iter,dim);
    P(isnan(P)) = 0;
    P(isinf(P)) = 1e5;
    new_F = P';
    norm_mat = repmat(sqrt(sum(new_F.*new_F,2)),1,size(new_F,2));
    % avoid divide by zero
    for ii = 1:size(norm_mat,1)
        if (norm_mat(ii,1)==0)
            norm_mat(ii,:) = 1;
        end
    end
    new_F = new_F./norm_mat;
    %rand('seed',230);
    pre_labels  = kmeans(real(new_F),num_cluster,'emptyaction','singleton','replicates',20,'display','off');
    result = EvaluationMetrics(truthF,pre_labels);
    AC_(i) = result(1)*100;
    NMI_(i) = result(2)*100;
    purity_(i) = result(3)*100;
    Fscore_(i) = result(4)*100;
    Precision_(i) = result(5)*100;
    Recall_(i) = result(6)*100;
    AR_(i) = result(7)*100;
end

% 求每个指标均值和方差
AC(1) = mean(AC_); AC(2) = std(AC_);
NMI(1) = mean(NMI_); NMI(2) = std(NMI_);
purity(1) = mean(purity_); purity(2) = std(purity_);
Fscore(1) = mean(Fscore_); Fscore(2) = std(Fscore_);
Precision(1) = mean(Precision_); Precision(2) = std(Precision_);
Recall(1) = mean(Recall_); Recall(2) = std(Recall_);
AR(1) = mean(AR_); AR(2) = std(AR_);
fprintf(fid, "per = %g,lambda1 = %g,lambda2 = %g,lambda3 = %g\n", per,lambda1,lambda2,lambda3);
fprintf(fid, "AC = %5.4f + %5.4f, NMI = %5.4f + %5.4f, purity = %5.4f + %5.4f\nFscore = %5.4f + %5.4f, Precision = %5.4f + %5.4f, Recall = %5.4f + %5.4f, AR = %5.4f + %5.4f\n",...
    AC(1), AC(2), NMI(1), NMI(2), purity(1), purity(2), Fscore(1), Fscore(2), Precision(1), Precision(2), Recall(1), Recall(2), AR(1), AR(2));
fprintf(fid,'********************************\n');
fclose(fid);
fprintf("%5f",AC(1));
% 画目标函数值收敛图
plot(obj); 
   

