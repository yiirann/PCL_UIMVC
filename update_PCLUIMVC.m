function [Y,obj] = update_SCPUIMVC(X,W,Lg,LM,Ind_ms,lambda1,lambda2,lambda3,max_iter,dim)
%X mxn 缺失位置为0
%W 构造EW的W，nixn
%Lv E'LgE中的
%Ls2 PLP中完整的
%Ind_ms 每个视图中缺失的序号 
mu = 1e-3;
rho = 1.1;
maxmu=10^8;
Temp2 = 0;
for iv = 1:length(X)
    %--------初始E m*nv----------%
    E{iv} = rand(size(X{iv},1),size(W{iv},1));
    %--------初始化P--------%
    options = [];
    options.ReducedDim = dim;
    Z{iv} = X{iv}+E{iv}*W{iv};
    [P1,~] = PCA1(Z{iv}', options);
    Piv{iv} = P1';
   Temp2 = Temp2+(eye(size(LM{iv},1))+lambda2*LM{iv});
   Y1{iv} = zeros(size(X{iv}));
end
invtemp2 = inv(Temp2);
numInst = size(X{1},2);
Y = rand(dim,numInst); 
clear Z{iv}
for iter = 1:max_iter
    % ------------- Y ------------- %
    Temp1 = 0;
    for iv = 1:length(X)
        Temp1 = Temp1+Piv{iv}*(X{iv}+E{iv}*W{iv});  
    end  
    Y = Temp1 * invtemp2;
    Y(isnan(Y)) = 0;
    Y(isinf(Y)) = 1e5;
    
    for iv = 1:length(X)
        % ------------ Piv -------------- %
        Z{iv} = X{iv}+E{iv}*W{iv};
        linshi_St = Z{iv}*Z{iv}'+lambda3*eye(size(Z{iv},1));
        St2{iv} = mpower(linshi_St,-0.5);
        St3{iv} = St2{iv}*Z{iv};
        
        linshi_M = St3{iv}*Y';%论文中的H
        linshi_M(isnan(linshi_M)) = 0;
        linshi_M(isinf(linshi_M)) = 0;
        [linshi_U,~,linshi_V] = svd(linshi_M','econ');
        linshi_U(isnan(linshi_U)) = 0;
        linshi_U(isinf(linshi_U)) = 0;
        linshi_V(isnan(linshi_V)) = 0;
        linshi_V(isinf(linshi_V)) = 0;
        Piv{iv} = linshi_U*linshi_V'*St2{iv};

        % ------- E{v} ------- %
        linshi  = Piv{iv}'*Y;
        linshi1 = linshi(:,Ind_ms{iv});%Ind_ms{iv}是第v个视图中缺失样本的序号
        tepB =  Z{iv} - X{iv} + Y1{iv}/mu;
        linshi2 = tepB(:,Ind_ms{iv});
        linshi3 = 2*linshi1+mu*linshi2;
        tepE = 2*lambda1*Lg{iv}+2*Piv{iv}'*Piv{iv}+mu*eye(size(Lg{iv},1));
        E{iv} = tepE\linshi3;
        %---------- Update Y1^v--------------%
        Y1{iv} = Y1{iv} + mu*(Z{iv}-X{iv}-E{iv}*W{iv});
    end
     % -------------- obj --------------- %
    linshi_obj = 0;
    for iv = 1:length(X)
        linshi_obj = linshi_obj+norm(Piv{iv}*(X{iv}+E{iv}*W{iv})-Y,'fro')^2+lambda1*trace(E{iv}'*Lg{iv}*E{iv})+lambda2*trace(Y*LM{iv}*Y')+lambda3*norm(Piv{iv},'fro')^2;        
    end
    obj(iter) = linshi_obj;
    fprintf('iter = %d, obj = %g\n', iter, obj(iter));
    %---------- Update mu--------------%
     mu = min(maxmu,mu*rho);
     %------条件--------%
    if iter >3 && abs(obj(iter)-obj(iter-1))<1e-6
        %iter
        break;
    end
end
    
end

