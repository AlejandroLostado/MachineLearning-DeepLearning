function [Predict_Y,w1,w2,b1,b2,tf,W1_alp,W2_gam,alpha,gamma] = TWSVM_cvx(TestX,DataTrain,FunPara)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TWSVM: Twin Support Vector Machine 
%
%       Predict_Y =TWSVM_cvx(TestX,DataTrain,FunPara)
% 
%       Input:
%               TestX       - Test Data matrix. 
%                             Each row vector of fea is a data point.
%
%               DataTrain   - Struct value in Matlab------Training data.
%                   DataTrain.A: Positive input of Data matrix.
%                   DataTrain.B: Negative input of Data matrix.
%
%               FunPara - Struct value in Matlab. The fields in options
%                         that can be set:
%                   c1: [0,inf] Paramter to tune the weight. 
%                   c2: [0,inf] Paramter to tune the weight. 
%                   c3: [0,inf] Paramter to tune the weight. 
%                   c4: [0,inf] Paramter to tune the weight. 
%                   kerfPara:Kernel parameters. See kernelfun.m.
%
%       Output:
%               Predict_Y - Predict value of the TestX.
%               w1,w2,b1,b2 - 
%
%       Examples:
%
%           DataTrain.A = rand(50,10);
%           DataTrain.B = rand(60,10);
%           TestX=rand(20,10);
%           FunPara.c1=0.1;
%           FunPara.c2=0.1;
%           FunPara.c3=0.1;
%           FunPara.c4=0.1;
%           FunPara.kerfPara.type = 'lin';
%           Predict_Y =TWSVM_cvx(TestX,DataTrain,FunPara);
% 
%Reference:
%   Y.-H. Shao, C.-H. Chun, X.-B. Wang, N.-Y. Deng.Improvements on Twin 
%    Support Vector Machines.IEEE Transactions on Neural Networks, 2011, 22
%   (6):962-968.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initailization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%tic;
Xpos = DataTrain.A;
Xneg = DataTrain.B;
cpos = FunPara.c1;
cneg = FunPara.c2;
eps1 = FunPara.c3;
eps2 = FunPara.c4;
kerfPara = FunPara.kerfPara;
m1=size(Xpos,1);
m2=size(Xneg,1);
e1=ones(m1,1);
e2=ones(m2,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute Kernel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(kerfPara.type,'lin')
    H=[Xpos,e1];
    G=[Xneg,e2];
else
    X=[DataTrain.A;DataTrain.B];
    H=[kernelfun(Xpos,kerfPara,X),e1];
    G=[kernelfun(Xneg,kerfPara,X),e2];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Compute (w1,b1) and (w2,b2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%DTWSVM1
HH=H'*H;
HH = HH + eps1*eye(size(HH));%regularization
HHG = HH\G';
kerG1=G*HHG;
kerG1=(kerG1+kerG1')/2;

t0=cputime;
cvx_begin
cvx_quiet true
cvx_precision('low')
cvx_solver sedumi
variables s2(m2)
minimize(0.5*quad_form(s2,kerG1)-sum(s2))
subject to
0<=s2<=cpos;
cvx_end
tf1=cputime-t0;
vpos=-HHG*s2;

%%%%DTWSVM2 
QQ=G'*G;
QQ=QQ + eps2*eye(size(QQ));%regularization
QQP=QQ\H';
kerH1=H*QQP;
kerH1=(kerH1+kerH1')/2;

t0=cputime;
cvx_begin
cvx_quiet true
cvx_precision('low')
cvx_solver sedumi
variables s1(m1)
minimize(0.5*quad_form(s1,kerH1)-sum(s1))
subject to
0<=s1<=cneg;
cvx_end
tf=tf1+cputime-t0;
vneg=QQP*s1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute W1(alpha), W2(gamma) for RFE in nonlinear case
%%%%%%%
if strcmp(kerfPara.type,'rbf')
    W1_alp=-s2'*G*vpos;
    W2_gam=s1'*H*vneg;
end
clear kerH1 kerG1 H G HH HHG QQ QQP;

w1=vpos(1:(length(vpos)-1));
b1=vpos(length(vpos));
w2=vneg(1:(length(vneg)-1));
b2=vneg(length(vneg));
%toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Predict and output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m=size(TestX,1);
if strcmp(kerfPara.type,'lin')
    H=TestX;
    w11=sqrt(w1'*w1);
    w22=sqrt(w2'*w2);
    y1=H*w1+b1*ones(m,1);
    y2=H*w2+b2*ones(m,1);
else
    C=[DataTrain.A;DataTrain.B];
    H=kernelfun(TestX,kerfPara,C);
    w11=sqrt(w1'*kernelfun(X,kerfPara,C)*w1);
    w22=sqrt(w2'*kernelfun(X,kerfPara,C)*w2);
    y1=H*w1+b1*ones(m,1);
    y2=H*w2+b2*ones(m,1);
end
wp=sqrt(2+2*w1'*w2/(w11*w22));
wm=sqrt(2-2*w1'*w2/(w11*w22));
clear H; clear C;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m1=y1/w11;
m2=y2/w22;
MP=(m1+m2)/wp;
MN=(m1-m2)/wm;
mind=min(abs(MP),abs(MN));
maxd=max(abs(MP),abs(MN));
Predict_Y = sign(abs(m2)-abs(m1));
end