clc;
close all;
rng(304)
%% Design matrix generation
N=20;                         %rows
M=40;                         %columns
phi=normrnd(0,1,[N,M]);       %each element from N(0,1)
%% Sparse vector generation
d_0=7;                        %no of non-zero elements in sparse vector
b=1:1:40;
i=sort(randsample(b,d_0));    %row position of non zero elements of sparse vector
j=ones(1,d_0);                %column position of non zero elements of sparse vector
v=normrnd(0,1,[1,d_0]);       %values of non zero elements of sparse vector in its coressponding row and column position
w=sparse(i,j,v,M,1);          %sparse vector generation
%% Noise variances
noise_dB=[-20 -15 -10 -5 0];  %variances
noise=10.^((noise_dB)./10);   
%% SBL alogorithm
posterior_mean=zeros(M,100,length(noise));
w_map=zeros(M,length(noise));
NMSE=zeros(1,length(noise));
alpha_updated=zeros(M, 100);
logL=zeros(1,100);
M=zeros(1,length(noise)); 
I=zeros(1,length(noise));
for k=1:length(noise)                                  %iteration over noise
    n=normrnd(0,sqrt(noise(1,k)),[N,1]);
    t=phi*w+n;                                         %generated data 
    %initial prior(assumed)
    prior_cov=eye(40);                                 %initial prior covariance matrix(assumed)
    beta=1/noise(1,k);                                 %initial beta values 
    beta_inv=1/beta;
    A=inv(prior_cov);                                  %diagonal matrix with alphas
    for j=1:100                                        %loop for estimation map value,updating alphas
        posterior_cov=pinv(beta*(phi'*phi)+A);         %posterior covariance matrix
        posterior_mean(:,j)=beta*posterior_cov*phi'*t; %posterior mean vector
        gamma=A*diag(posterior_cov);              
        post_mean_square=posterior_mean(:,j).^2;  
        for i=1:length(gamma)                
            alpha_updated(i,j)=(1-gamma(i,1))/post_mean_square(i,1);    %alpha_updated=1-gamma(i)/(posterior_mean(i)).^2
        end
        beta_new_inv=sum((t-phi*posterior_mean(:,j)).^2)/sum(gamma);    %beta_updated_=norm(t-phi*map)^2/sum(gammas)
        A_new=diag(alpha_updated(:,j));                                 %new diagonal matrix with updated alphas
        S=(beta_new_inv*eye(20))+phi*(A_new\phi');
        logL(1,j)=-(1/2)*(log(det(S))+t'*(S\t));                        %log of marginal likelihood
        A=A_new;                                                        %assiging A with the updated A
        beta=1/beta_new_inv;                                            %assiging beta with the updated beta
    end
    [M(k),I(k)]=max(logL);                                              %max of log marginal likelihood and its index
    w_map(:,k)=posterior_mean(:,I(k));                                  
    NMSE(1,k)=sum((w_map(:,k)-w).^2)/sum(w.^2);                         %normalized mean square error of map estimate
end
%% Plot Noise-dB vs NMSE 
figure;
semilogy(noise_dB, NMSE, 'o-', 'LineWidth', 2);
xlabel('Noise variance dB');
ylabel('NMSE')
title('NMSE v/s Noise variance')
grid on;