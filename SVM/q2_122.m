%sig2 =0.4;
%gam = 10; 
%crit_L1 = bay_lssvm({Xtrain, Ytrain, 'f', gam, sig2}, 1); 
%crit_L2 = bay_lssvm({Xtrain, Ytrain, 'f', gam, sig2}, 2);
%crit_L3 = bay_lssvm({Xtrain, Ytrain, 'f', gam, sig2}, 3);
%[~,alpha,b] = bay_optimize({Xtrain, Ytrain, 'f', gam, sig2}, 1);
%[~,gam] = bay_optimize({Xtrain, Ytrain, 'f', gam, sig2}, 2);
%[~,sig2] = bay_optimize({Xtrain, Ytrain, 'f', gam, sig2}, 3);
%sig2e = bay_errorbar({Xtrain, Ytrain, 'f', gam, sig2}, 'figure');
%sig2 = 0.4;
%gam = 0.4625;
%X = 6.*rand(100, 3)- 3;
%Y = sinc(X(:,1)) + 0.1.*randn(100,1);
%[selected, ranking, costs] = bay_lssvmARD({X, Y, 'f', gam, sig2, 'RBF_kernel','preprocess'});
for i=1:3
    costFun = 'crossvalidatelssvm'; 
    [gam, sig2, cost] = tunelssvm({X(:,i), Y, 'f', [], [], 'RBF_kernel'}, 'simplex', costFun, {10, 'mse';});
    plotlssvm({X(:,i), Y, 'f', gam, sig2, 'RBF_kernel'});
    hold on;
    xlabel('X1');
    hold off;
    grid on;
end

% Example visualization of ARD results
figure;
bar(ranking, sort(ranking, 'descend'));
xlabel('Feature Index');
ylabel('Relevance (Ranking)');
title('Feature Relevance Based on ARD');
xticks(1:3); % Assuming 3 features