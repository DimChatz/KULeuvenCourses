X = (-3:0.01:3);
X=X';
Y = sinc(X) + 0.1.*randn(length(X), 1);  % Ensure Y is a column vector

Xtrain = X(1:2:end);   % Ensure this is a column vector (301x1 double)
Ytrain = Y(1:2:end);    % Corresponding Y values (301x1 double)

Xtest = X(2:2:end);    % Ensure this is a column vector (300x1 double)
Ytest = Y(2:2:end);     % Corresponding Y values (300x1 double)

gam = [10,10^3,10^6];
sig2 = [0.01,1,100];
for i=1:3
    for j=1:3
        [alpha, b] = trainlssvm({Xtrain, Ytrain, 'f', gam(i), sig2(j), 'RBF_kernel'});% Plot the results and get the struct of handles
        h = plotlssvm({Xtrain, Ytrain, 'f', gam(i), sig2(j), 'RBF_kernel', 'preprocess'}, {alpha, b});
        hold on; 
        scatter(Xtest, Ytest, 'Marker', '.');
        plot(min(X):.1:max(X),sinc(min(X):.1:max(X)),'c');
        Ypred = simlssvm({Xtrain, Ytrain, 'f', gam(i), sig2(j), 'RBF_kernel'}, {alpha, b}, Xtest);
        mse = mean((Ytest - Ypred).^2);
        title(sprintf('RBF: C = 10^{%d}, sig2 = %.2f, MSE = %.4f', log10(gam(i)), sig2(j), mse));
        legend('show', 'Estimated Function', 'Train Data', 'Test Data', 'True Function', 'Location', 'northwest');
        hold off;
        grid on;
    end
end