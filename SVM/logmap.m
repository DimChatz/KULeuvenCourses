Ztot = cat(1, Z, Ztest);
order = [5, 50, 75, 100, 150, 200, 250, 10, 25];
for i=1:9
    X = windowize(Z, 1:(order(i) + 1));
    Y = X(:, end);
    X = X(:, 1:order(i));

    Xtest = windowize(Ztot, 1:(order(i) + 1));
    Xtest = Xtest(end-200:end,:);
    Ytest = Xtest(:, end);
    Xtest = Xtest(:, 1:order(i));

    c = tspartition(1000-order(i),"ExpandingWindow",5);
    trainSize = c.TrainSize(5);
    testSize = c.TestSize(1)-2;
    Xtrain = X(1:trainSize,:);
    Xval = X(trainSize+1:trainSize+2+testSize,:);
    
    Ytrain = Y(1:trainSize,:);
    Yval = Y(trainSize+1:trainSize+2+testSize,:);

    SVMModel = fitrsvm(Xtrain, Ytrain, 'KernelFunction', 'RBF', 'Standardize', true);
    Y_pred = predict(SVMModel, Xval);
    Y_pred2 = predict(SVMModel, Xtest);
    % Calculate the Mean Squared Error (MSE)
    mseVal = mean(abs(Y_pred - Yval).^2);
    mseTest = mean(abs(Y_pred2 - Ytest).^2);

    gamma = SVMModel.BoxConstraints(1);
    sig2 = mean(SVMModel.Sigma).^2;
    % Plot actual vs. predicted values
    figure;
    plot(Ytest, 'b', 'DisplayName', 'Test Data');
    hold on;
    plot(Y_pred2, 'r--', 'DisplayName', 'Predictions');
    legend('show');
    title(sprintf('order=%d with Val MSE=%.0f, Test MSE=%.0f, sig2= %.0f, and C=%.2f',order(i), mseVal, mseTest, sig2, gamma));
    xlabel('Time');
    ylabel('Value');
end
