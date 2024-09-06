order = [5, 150, 200, 250, 50, 75, 100, 10, 25];
for i=1:9
    X = windowize(Z, 1:(order(i) + 1));
    Y = X(:, end);
    X = X(:, 1:order(i));
    Xs = Z(end-order(i)+1:end, 1);
    nb = 200;
    
    model = initlssvm(X, Y, 'f', [], [], 'RBF_kernel');
    wFun = 'whampel';
    costFun = 'rcrossvalidatelssvm'; 
    model = tunelssvm(model, 'simplex', costFun, {10, 'mae';}, wFun);
    model = robustlssvm(model);
    prediction = predict(model, Xs, nb);
    
    gam = model.gam(1,1);
    sig2 = model.kernel_pars;
    mse = mean((Ztest - prediction).^2);
    figure;
    hold on;
    plot(Ztest, 'k');
    plot(prediction, 'r');
    title(sprintf('order = %d, C = %.3f, sig2 = %.3f, MSE = %.3f', order(i), gam, sig2, mse));
    xlabel("Time");
    ylabel("Value");
    legend('show', 'Test Data', 'Predictions', 'Location','southeast');
    hold off;
    grid on;
end
