run('./vlfeat-0.9.21/toolbox/vl_setup')
load('split_training.mat')

feats = cat(1,training_pos_set,training_neg_set);
labels = cat(1,ones(num_training,1),-1*ones(num_training,1));

lambdas = [0 linspace(0.01, 0.09, 9) linspace(0.1, 2, 20) linspace(2.2, 3, 5)];
% lambdas = 0.1;
accs = zeros(size(lambdas,1), 1);

for i=1:size(lambdas,2)
    lambda = lambdas(i);
    [w,b] = vl_svmtrain(feats',labels',lambda);
    fprintf('Lambda: %.4f\n', lambda);
    fprintf('Classifier performance on train data:\n')
    confidences = [training_pos_set; training_neg_set]*w + b;

    [tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy(confidences, labels);
    
    correct_classification = sign(confidences .* labels);
    accuracy = 1 - sum(correct_classification <= 0)/length(correct_classification);
    accs(i) = accuracy;
end

plot(lambdas, accs);
title('Accuracy with different values of lambda');

% find the best lambda
[best_acc, best_lambda] = max(accs);
best_lambda = lambdas(best_lambda);
fprintf('Best lambda: %.4f, Accuracy on training: %.4f\n', best_lambda, best_acc);
% train using best lambda
[w,b] = vl_svmtrain(feats',labels',best_lambda);