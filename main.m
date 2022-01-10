run('./vlfeat-0.9.21/toolbox/vl_setup')
% Step 1: Generate randomly cropped image for negative training data
run('generate_cropped_notfaces.m');

% Step 2: Run HOG on both training and validation images
run('get_features.m');
load('pos_neg_feats.mat');

% Step 3: Randomly split training and validation images
split_data(pos_feats,neg_feats,pos_nImages,neg_nImages)
load('split_training.mat');
load('split_validation.mat');

% Step 4: Train SVM using training images
run('train_svm.m');
save('my_svm.mat', 'w', 'b');

% Step 5: Test SVM on the validation set
feats_validate = cat(1,validation_pos_set,validation_neg_set);
labels_validate = cat(1,ones(num_validation,1),-1*ones(num_validation,1));

fprintf('Classifier performance on validate data:\n')
confidences = feats_validate*w + b;

[tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy(confidences, labels_validate);

% Step 6: Run detection
run('detect.m');