# face_recognition_matlab
 Face recognition model in Matlab using Sliding Windows technique. HoG Feature Extraction and Support Vector Machine are used to obtain weight matrix.
 
 ![Face recognition model in Matlab using Sliding Windows technique](https://github.com/minhvnhat2711/face_recognition_matlab/blob/main/sample.jpg?raw=true)
 
 Library used: [Vlfeat-0.9.21](https://www.vlfeat.org/ "Vlfeat") 
 
 
 ## How to run
 ### Install [Vlfeat-0.9.21](https://www.vlfeat.org/ "Vlfeat") in the main directory
 
 ### Run `main.m`
 
 ## Flow
 
 ### Split training data
 ```
 % number of validation images (per label)
num_validation = floor(pos_nImages / 5);
num_training = pos_nImages - num_validation;

% shuffle
shuffl = randperm(pos_nImages);

% indexes of images chosen to be in validation set
validation_indexes = shuffl(1:num_validation);

% indexes of images chosen to be in training set
training_indexes = shuffl(num_validation+1:end);

% compile into sets
validation_neg_set = neg_feats(validation_indexes, :);
validation_pos_set = pos_feats(validation_indexes, :);
training_neg_set = neg_feats(training_indexes, :);
training_pos_set = pos_feats(training_indexes, :);
```

### Train SVM Model, including hyper-parameter searching
```
lambdas = [0 linspace(0.01, 0.09, 9) linspace(0.1, 2, 20) linspace(2.2, 3, 5)];
for i=1:size(lambdas,2)
    lambda = lambdas(i);
    [w,b] = vl_svmtrain(feats',labels',lambda);
    confidences = [training_pos_set; training_neg_set]*w + b;

    [tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy(confidences, labels);
    
    correct_classification = sign(confidences .* labels);
    accuracy = 1 - sum(correct_classification <= 0)/length(correct_classification);
    accs(i) = accuracy;
end
```

### Test the SVM Model on the validation set
```
feats_validate = cat(1,validation_pos_set,validation_neg_set);
labels_validate = cat(1,ones(num_validation,1),-1*ones(num_validation,1));
```

### Run Object detection using Sliding Windows Technique
 
```
[rows,cols,~] = size(feats);    
confs = zeros(rows,cols);
for r=1:rows-5
    for c=1:cols-5
    % create feature vector for the current window and classify it using the SVM model, 
    % take dot product between feature vector and w and add b,
    window = feats(r:r+5, c:c+5,:);
    window_feat = window(:)';
    conf = window_feat*w + b;
    % store the result in the matrix of confidence scores confs(r,c)
    confs(r, c) = conf;
    end
end
```

For better accuracy, also employ different scalings and non-max suppression.
