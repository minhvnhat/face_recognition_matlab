function [] = split_data(pos_feats,neg_feats,pos_nImages,neg_nImages)

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

save('split_training.mat', 'training_neg_set','training_pos_set', 'num_training');
save('split_validation.mat', 'validation_neg_set','validation_pos_set', 'num_validation');
end

