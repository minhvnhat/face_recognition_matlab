% you might want to have as many negative examples as positive examples
n_have = 0;
n_want = numel(dir('cropped_training_images_faces/*.jpg'));

imageDir = 'images_notfaces';
% dir of imageDir/*.jpg
imageList = dir(sprintf('%s/*.jpg',imageDir));
% number of "images_notfaces"
nImages = length(imageList);

% create new directory
new_imageDir = 'cropped_training_images_notfaces';
mkdir(new_imageDir);

dim = 36;

new_imageBasename = 'crop_noface';

while n_have < n_want
    i = mod(n_have, nImages) + 1;
    n_have = n_have + 1;
    % generate random 36x36 crops from the non-face images
    im2 = im2gray(imread(sprintf('%s/%s',imageDir,imageList(i).name)));
    % randomly pick new center of the crop
    x_center = randi(size(im2,1) - dim);
    y_center = randi(size(im2,2) - dim);
    % taking a crop
    crop = im2(x_center : x_center+dim-1, y_center : y_center+dim-1);
    % write into specified folder
    imwrite(crop, sprintf("%s/%s_%d.tiff", new_imageDir, new_imageBasename, n_have));
end

