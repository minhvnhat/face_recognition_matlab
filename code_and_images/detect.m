load('my_svm.mat');

imageDir = 'test_images';
imageList = dir(sprintf('%s/*.jpg',imageDir));
nImages = length(imageList);

bboxes = zeros(0,4);
confidences = zeros(0,1);
image_names = cell(0,1);

cellSize = 6;
dim = 36;
% create multiple scales 0.1, 0.2, ..., 1
scales = linspace(0.1, 1, 10);

for i=1:nImages
    % initialize
    bound_box_main = zeros(0,4);
    confs_main = zeros(0,1);
    
    % load and show the image
    im = im2single(imread(sprintf('%s/%s',imageDir,imageList(i).name)));
    
    
%     imshow(im);
%     hold on;
    
    for j=1:size(scales, 2)
        scale = scales(j);
        pixel_cell_size = cellSize/scale;
        im_resize = imresize(im, scale);
        
        if size(im_resize, 1) < 36 || size(im_resize, 2) < 36
            continue
        end
        
        % generate a grid of features across the entire image. you may want to 
        % try generating features more densely (i.e., not in a grid)
        feats = vl_hog(im_resize,cellSize);

        % concatenate the features into 6x6 bins, and classify them (as if they
        % represent 36x36-pixel faces)
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
        % get the most confident predictions for this scale
        [picked_confs,inds] = sort(confs(:),'descend');
        inds = inds(1:20); % (use a bigger number for better recall)
        
        [rowx,colx] = ind2sub([size(feats,1) size(feats,2)],inds);   
        
        % need to store confs and their bound_box for each scale
        bound_box = [ ((colx-1)*pixel_cell_size) ...
         ((rowx-1)*pixel_cell_size) ...
        (colx+cellSize-1)*pixel_cell_size ...
        (rowx+cellSize-1)*pixel_cell_size];
        bound_box_main = [bound_box_main; bound_box];
        confs_main = [confs_main; picked_confs(1:20)];
    end
       
    % get the most confident predictions 
    [picked_confs,inds] = sort(confs_main(:),'descend');
    inds = inds(1:20); % (use a bigger number for better recall)
    
    % Apply non-max here
    % Utilize confs, inds, feats
    % O(n^2), compare each pair of prediction and get rid of the
    % overlapping 
    [rowx, colx] = ind2sub([size(feats,1) size(feats,2)],inds);
    % bound_boxes = [x_min, y_min, x_max, y_max]
    bound_boxes = bound_box_main(inds, :);
    % array contains marking for delete
    delete_mark = zeros(size(inds));
    for m=1:numel(inds)
        for n=1:numel(inds)
            if m ~= n
                % take the bound of the prediction box
                bb = bound_boxes(m, :);
                bbgt = bound_boxes(n, :);

                % calculate overlap area
                bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
                iw=bi(3)-bi(1);
                ih=bi(4)-bi(2);
                % check if there is overlap
                if iw>0 && ih>0       
                    % compute overlap as area by 
                    % taking the intersect area, divide by area of the current box
                    % then overlapping occurs when more than half of the current
                    % box is overlapped
                    overlap_area = iw*ih;
                    area_m = (bb(3)-bb(1))*(bb(4)-bb(2));
                    ov = overlap_area / area_m;
                    % check if overlap too much
                    if ov >= 0.5
                        curr_conf = picked_confs(m);
                        compared_conf = picked_confs(n);
                        if curr_conf < compared_conf
                            % delete
                            delete_mark(m) = 1;
                        else
                            % delete the other one
                            delete_mark(n) = 1;
                        end
                    end
                end
            end
        end
    end   
    inds = inds(delete_mark == 0);
    
    for n=1:numel(inds)        
        % convert index (ind) -> subscript (x, y)
        % this line converts inds to position in 'feats'
        [row,col] = ind2sub([size(feats,1) size(feats,2)],inds(n));
        % bbox: [left-col bottom-row right-col top-row]

%         bbox = [ ((col-1)*cellSize) ...
%              ((row-1)*cellSize) ...
%             (col+cellSize-1)*cellSize ...
%             (row+cellSize-1)*cellSize];
        bbox = bound_box_main(inds(n),:);

        % access confident value
        conf = confs(row,col);

        image_name = {imageList(i).name};

        % plot
        plot_rectangle = [bbox(1), bbox(2); ...
            bbox(1), bbox(4); ...
            bbox(3), bbox(4); ...
            bbox(3), bbox(2); ...
            bbox(1), bbox(2)];
        plot(plot_rectangle(:,1), plot_rectangle(:,2), 'g-');

        % save         
        bboxes = [bboxes; bbox];
        confidences = [confidences; conf];
        image_names = [image_names; image_name];
    end
%     pause;
    fprintf('got preds for image %d/%d\n', i,nImages);
end

% evaluate
label_path = 'test_images_gt.txt';
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections_on_test(bboxes, confidences, image_names, label_path);
