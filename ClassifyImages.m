function conf_matrix = ClassifyImages(folder,C,training,group)

fprintf("Starting classification...\n");

% Initialize data
k = 3; % Used for knnclassify
folder_count = 1;
image_count = 1;
num_classes = size(unique(group),1);
classranges = 1:num_classes;
conf_matrix = zeros(num_classes, num_classes);
binranges = 1:size(C,2);
samples = zeros(100,size(C,2)); % 100 images in each folder

% Fit knn classifier
model = fitcknn(training, group, 'NumNeighbors', k);

% Transpose C for usage in knnsearch
C = C';

% Read folder
folders_categories = dir(folder);

% Loop through all category folders and extract features
for folder_category = folders_categories(3:end)'

    % Read folder
    images = dir(strcat(folder_category.folder, "\", folder_category.name));
    
    % Loop through all images in the current category folder
    for image = images(3:end)'
        
        % Read and convert image (as needed)
        image_path = strcat(image.folder, "\", image.name);
        I = imread(image_path);
        if size(I,3) == 3
            I = rgb2gray(I);
        end
        I = single(I);
        
        % Calculate features
        [~, descriptors] = vl_dsift(I, 'Step', 2, 'Fast');
        descriptors = single(descriptors);
        
        % Assign features to visual words
        indices = knnsearch(C, descriptors');
        
        % Build histogram from indices and normalize it
        bincounts = histc(indices, binranges);
        bincounts = bincounts/norm(bincounts);
        
        % Assign training values of current image
        samples(image_count, :) = bincounts';
        
        image_count = image_count + 1;
    end
    
    % Classify batch
    classes = predict(model, samples);
    classescounts = histc(classes, classranges);
    
    % Increase respective count in confusion matrix
    conf_matrix(folder_count, :) = conf_matrix(folder_count, :) + classescounts';
    
    % Give at least some feedback
    fprintf("Classified %d of %d folders\n", folder_count, num_classes);
    
    % "Switch" to new class label
    folder_count = folder_count + 1;
    
    % Reset image_count for next run
    image_count = 1;
end

end

