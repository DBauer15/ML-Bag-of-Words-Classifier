function [training,group] = BuildKNN(folder,C)

fprintf("Starting knn build process...\n");

% Initialize data
binranges = 1:size(C,2);
image_count = 1; % Used for assignments in "training" and "group"
folder_count = 1; % Used to store current class label
group = zeros(800,1);
training = zeros(800,size(C,2));

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
        indices = knnsearch(C', descriptors');
        
        % Build histogram from indices and normalize it
        bincounts = histc(indices, binranges);
        bincounts = bincounts/norm(bincounts);
        
        % Assign training values of current image
        training(image_count, :) = bincounts';
              
        % Assign label of current image
        group(image_count) = folder_count;
        
        image_count = image_count + 1;
    end
    
    fprintf("Went through %d of %d images\n", image_count); 
    
    % "Switch" to new class label
    folder_count = folder_count + 1;
end

end

