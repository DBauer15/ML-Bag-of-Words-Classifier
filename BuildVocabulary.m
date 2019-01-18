function C = BuildVocabulary(folder,num_clusters)

% Create data structure to store features
features = [];

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
        
        % Determine density of features
        size_I = numel(I);
        step = floor(sqrt(size_I / 100));
        
        % Calculate features and add them to the list
        [~, descriptors] = vl_dsift(I, 'Step', step, 'Fast');
        features = [features descriptors];
    end
end

% Convert format
features = single(features);

% Cluster features into num_clusters clusters using k-means clustering
[centers, ~] = vl_kmeans(features, num_clusters);
C = centers;

end