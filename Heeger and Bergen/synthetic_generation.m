%% Generate the synthetic dataset and save them at the right files
% Define the root directory
rootDir = './dtd_torch/dtd/dtd/images';

% Get a list of all subfolders within the root directory
subfolders = dir(fullfile(rootDir, '*'));
isSubfolder = [subfolders(:).isdir]; %# returns logical vector
subfolders = {subfolders(isSubfolder).name}';
subfolders(ismember(subfolders,{'.','..'})) = []; % remove '.' and '..'

% Loop over each subfolder
for i = 1:numel(subfolders)
    
    % Define the full path to the current subfolder and its corresponding output folder
    subfolderPath = fullfile(rootDir, subfolders{i});
    outputFolder = fullfile(subfolderPath, 'output');

    % Get a list of all jpg files in the current subfolder
    files = dir(fullfile(subfolderPath, '*.jpg'));
    files = {files(:).name}'
    
    % Loop over each file
    for j = 1:numel(files)
        
        % Define the full path to the current file
        filePath = fullfile(subfolderPath, files{j});

        % Call your function
        output_img = grayscale_main(filePath);

        % Save the output image in the output folder
        [~, name, ~] = fileparts(filePath);
        outputName = fullfile(outputFolder, [name, '_output.jpg']);
        imwrite(output_img, outputName);
    end
end
