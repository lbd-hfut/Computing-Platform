% Get the current working directory
folder_path = pwd;

% Get all files in the directory
files = dir(fullfile(folder_path, '*.*'));
index = 1;
% Initialize counter
for i = 1:length(files)
    file_name = files(i).name;
    
    % Get the full path of the file
    old_file_path = fullfile(folder_path, file_name);
    
    % Get the file extension
    [~, ~, ext] = fileparts(file_name);

    % Check if the file is an image of the specified formats
    if ismember(lower(ext), {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'})
        % Rename based on the index
        if index == 1
            new_file_name = sprintf('r%05d%s', index - 1, ext);  % First image starts with 'r'
        else
            new_file_name = sprintf('d%05d%s', index - 1, ext);  % Subsequent images start with 'd'
        end
        index = index+1;
        new_file_path = fullfile(folder_path, new_file_name);
        
        % Rename the file
        movefile(old_file_path, new_file_path);
    end
end

disp('Renaming complete!');
