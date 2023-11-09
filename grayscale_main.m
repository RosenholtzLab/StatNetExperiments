function output_img = grayscale_main(input_image)    
    % Inputs: 
    % input image - reference texture
        
    % Outputs:
    % output_img - heeger beergen synthesized texture

    % Load the input image 
    img2 = imread(input_image);
    img = im2double(img2);
    oim22 = im2gray(img);
    
    % assuming your image is stored in variable 'img', for cropping
    [h, w, ~] = size(oim22);
    start_row = round((h-256)/2);
    start_col = round((w-256)/2);
    oim = imcrop(oim22, [start_col, start_row, 256, 256]);
  
    
    % Create the white noise
    img_size = size(oim);
    noise_img = rand(img_size);
    
    % First step: Match histogram of input and white noise
    u_matched = histogram_matching_CK(noise_img, oim); 
    
    % Build the steerable images through the spatial domain
    % Parameters
    stages = 4;
    num_bands = 3;
    
    %Build steerable pyramids for texture 
    [pyr,pind] = buildSCFpyr(oim, stages, num_bands);
    rec_im = reconSCFpyr(pyr, pind);
    
    % Get total number of images in pyramid
    num = length(pind);

    % Set the number of iterations
    iterations = 6;

    rec_im = u_matched;
    
    for i = 1:iterations
        [pyr_m, pind_m] = buildSCFpyr(u_matched, stages, num_bands); %parameter anpassen
    
        for j = 1:num
    
            indices = pyrBandIndices(pind_m, j); % get indices
            inp_img = pyr_m(indices); % find the current white noise band
            ref_img = pyr(indices); % find the reference image
    
            if isreal(inp_img)
                inp_matched = histogram_matching_CK(inp_img, ref_img);
                print
            else    
                magnitude_img = abs(inp_img);
                phase_img = angle(inp_img);
                magnitude_ref = abs(ref_img);
                phase_ref = angle(ref_img);
                inp_matched_magnitude = histogram_matching_CK(magnitude_img, magnitude_ref);
                inp_matched_phase = histogram_matching_CK(phase_img, phase_ref);
                inp_matched = inp_matched_magnitude .* (cos(inp_matched_phase) + 1i * sin(inp_matched_phase));
            end
            
            pyr_m(indices) = inp_matched; % save the newly matched histogra

        end
    
        rec_im = reconSCFpyr(pyr_m, pind_m);
        u_matched = histogram_matching_CK(rec_im, oim);
        output_img = u_matched;
    
    end
end
