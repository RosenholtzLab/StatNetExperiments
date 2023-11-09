%%
%Matches histogram of the input image u to the reference image v

function u_matched = histogram_matching_CK(u, v)
    % Inputs: 
    % u - input image
    % v - reference image
    
    % Outputs:
    % u_matched - output image with the same histogram as v
    
    % Save the original size of u for reshaping later
    original_size = size(u);
    k = u;

    % Reshape the images as vectors
    u = u(:);
    v = v(:);
    L = numel(u); % or numel(v), as they should have the same size
    
    % Sort the reference image v and get sorting index
    [v_sorted, tau] = sort(v);
    
    % Sort the input image u and get sorting index
    [u_sorted, sigma] = sort(u);
    
    % Match the histogram of u
    u_matched = u;
    for k = 1:L
        % the k-th pixel of u takes the gray-value of the k-th pixel of v
        u_matched(sigma(k)) = v_sorted(k);
    end
    
    % Reshape u_matched back into its original size, only if input is
    % matrix
    if ndims(k) >= 2
        u_matched = reshape(u_matched, original_size);
    end
end

