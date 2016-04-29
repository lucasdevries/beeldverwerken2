    %% Question 1
    img = imread('cameraman.jpg');
    SPM = 4;
    SPN = 3;
    % Motion blur
    subplot(SPM,SPN,[1,4]);
    imshow(img)
    title('Original')
    G = (1/6)*[1,1,1,1,1,1];
    image = filterImage(img, G);
    subplot(SPM,SPN,2);
    imshow(image);
    title('Q1.1 Motion blur')