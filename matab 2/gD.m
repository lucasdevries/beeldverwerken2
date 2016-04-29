function H =  gD(image, sigma, xorder, yorder )
M = 4*sigma;
N = 4*sigma;
x = -M : M ;
y = -N : N ;
% create a sampling grid
[X , Y ] = meshgrid (x , y ); 
switch xorder
    case '1'
    dx = -X./sigma^2.* gauss1(sigma)
        switch yorder
            case '1'
                dy = -Y./sigma^2.* gauss1(sigma)
                G = dx.*dy'
                H = imfilter(image, G, 'conv', 'replicate')
            case '2'
                disp('hoi')
                ddy = (Y.^2 ./ sigma^4 - 1/sigma^2).* gauss1(sigma)
                G = dx.*ddy'
                H = imfilter(image, G, 'conv', 'replicate')
           end
    case '2'
    ddx = (X.^2 ./ sigma^4 - 1/sigma^2).* gauss1(sigma)
            switch yorder
            case '1'
                disp('hoi')
                dy = -Y./sigma^2.* gauss1(sigma)
                G = ddx.*dy'
                H = imfilter(image, G, 'conv', 'replicate')
            case '2'
                disp('hoi')
                ddy = (Y.^2 ./ sigma^4 - 1/sigma^2).* gauss1(sigma)
                G = ddx.*ddy'
                H = imfilter(image, G, 'conv', 'replicate')
            end
end      
        
% end
% switch yorder
%     case '1'
%         dy = -Y./sigma^2.* gauss1(sigma)
%     case '2'
%         ddy = (Y.^2 ./ sigma^4 - 1/sigma^2).* gauss1(sigma)
% end



% dxy = -X./sigma^2 .* dy
% ddx = (X.^2 ./ sigma^4 - 1/sigma^2).* gauss1(sigma)
% ddy = (Y.^2 ./ sigma^4 - 1/sigma^2).* gauss1(sigma)
end

