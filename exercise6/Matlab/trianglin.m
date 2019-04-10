function X=trianglin(P1,P2,x1,x2)
    
    % Inputs:
    %   P1 and P2, projection matrices for both images
    %   x1 and x2, image coordinates for both images
    
    % Form A and get the least squares solution from the eigenvector 
    % corresponding to the smallest eigenvalue
    %%-your-code-starts-here-%%
    x1x = [0 -x1(3) x1(2);...
           x1(3) 0 -x1(1);...
           -x1(2) x1(1) 0]; 
    x2x = [0 -x2(3) x2(2);...
           x2(3) 0 -x2(1);...
           -x2(2) x2(1) 0];
     
    A= [x1x*P1; x2x*P2];    
    
    evs=A'*A;
    [V,D]= eig(evs)
    [d,ind]= min(diag(D));
    X= V(:,ind);
        
    %X = ones(4,1);  % replace with your implementation
    %%-your-code-ends-here-%%
    

end
