function P=camcalibDLT(Xworld,Xim)
    % Inputs: 
    %   Xworld, world coordinates in the form (id, coordinates)
    %   Xim, image coordinates in the form (id, coordinates)
    
    % Create the matrix A 
    %%-your-code-starts-here-%%
    n = 8;
    A = zeros(16,12);
    y = Xim(:,2);
    x = Xim(:,1);
    for i=1:8 
        A(i*2-1,:) = [zeros(1,4) Xworld(i,:) -y(i)*Xworld(i,:)];
        A(i*2,:) = [Xworld(i,:) zeros(1,4) -x(i)*Xworld(i,:)];
    end
    disp(A)
    %%-your-code-ends-here-%%
       
    
    % Perform homogeneous least squares fitting,
    % the best solution is given by the eigenvector of 
    % A.T*A with the smallest eigenvalue.
    %%-your-code-starts-here-%%
    evs=A'*A;
    [V,D]= eig(evs)
    [d,ind]= min(diag(D));
    ev= V(:,ind);
    %%-your-code-ends-here-%%

    % Reshape the eigenvector into a projection matrix P
    P =(reshape(ev,4,3))'; % uncomment this
    %P = [1 0 0 0;0 1 0 0;0 0 1 0];  % remove this

end