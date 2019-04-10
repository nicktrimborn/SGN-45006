import numpy as np

# Direct linear transformation (DLT) is an algorithm which 
# solves a set of variables from a set of similarity relations,
# e.g  the relation between 3D points in a scene and 
# their projection onto an image plane

def camcalibDLT(Xworld, Xim):
    # Inputs: 
    #   Xworld, world coordinates in the form (id, coordinates)
    #   Xim, image coordinates in the form (id, coordinates)
    
    # Create the matrix A 
    ##-your-code-starts-here-##

    ##-your-code-ends-here-##
    
    # Perform homogeneous least squares fitting,
    # the best solution is given by the eigenvector of 
    # A.T*A with the smallest eigenvalue.
    ##-your-code-starts-here-##

    ##-your-code-ends-here-##
    
    # Reshape the eigenvector into a projection matrix P
    #P = np.reshape(ev, (3,4))  # uncomment this
    P = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=float) # remove this
    
    return P