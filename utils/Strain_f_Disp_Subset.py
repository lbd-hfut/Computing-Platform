import numpy as np

def Strain_from_Displacement_Subset(u, v, flag, step, SmoothLen):
    """
    Calculate strain components (Ex, Ey, Exy) from displacement fields (u, v) 
    using a subset method.
    
    Parameters:
    u: 2D numpy array
        The horizontal displacement field.
    v: 2D numpy array
        The vertical displacement field.
    flag: 2D numpy array
        A binary matrix indicating valid points in the displacement fields 
        (1 for valid, 0 for invalid).
    step: float
        The spatial resolution or step size between grid points in the displacement field.
    SmoothLen: int
        The size of the smoothing window. This value must be an odd number; if it's even, 
        the function will increment it by 1.
    
    Returns:
    Ex, Ey, Exy: 2D numpy arrays
        The calculated strain components, corresponding to the normal strains in the 
        x- and y-directions (Ex and Ey) and the shear strain (Exy), respectively.
    """

    # Transpose the displacement and flag matrices
    u = u.T
    v = v.T
    flag = flag.T
    
    # Ensure SmoothLen is an odd number
    m = SmoothLen
    if m % 2 == 0:  # Check if m is even
        m += 1  # Increment by 1 to make m odd
    hfm = (m - 1) // 2  # Half the filter size

    # Extend the displacement and flag matrices with NaN padding
    uc = np.full((u.shape[0] + 2*hfm, u.shape[1] + 2*hfm), np.nan)
    vc = np.full((v.shape[0] + 2*hfm, v.shape[1] + 2*hfm), np.nan)
    flag_c = np.zeros((flag.shape[0] + 2*hfm, flag.shape[1] + 2*hfm))
    ny, nx = uc.shape

    # Copy the original displacement data into the center of the extended matrices
    uc[hfm:ny-hfm, hfm:nx-hfm] = u
    vc[hfm:ny-hfm, hfm:nx-hfm] = v
    flag_c[hfm:ny-hfm, hfm:nx-hfm] = flag
    
    # Initialize the strain matrices with NaN values
    Ex = np.full(uc.shape, np.nan)
    Ey = np.full(uc.shape, np.nan)
    Exy = np.full(uc.shape, np.nan)

    # Loop through each pixel in the extended matrices
    for i in range(nx):
        for j in range(ny):
            if flag_c[j, i] == 0:  # Skip if the flag indicates an invalid point
                continue
            
            # Determine the start and end coordinates of the current window
            startx = i - hfm
            starty = j - hfm
            stopx = i + hfm
            stopy = j + hfm

            # Extract the current window of displacement and flag data
            uu = uc[starty:stopy+1, startx:stopx+1]
            vv = vc[starty:stopy+1, startx:stopx+1]
            FLAG = flag_c[starty:stopy+1, startx:stopx+1]

            # Generate grid coordinates for the window
            X, Y = np.meshgrid(np.arange(-hfm, hfm+1), np.arange(-hfm, hfm+1))
            xx = X * step
            yy = Y * step
            X = np.column_stack((np.ones(m**2), yy.flatten(), xx.flatten()))

            # Find the indices of valid points in the window
            f_valid = np.where(FLAG.flatten() == 1)[0]

            if len(f_valid) > 0:
                if np.sum(FLAG) > 3:  # Proceed if there are more than 3 valid points
                    f_invalid = np.where(FLAG.flatten() == 0)[0]
                    U = uu.flatten()[f_valid]
                    V = vv.flatten()[f_valid]
                    X = np.delete(X, f_invalid, axis=0)
                    
                    # Perform least squares fitting to estimate the displacement gradients
                    a = np.linalg.lstsq(X, U, rcond=None)[0]
                    b = np.linalg.lstsq(X, V, rcond=None)[0]
                else:
                    # If not enough valid points, return NaN
                    a = np.full(3, np.nan)
                    b = np.full(3, np.nan)
            else:
                # If no valid points, return NaN
                a = np.full(3, np.nan)
                b = np.full(3, np.nan)

            # Store the computed strain components
            Ex[j, i] = a[1]
            Ey[j, i] = b[2]
            Exy[j, i] = (a[2] + b[1]) / 2

    # Trim the extended edges of the strain matrices
    Ex = Ex[hfm:ny-hfm, hfm:nx-hfm]
    Ey = Ey[hfm:ny-hfm, hfm:nx-hfm]
    Exy = Exy[hfm:ny-hfm, hfm:nx-hfm]

    # Return the strain components with the original orientation
    return Ex.T, Ey.T, Exy.T


## for test
# H = 4; L = 5
# u = np.zeros((H,L))
# v = np.ones((H,L))
# flag = np.ones((H,L))
# u = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]])
# v = np.array([[1,1,1,1],[3,3,3,3],[5,5,5,5],[7,7,7,7]])
# flag = np.ones((4,4))
# step = 1
# SmoothLen = 3
# Ex, Ey, Exy = Strain_from_Displacement_Subset(u, v, flag, step, SmoothLen)