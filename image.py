################################################################################
# This is a short example on how to read and plot the snapshots produced by my #
# simulation code version 3+. It reads a snapshot given as the first argument   #
# and saves the created image under the file name given as second argument.     #
# specifications (domain size, number of points, parameter values) must be      #
# manually set below or given as additional arguments.                         #
#                                          Dominik Suchla, 2020                #
################################################################################
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as col
################################################################################
spectral = False

# snapshot that is drawn:
if len(sys.argv)==3:
    # I/O:
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    # the specifications (to be edited by hand)
    Lx = 200*np.pi
    Ly = Lx
    Nx = 1024
    Ny = Nx
    alpha = -0.800
    beta = 0.01
    lamb = 7
elif len(sys.argv)==7:
    # I/O:
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    # the specifications (assuming a square domain and beta=0.01)
    Lx = float( sys.argv[3] )
    Ly = Lx
    Nx = int( sys.argv[4] )
    Ny = Nx
    alpha = float( sys.argv[5] )
    beta = 0.01
    lamb = float( sys.argv[6] )
else:
    print( "Error: arguments for I/O missing" ) 
    sys.exit(1)



################################################################################
# reading one snapshot, the last two values are the mean velocity components
arr = np.fromfile( input_file, sep="," )
arr = arr[:-2].reshape((Ny,Nx+2))
# interpret the tuples of real and imaginary part as one complex number
arr = arr.view( dtype=np.complex128 )

if spectral:
    # add the second half of the Fourier transform in four steps
    # 1. copy relevant part and conjugate
    sec_half = np.conj(arr[:,1:-1])
    # 2. reverse the order along the x-axis
    sec_half = sec_half[::,::-1]
    # 3. reverse the order along the y-axis, except for the fist row
    sec_half = np.concatenate( (sec_half[0:1], sec_half[:0:-1]), axis=0 )
    # 4. we put both parts back together and rearange the y-direction
    # (0,1,2,-2,-1 -> -2,-1,0,1,2) 
    arr = np.concatenate( (sec_half, arr), axis=1 )
    arr = np.roll(arr, Nx//2-1, axis=0)
    # absolute value
    arr = np.sqrt( arr.real**2 + arr.imag**2 )
else:
    arr = np.fft.irfft2( arr )

################################################################################
# plotting
fig = plt.figure( figsize=(8,8) )
ax_im = fig.add_axes( [0.1,0.15,0.80,0.80] )
ax_cb = fig.add_axes( [0.05,0.05,0.90,0.05] )
ax_im.set_title( "vorticity of a 2D active flow, $\\alpha=%.3f$, $\\beta=%.2f$,\
 $\\lambda=%.1f$" %(alpha, beta, lamb) )
if spectral:
    # 2D colour map of the Fourier field
    dk = 2*np.pi/Lx
    image = ax_im.imshow( arr, cmap="inferno", interpolation="nearest",\
            extent=[-(Nx/2+1)*dk,(Nx/2+1)*dk,-(Ny/2+1)*dk,(Ny/2+1)*dk], origin="lower" )
else:
    # 2D colour map of the real field
    image = ax_im.imshow( arr, cmap="bwr", interpolation="nearest",\
                       extent=[0,Lx,0,Ly], origin="lower" )

cbar = fig.colorbar( image, cax=ax_cb, orientation="horizontal" )
#cbar = fig.colorbar(image, ax=ax, pad=0.05, fraction=0.047, orientation="vertical")
cbar.ax.tick_params( labelsize=12 )
if spectral==False:
    # set the range of the colors such that white is the zero
    max_val = np.absolute( arr ).max()
    image.set_clim( -max_val, max_val )
# save the figure (note that the image is 0.8*8"=6.4" wide and contains 2^x
# ponts, so the dpi should be 80*2^y)
fig.savefig( output_file+".png", format="png", dpi=320 )

################################################################################
sys.exit(0)
