/******************************************************************************* 
 * This is a simulation of the simple model for active turbulence prposed by   * 
 * Wensink et. al. in the paper "Meso-scale turbulence in living fluids".      * 
 * All equations used are based on a normalized form of the model derived by   * 
 * Martin James, Wouter Bos and Michael Wilczek in the paper "Turbulence and   * 
 * turbulent pattern formation in a minimal model for active fluids". It uses  * 
 * a spseudo-spectral algorithm with integrating factor to simultaneously      * 
 * solve the vorticity equation as well as a seperate equation for the mean    * 
 * velocity:                                                                  * 
 *                                                                            * 
 * \partial_t\omega + \lambda\left(\vec v\cdot\vec\nabla\right)\omega         * 
 * = -\big(1+\Delta\big)^2\omega -\alpha\omega -\beta\vec\nabla\times\left(   * 
 * \left|\vec v^2\right|\,\vec v\right)                                       * 
 *                                                                            * 
 * \partial_t\langle\vec v\rangle = -\big(1+\alpha\big)\langle\vec v\rangle   * 
 * -\beta\left\langle\left|\vec v\right|^2\,\vec v\right\rangle               *  
 *                                                                            * 
 * Warnings:                                                                  * 
 * - The simulation is optimized for speed, not for memory usage. It creates  * 
 *   10 arrays of size Ny*(Nx+2) and precision <float>, as well as several    * 
 *   scalars.                                                                 * 
 * - Binary operators for two objects of the cuComplex class as well as one   * 
 *   object and a float or integer are required. I have written a custom      * 
 *   header called "cuComplexBinOp.h". If you don't have that available, you  * 
 *   will need to write it yourself.                                          * 
 *                                                                            * 
 * Notation remarks:                                                          * 
 * - The same arrays are used for real space and Fourier space. In Fourier    * 
 *   space, two consecutive elements form a pair of real and imaginary part.  * 
 *   This is handled by using a different pointer to the same array which has * 
 *   "_comp" added to the original name (and uses the cufft native container  * 
 *   class "cuComplex"). All loops over the real array have "r" as running    * 
 *   variable, all loops over the complex array have "c" as running variable. * 
 *   Some arrays (like all wavenumber related ones) consist only of a real    * 
 *   part, but are used for complex calculations. These arrays don't have a   * 
 *   "_comp" in the end (they don't use cuComplex), but may still appear in   * 
 *   calculations with "c" as running variable.                               * 
 * - Functions that start with "cuC" as well as "make_cuComplex()" are part   * 
 *   of <cuComplex.h> and explained there. (They mostly do what you would     * 
 *   expect based on their name.)                                             * 
 * - Things marked with ######## are for testing purposes and should not      * 
 *   remain (uncommented) in the final code.                                  * 
 *                                                                            * 
 * If you have trouble understanding the code but want to use it anyway, you  * 
 * can compile this using "nvcc -std=c++11 this_file.cu -I ~/path_to_BinOp -l * 
 * cufft -o output" . Of course you have to replace "this_file" with the name * 
 * of this file and path_to_BinOp. with the path to the cuComplexBinOp.h      *
 * library. Then execute it with "./output help" to display some information  * 
 * that might help you.                                                       * 
 *                                                                            * 
 *                                                   Dominik Suchla, 2018     * 
 ******************************************************************************/

//________change_log____________________________________________________________
//
// version 2-0:
// - outputs the mean energy in an additional file
// version 2-1
// - accepts the words "random" and "blank" below "initial_condition" in the
//   specifications file, will create an initial condition accordingly
// - uses the time interval between two snapshots ("dt_snap") instead of the
//   number of timesteps between snapshots ("snap")
// version 2-2,2-3:
// - bug fixes
// version 2-5:
// - adding a number behind the "random" will change the seed, which otherwise
//   only depends on time
// version 3-0:
// - The command line continuation has been removed, as have the old initial
//   conditions (which used a different format than the snapshots).
// - To continue from a previous simulation, type "continue" into the initial
//   condition line in the specifications file, optionally followed by two
//   numbers (divided by spaces). Without numbers the program will simply
//   continue from the last snapshot, until the total time is reached. With one
//   number, it will continue from the last snapshot for a time intervall
//   specified by that number. With two numbers, the first does the same as
//   above and the second specifies the snapshot from which it will continue.
// version 3-1:
// - bug fixes
// - now uses round() to avoid the problems with calculating the right amount
//   of time steps
// version 3-2:
// - an unknown phrase for the initial condition will no longer result in a
//   blank (all-zero) simulation but output an error instead
// - the dealiasing is now based on the absolute value of the wavevector


#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>
#include <random>

#include <cufft.h>
#include <cuComplex.h>
#include <cuComplexBinOp.h>
#include <cudaErr.h>

#define _pi 3.14159265358979323846
// amount of kernels called, block_size must be n*32
#define BLOCK_SIZE 1024
#define BLOCK_NUM 64


using namespace std;



/******************************************************************************* 
 * prototypes                                                                  * 
 *******************************************************************************/

inline bool check_func(cuComplex*);

__global__
void cutoff_func(float*, int, int);

__global__
void dealiasing_func(cuComplex*, cuComplex*, float*, int);

inline bool existence_check(string, int);

__global__
void integrate_func_0(cuComplex*, cuComplex*, cuComplex*, float*, int);

__global__
void integrate_func_1(cuComplex*, cuComplex*, cuComplex*, float*, float*,
                      cuComplex*, int, float);
__global__
void integrate_func_2(cuComplex*, cuComplex*, cuComplex*, float*, cuComplex*,
                      int, float);
__global__
void integrate_func_3(cuComplex*, cuComplex*, cuComplex*, float*, float*,
                      cuComplex*, int, float);
__global__
void integrate_func_4(cuComplex*, cuComplex*, int, float);

__global__
void lin_func(float*, float*, float*, int, float, float);

inline void nonl_func(float*, cuComplex*, float*, cuComplex*, float*, cuComplex*,
                      float*, cuComplex*, float*, float*, float*, float*,
                      cuComplex*, int, float, cufftHandle, cufftHandle, float*,
                      float*, float*, float*);

inline void modified_nonl_func(float*, cuComplex*, float*, cuComplex*, float*,
                               cuComplex*, float*, cuComplex*, float*, float*,
                               float*, float*, cuComplex*, int, int, int, float,
                               cufftHandle, cufftHandle, float*, float*, float*,
                               float*, int);

__global__
void nonl_subfunc_0(cuComplex*, cuComplex*, cuComplex*, float*, float*, float*,
                    int);
__global__
void nonl_subfunc_1(float*, float*, float*, float*, int);

__global__
void nonl_subfunc_2(cuComplex*, cuComplex*, cuComplex*, cuComplex*, cuComplex*,
                    float*, float*, float*, float, float, int);
__global__
void reality_func(cuComplex*, int, int, int);

bool save_func(cuComplex*, cuComplex*, float*, float*, int, int, float);

bool save_energy_func(float, int, float);

void unallocate_func( cufftHandle, cufftHandle, float*, float*, float*, float*,
                      cuComplex*, cuComplex*, cuComplex*, cuComplex*,
                      cuComplex*, cuComplex*, cuComplex*, float*, float*,
                      cuComplex*, float*, float*, float*, float*, float*,
                      float*, float*, float*, float*, float*, float*, float*);

__global__
void wavenumber_func(float*, float*, float*, float, float, int, int);


/******************************************************************************* 
 * global variables                                                            * 
 *******************************************************************************/

// total time, length of one time step
float T{ 100 };
float dt{ 0.002 };
// total length, number of gridpoints and array size
float Lx{ 250 };
float Ly{ Lx };
int Nx{ 1024 };
int Ny{ Nx };
// decides what initial values are used
string initial{"random"};
// determines the folder where snapshots are saved
string directory{ "./" };
// after what time intervall / how many timesteps a snapshot is taken
float dt_snap{ 0.5 };
int snap{ 200 };

// advection coefficient
float lambda{ 7. };
// linear friction coefficient (not really a friction)
float alpha{ -0.8 };
// cubic friction coefficient
float beta{ 0.01 };

// another memory location for w[0] to optimize memory traffic in check_func
cuComplex w_check{ 0, 0 };


/******************************************************************************* 
 * main function                                                               * 
 *******************************************************************************/

int main(int argc, char *argv[])
{
	if( argc==2 )
	{
		// 1st optional argument = help: print some helping information
		if( !strcmp(argv[1], "help") || !strcmp(argv[1], "h") )
		{
			cout << "\n--------------------------------------------------\n"
				<< "This is a program to simulate bacteria movement in a "
				<< "two-dimensional continuum model. The source code is well "
				<< "commented and contains a small introductory paragraph.\n"
				<< "The program will try to load settings from a file "
				<< "called \"specifications.txt\".\n"
				<< "Currently allowed commands are:\n" << endl
				<< "./this_file help    displays this text.\n" << endl
				<< "./this_file spec    creates an example for a "
				<< "specifications file using the default settings.\n" << endl
				<< "./this_file math    creates a text file with the "
				<< "underlaying equations and references to the scientific "
				<< "papers they are from.\n" << endl
				<< "./this_file    starts the simulation with default "
				<< "settings. All snapshots are saved in the same directory "
				<< "where it is run.\n" << endl
				<< "./this_file path    starts a simulation. When the program "
				<< "looks for specifications or saves snapshots, it will use "
				<< "the directory specified by \"path\".\n" << endl
				<< "./this_file path continue number    continues a previous "
				<< "(interrupted) simulation. The program will search the "
				<< "directory given by \"path\" for specifications and for a "
				<< "snapshot with the number given by \"number\". It will use "
				<< "this snapshot as initial condition and continue until the "
				<< "end of the simulation (given in the specifications file) "
				<< "is reached.\n"
				<< "--------------------------------------------------" << endl;
			return 0;
		}
		// 1st optional argument = spec: create an example for the spec file
		else if( !strcmp(argv[1], "spec") )
		{
			ofstream file( "specifications.txt", ios::out );
			if( !file.is_open() )
			{
				cerr << "Error: unable to create example file" << endl;
				return 1;
			}
			file << "T:\n" << T << "\ndt:\n" << dt << "\nLx:\n" << Lx
				<< "\nLy:\n" << Ly << "\nNx:\n" << Nx << "\nNy:\n" << Ny
				<< "\nlambda:\n" << lambda << "\nalpha:\n" << alpha
				<< "\nbeta:\n" << beta << "\ndt_snap:\n" << dt_snap
				<< "\ninitial_condition:\n" << initial << "\nsave_directory:\n"
				<< directory;
			file.close();
			cout << "example has been created" << endl;
			return 0;
		}
		// 1st optional argument = math: create an text file with some theory
		else if( !strcmp(argv[1], "math") )
		{
			ofstream file( "theory.tex", ios::out );
			if( !file.is_open() )
			{
				cerr << "Error: unable to create theory file" << endl;
				return 1;
			}
			file << "\\documentclass{article}\n\\begin{document}\n\nThis is a "
				<< "simulation of the simple model for active turbulence "
				<< "prposed by Wensink et. al. in the paper \"Meso-scale "
				<< "turbulence in living fluids\". All equations used are "
				<< "based on a normalized form of the model derived by Martin "
				<< "James, Wouter Bos and Michael Wilczek in the paper "
				<< "\"Turbulence and turbulent pattern formation in a minimal "
				<< "model for active fluids\".\nTaking the curl of the given "
				<< "velocity equation yields a nonlinear partial differential "
				<< "equation for the vorticity $\\omega=\\vec\\nabla\\times"
				<< "\\vec v$:\n\\begin{equation}\n\\partial_t\\omega + \\lambda"
				<< "\\left(\\vec v\\cdot\\vec\\nabla\\right)\\omega = -\\big("
				<< "1+\\Delta\\big)^2\\omega -\\alpha\\omega -\\beta\\vec"
				<< "\\nabla\\times\\left(\\left|\\vec v^2\\right|\\,\\vec v"
				<< "\\right)\n\\end{equation}\nThis equation is solved in "
				<< "Fourier space with a Runge-Kutta method of 4th order and "
				<< "an integrating factor.\nUnfortunately, the vorticity "
				<< "equation contains no information about the mean velocity "
				<< "$\\langle\\vec v\\rangle$. One can derive a second partial "
				<< "differential equation to restore the lost information:\n"
				<< "\\begin{equation}\n\\partial_t\\langle\\vec v\\rangle = "
				<< "-\\big(1+\\alpha\\big)\\langle\\vec v\\rangle -\\beta\\left"
				<< "\\langle\\left|\\vec v\\right|^2\\,\\vec v\\right\\rangle\n"
				<< "\\end{equation}\nBoth are simoultaneously evaluated.\n"
				<< "\\end{document}" << endl; 
			file.close();
			cout << "a Latex file \"theory.tex\" has been created" << endl;
			return 0;
		}
		// 1 optional argument: the directory with the specifications file
		else
			directory = argv[1];
	}
	else
	{
		cerr << "More than one argument is no longer supported. See the change"
		     << " log in the beginning of the source file." << endl;
		return 1;
	}
/******************************************************************************* 
 * reading the parameters and defining additional variables (tested)           * 
 *******************************************************************************/
	cout << "reading the specifications (path=" << directory 
		<< "/specifications.txt)" << endl;
	
	// generic string, used for reading files on multiple occasions
	string line;
	
	ifstream specif( directory + "/specifications.txt" );
	
	if (specif.is_open())
	{
		while( getline(specif,line) )
		{
			if (line=="T:")
			{
				getline(specif,line);
				T = stof(line);
			}
			else if (line=="dt:")
			{
				getline(specif,line);
				dt = stof(line);
			}
			else if (line=="Lx:")
			{
				getline(specif,line);
				Lx = stof(line);
			}
			else if (line=="Ly:")
			{
				getline(specif,line);
				Ly = stof(line);
			}
			else if (line=="Nx:")
			{
				getline(specif,line);
				Nx = stoi(line);
			}
			else if (line=="Ny:")
			{
				getline(specif,line);
				Ny = stoi(line);
			}
			else if (line=="lambda:")
			{
				getline(specif,line);
				lambda = stof(line);
			}
			else if (line=="alpha:")
			{
				getline(specif,line);
				alpha = stof(line);
			}
			else if (line=="beta:")
			{
				getline(specif,line);
				beta = stof(line);
			}
			else if (line=="dt_snap:")
			{
				getline(specif,line);
				dt_snap = stof(line);
			}
			else if (line=="initial_condition:")
			{
				getline(specif,line);
				initial = line;
			}
			else if (line=="save_directory:")
			{
				getline(specif,line);
				directory = line;
			}
			else
			{
				cerr << "Warning: specification file contains the following"
					<< " unknown specification: " << line << endl;
				cerr << "For additional informatin on the specifications file"
					<< " start the program with the argument \"help\"." << endl;
				specif.close();
				return 1;
			}
		}
		specif.close();
	}
	else
	{
		cerr << "Error: Unable to open \"specifications.txt\"." << endl;
		return 1;
	}
	// We calculate some variables depending on the specifications:
	// number of time steps
	int M{ static_cast<int>(std::round(T/dt)) };
	// distance between gridpoints
	float dx{ static_cast<float>(Lx) / Nx };
	float dy{ static_cast<float>(Ly) / Ny };
	// half of Nx plus one, size of the array used in rfft
	int Nxh{ Nx/2 + 1 };
	// The size of the array used for the fft algorithm
	// (in-place transform -> 2 columns for padding)
	int size{ Ny*Nxh };
	// After how many time steps a snapshot is taken:
	snap = static_cast<int>( std::round(dt_snap/dt) );

	cout << "--------------------------------------------------\n"
		<< "T = " << T << "\ndt = " << dt << "\nM = " << M << "\nLx = "
		<< Lx << "\nLy = " << Ly << "\nNx = " << Nx << "\nNy = " << Ny
		<< "\ndx = " << dx << "\ndy = " << dy << "\nalpha = " << alpha 
		<< "\nbeta = " << beta << "\nlambda = " << lambda 
		<< "\ninitial condition: " << initial
		<< "\ntotal amount of snapshots: " << static_cast<int>(M/snap)
		<< "\n--------------------------------------------------" << endl;
	
	// The ifft is not normalized. Since it is only used in nonl_func, we can
	// put the normalization into the constants beta and lambda:
	float Fnorm{ static_cast<float>(Nx*Ny) };
	beta /= (Fnorm * Fnorm * Fnorm);
	lambda /= (Fnorm * Fnorm);
	
/****************************************************************************** 
 * allocating memory on the unified CPU/GPU memory introduced with cuda-6     * 
 ******************************************************************************/
	cout << "allocating unified memory" << endl;
	
	// The input and output arrays are large enough for 2* Ny*(Nx/2+1) elements
	// in real space or Ny*(Nx/2+1) real-complex touples in Fourier space.
	// In real space, the last two columns have no meaning and are ignored.
	
	// The arrays that store the velocity and vorticity fields and have a
	// Fourier transform:
	float *vx, *vy, *w_appr, *arr;
	cuda_error_func( cudaMallocManaged( &vx, 2*size*sizeof(float) ) );
	cuda_error_func( cudaMallocManaged( &vy, 2*size*sizeof(float) ) );
	cuda_error_func( cudaMallocManaged( &w_appr, 2*size*sizeof(float) ) );
	cuda_error_func( cudaMallocManaged( &arr, 2*size*sizeof(float) ) );
	
	// The arrays that store the vorticity fields and only exist in Fourier
	// space (last one is for save_func):
	cuComplex *w_old_comp, *w_new_comp, *w_save_comp;
	cuda_error_func( cudaMallocManaged( &w_old_comp, 2*size*sizeof(float) ) );
	cuda_error_func( cudaMallocManaged( &w_new_comp, 2*size*sizeof(float) ) );
	cuda_error_func( cudaMallocManaged( &w_save_comp, 2*size*sizeof(float) ) );
	
	// These four pointers point at the arrays with Fourier transform and allow
	// us to interpret two consecutive float elements as one cuComplex element.
	// This way we can use complex operators on the transformed arrays.
	cuComplex *w_appr_comp = reinterpret_cast< cuComplex* >(w_appr);
	cuComplex *vx_comp = reinterpret_cast< cuComplex* >(vx);
	cuComplex *vy_comp = reinterpret_cast< cuComplex* >(vy);
	cuComplex *arr_comp = reinterpret_cast< cuComplex* >(arr);
	
	// The arrays used for the integraing algorithm:
	float *lin, *lin2;
	cuComplex *nonl_comp;
	cuda_error_func( cudaMallocManaged( &lin, size*sizeof(float) ) );
	cuda_error_func( cudaMallocManaged( &lin2, size*sizeof(float) ) );
	cuda_error_func( cudaMallocManaged( &nonl_comp, 2*size*sizeof(float) ) );
	
	// The wavenumber related arrays: 
	float *cutoff, *kx, *ky, *k_squared;
	cuda_error_func( cudaMallocManaged( &kx, size*sizeof(float) ) );
	cuda_error_func( cudaMallocManaged( &ky, size*sizeof(float) ) );
	cuda_error_func( cudaMallocManaged( &k_squared, size*sizeof(float) ) );
	cuda_error_func( cudaMallocManaged( &cutoff, size*sizeof(float) ) );
	
	// the variables used for the mean velocity
	float lin_v, lin2_v;
	float *vx_mean_old, *vy_mean_old, *vx_mean_new, *vy_mean_new,
	      *vx_mean_appr, *vy_mean_appr, *nonl_vx, *nonl_vy;
	cuda_error_func( cudaMallocManaged( &vx_mean_old, sizeof(float) ) );
	cuda_error_func( cudaMallocManaged( &vy_mean_old, sizeof(float) ) );
	cuda_error_func( cudaMallocManaged( &vx_mean_new, sizeof(float) ) );
	cuda_error_func( cudaMallocManaged( &vy_mean_new, sizeof(float) ) );
	cuda_error_func( cudaMallocManaged( &vx_mean_appr, sizeof(float) ) );
	cuda_error_func( cudaMallocManaged( &vy_mean_appr, sizeof(float) ) );
	cuda_error_func( cudaMallocManaged( &nonl_vx, sizeof(float) ) );
	cuda_error_func( cudaMallocManaged( &nonl_vy, sizeof(float) ) );
	
/****************************************************************************** 
 * creating plans for the cuFFT library                                       * 
 ******************************************************************************/
	cout << "creating cuFFT plan" << endl;
	
	// We create plans for the Fourier transforms of Ny*Nx arrays. This process
	// is automated and includes memory allocation and management.
	cufftHandle transf;
	cufftHandle inv_transf;
	
	cufft_error_func( cufftPlan2d( &transf, Ny, Nx, CUFFT_R2C ) );
	cufft_error_func( cufftPlan2d( &inv_transf, Ny, Nx, CUFFT_C2R ) );
	
/****************************************************************************** 
 * wavenumbers and Dealiasing                                                 * 
 ******************************************************************************/
	cout <<"calculating the wavenumber arrays" << endl;
	
	wavenumber_func<<< BLOCK_NUM,BLOCK_SIZE >>>(kx, ky, k_squared, Lx, Ly, Nxh,
													Ny);
	cuda_error_func( cudaPeekAtLastError() );
	cutoff_func<<< BLOCK_NUM,BLOCK_SIZE >>>(cutoff, Nxh, Ny);
	cuda_error_func( cudaPeekAtLastError() );
	
	cuda_error_func( cudaDeviceSynchronize() );
	
/****************************************************************************** 
 * reading and processing of the initial condition                            * 
 ******************************************************************************/
	
	// Th etime step counting starts at 1, since zero is the initial condition.
	// This value may change if a previous simulation is continued.
	int m{1};
	
	// We initialize the mean velocity as zero. This may also change.
	*vx_mean_new = 0;
	*vy_mean_new = 0;
	
	if( initial.substr(0,6)=="random" )
	{
		cout << "generating random initial condition" << endl;
		// We set up a random number generator.
		int seed = chrono::system_clock::to_time_t( 
		                             chrono::high_resolution_clock::now() );
		// The seed is further modified if a number is given as well:
		if( initial.find(" ")!=string::npos )
		{
			seed /= stoi( initial.substr(initial.find(" ")+1) );
		}
		mt19937_64 generator(seed);
		uniform_real_distribution<float> distribution(-1., 1.);
		
		// We fill the initial vorticity field with random values in [-1,1]
		// and save it in w_appr[]. Due to padding, the last two elements of
		// every line are not used for the data.
		for( int r{0}; r<2*size; ++r )
		{
			if( r%(Nx+2)==Nx || r%(Nx+2)==Nx+1 )
				w_appr[r] = 0.f;
			else
				w_appr[r] = distribution(generator);
		}
	}
	else if( initial=="blank" )
	{
		cout << "generating blank initial condition" << endl;
		// We fill the initial vorticity field with zeros.
		for( int r{0}; r<2*size; ++r )
			w_appr[r] = 0.;
	}
	else if( initial.substr(0,8)=="continue" )
	{
		// We continue from a previous simulation. The last snapshot:
		int cont_snap{-1};
		for( int i{0}; i<1000000; ++i )
		{
			if( existence_check(directory, i) )
				cont_snap = i;
			else
				break;
		}
		// We stop the execution if no previous snapshot was found:
		if( cont_snap==-1 )
		{
			cerr << "Error: Tried to continue from previous simulation,"
			     << " but no previous snapshot was found." << endl;
			unallocate_func(transf, inv_transf, vx, vy, w_appr, arr, w_old_comp,
			                 w_new_comp, w_save_comp, w_appr_comp, vx_comp,
			                 vy_comp, arr_comp, lin, lin2, nonl_comp, cutoff,
			                 kx, ky, k_squared, vx_mean_old, vy_mean_old,
			                 vx_mean_new, vy_mean_new, vx_mean_appr,
			                 vy_mean_appr, nonl_vx, nonl_vy );
			return 1;
		}
		// Check for additional arguments (seperated from "continue" by a
		// blank):
		if( initial.find(" ")!=string::npos )
		{
			string in_args = initial.substr(initial.find(" ")+1);
			if( in_args.find(" ")!=string::npos )
			{
				// Two additional arguments: Time to extend and snapshot to
				// continue from.
				cont_snap = stoi( in_args.substr(in_args.find(" ")+1) );
				M = static_cast<int>( cont_snap*snap + std::round(
				      stof( in_args.substr(0, in_args.find(" ")) )/dt) );
			}
			else
			{
				// One additional argument: The time to extend.
				M = static_cast<int> (cont_snap*snap
				    + std::round(stof(in_args)/dt) );
			}
		}
		cout << "continue execution from snapshot " << cont_snap << endl; 
		m = cont_snap*snap;
		
		// We read the snapshot.
		string filename{ directory + "/ActTurb_snap" + to_string(cont_snap) + ".bin" };
		ifstream data_wf( filename, ios::in | ios::binary );
		if (data_wf.is_open())
		{
			// Since we look at the array in Fourier space, we use size:
			for (int c{0}; c<size; ++c)
			{
				data_wf >> w_new_comp[c].x;
				data_wf.ignore(1000, ',');
				data_wf >> w_new_comp[c].y;
				data_wf.ignore(1000, ',');
			}
			// The last two values of a snapshot are the two mean velocity
			// components.
			data_wf >> *vx_mean_new;
			data_wf.ignore(1000, ',');
			data_wf >> *vy_mean_new;
			
			data_wf.close();
		}
		else
		{
			cerr << "Error: unable to open previous simulation" << endl;
			unallocate_func(transf, inv_transf, vx, vy, w_appr, arr, w_old_comp,
			                w_new_comp, w_save_comp, w_appr_comp, vx_comp,
			                vy_comp, arr_comp, lin, lin2, nonl_comp, cutoff,
			                kx, ky, k_squared, vx_mean_old, vy_mean_old, 
			                vx_mean_new, vy_mean_new, vx_mean_appr,
			                vy_mean_appr, nonl_vx, nonl_vy );
			return 1;
		}
		
	}
	else
	{
		cerr << "Error: unknown specification for the initial condition" << endl;
		unallocate_func(transf, inv_transf, vx, vy, w_appr, arr, w_old_comp,
		                 w_new_comp, w_save_comp, w_appr_comp, vx_comp,
		                 vy_comp, arr_comp, lin, lin2, nonl_comp, cutoff,
		                 kx, ky, k_squared, vx_mean_old, vy_mean_old,
		                 vx_mean_new, vy_mean_new, vx_mean_appr,
		                 vy_mean_appr, nonl_vx, nonl_vy );
		return 1;
		
	}
	if( m==1 )
	{
		cout << "processing the initial condition" << endl;
		// We transform the initial condition. Now we have complex data, so we
		// use a pointer to the cuComplex type (w_appr_comp).
		cufft_error_func( cufftExecR2C(transf, w_appr, w_appr_comp) );
		cuda_error_func( cudaDeviceSynchronize() );
		
		// High frequncies (k > 2pi/L * N/2) are misinterpreted by DFT and
		// therefore must be set to zero (de-aliasing). We multiply with cutoff
		// and store the result in w_new_comp[].
		dealiasing_func<<< BLOCK_NUM, BLOCK_SIZE >>>(w_new_comp, w_appr_comp,
													cutoff, size);
		cuda_error_func( cudaPeekAtLastError() );
		cuda_error_func( cudaDeviceSynchronize() );
		
		// In Fourier space w[0] is proportinal to the mean vorticity. We set it
		// to zero, because in that case it can't change analytically and
		// becomes a control variable we can check throughout the simulation.
		w_new_comp[0] = make_cuComplex( 0.f, 0.f );
		
		// We save the transformed initial condition as first snapshot:
		if( save_func(w_new_comp, w_save_comp, vx_mean_new, vy_mean_new, size,
		              0, dt) )
		{
			cerr << "Error: unable to create snapshot file" << endl;
			unallocate_func(transf, inv_transf, vx, vy, w_appr, arr, w_old_comp,
			                w_new_comp, w_save_comp, w_appr_comp, vx_comp,
			                vy_comp, arr_comp, lin, lin2, nonl_comp, cutoff,
			                kx, ky, k_squared, vx_mean_old, vy_mean_old,
			                vx_mean_new, vy_mean_new, vx_mean_appr,
			                vy_mean_appr, nonl_vx, nonl_vy );
			return 1;
		}
	}
	
/******************************************************************************
 * calculating the linear part of the integrating factor                      *
 ******************************************************************************/
	cout <<"calculating the linear part of the integrating factor" << endl;
	
	lin_func<<< BLOCK_NUM,BLOCK_SIZE >>>(k_squared, lin, lin2, size, alpha, dt);
	cuda_error_func( cudaPeekAtLastError() );
	
	lin_v = exp( -(1+alpha) * dt );
	lin2_v = exp( -(1+alpha) * dt/2 );
	
/******************************************************************************
 * time stepping                                                              *
 ******************************************************************************/
	cout <<"initializing the time stepping" << endl;
	
	// We start measuring the computation time.
	auto starting_time = chrono::high_resolution_clock::now();
	chrono::duration<double> exe_time;
	
	// The time step counter m starts either at 1 or at a higher value if a
	// previous simulation is continued.
	for (; m<=M; ++m)
	{
		// First, the new vorticity (w_new_comp) from the last iteration becomes
		// the w_old_comp of this iteration, than it is copied to w_appr_comp to
		// calculate the 1st Runge-Kutta coefficient. In the last step, we begin
		// calculating a new w_new_comp with w_old_comp and lin.
		integrate_func_0<<< BLOCK_NUM,BLOCK_SIZE >>>(w_old_comp, w_new_comp,
		                                             w_appr_comp, lin, size);
		cuda_error_func( cudaPeekAtLastError() );
		cuda_error_func( cudaDeviceSynchronize() );
		// We do the same for the mean velocity.
		*vx_mean_old = *vx_mean_new;
		*vy_mean_old = *vy_mean_new;
		*vx_mean_appr = *vx_mean_new;
		*vy_mean_appr = *vy_mean_new;
		
		// We calculate of the 1st Runge-Kutta coefficient and save the current
		// mean energy.
		modified_nonl_func(w_appr, w_appr_comp, arr, arr_comp, vx, vx_comp, vy,
		          vy_comp, kx, ky, k_squared, cutoff, nonl_comp, size, Nx, Ny,
		          dt, transf, inv_transf, nonl_vx, nonl_vy, vx_mean_appr,
		          vy_mean_appr, m);
		
		// The 1st column should follow a certain symmetry, so that the array
		// in real space remains real. However, the algorithm will destroy this
		// symmetry over time due to numerical errors in nonl_func(). Therefore
		// we need to manually restore the symmetry.
		reality_func<<< BLOCK_NUM, BLOCK_SIZE >>>(w_new_comp, Nxh, Ny, size);
		cuda_error_func( cudaPeekAtLastError() );
		cuda_error_func( cudaDeviceSynchronize() );
		
		// We add the 1st coefficient to w_new_comp and calculate a new w_appr.
		integrate_func_1<<< BLOCK_NUM,BLOCK_SIZE >>>(w_old_comp, w_new_comp, 
		                           w_appr_comp, lin, lin2, nonl_comp, size, dt);
		cuda_error_func( cudaPeekAtLastError() );
		cuda_error_func( cudaDeviceSynchronize() );
		
		// We do the same for the mean velocity.
		*vx_mean_new += dt/6 * lin_v * *nonl_vx;
		*vy_mean_new += dt/6 * lin_v * *nonl_vy;
		*vx_mean_appr = lin2_v * ( *vx_mean_old + dt/2 * *nonl_vx );
		*vy_mean_appr = lin2_v * ( *vy_mean_old + dt/2 * *nonl_vy );
		
		// We calculate the 2nd Runge-Kutta coefficient using the new w_appr.
		nonl_func(w_appr, w_appr_comp, arr, arr_comp, vx, vx_comp, vy, vy_comp,
		          kx, ky, k_squared, cutoff, nonl_comp, size, dt, transf,
		          inv_transf, nonl_vx, nonl_vy, vx_mean_appr, vy_mean_appr);
		
		reality_func<<< BLOCK_NUM,BLOCK_SIZE >>>(w_new_comp, Nxh, Ny, size);
		cuda_error_func( cudaPeekAtLastError() );
		cuda_error_func( cudaDeviceSynchronize() );
		
		// We add the 2nd coefficient to w_new_comp and calculate a new w_appr.
		integrate_func_2<<< BLOCK_NUM,BLOCK_SIZE >>>(w_old_comp, w_new_comp,
		                                             w_appr_comp, lin2,
		                                             nonl_comp, size, dt);
		cuda_error_func( cudaPeekAtLastError() );
		cuda_error_func( cudaDeviceSynchronize() );
		
		// And the same for the mean velocity.
		*vx_mean_new += dt/3 * lin2_v * *nonl_vx;
		*vy_mean_new += dt/3 * lin2_v * *nonl_vy;
		*vx_mean_appr = lin2_v * ( *vx_mean_old + dt/2 * *nonl_vx );
		*vy_mean_appr = lin2_v * ( *vy_mean_old + dt/2 * *nonl_vy );
		
		// We calculate the 3rd Runge-Kutta coefficient.
		nonl_func(w_appr, w_appr_comp, arr, arr_comp, vx, vx_comp, vy, vy_comp,
		          kx, ky, k_squared, cutoff, nonl_comp, size, dt, transf,
		          inv_transf, nonl_vx, nonl_vy, vx_mean_appr, vy_mean_appr);
		
		reality_func<<< BLOCK_NUM, BLOCK_SIZE >>>(w_new_comp, Nxh, Ny, size);
		cuda_error_func( cudaPeekAtLastError() );
		cuda_error_func( cudaDeviceSynchronize() );
		
		// We add the 3rd coefficient to w_new_comp and calculate a new w_appr.
		integrate_func_3<<< BLOCK_NUM, BLOCK_SIZE >>>(w_old_comp, w_new_comp,
		                                              w_appr_comp, lin, lin2,
		                                              nonl_comp, size, dt);
		cuda_error_func( cudaPeekAtLastError() );
		cuda_error_func( cudaDeviceSynchronize() );
		// We do the same for the mean velocity.
		*vx_mean_new += dt/3 * lin2_v * *nonl_vx;
		*vy_mean_new += dt/3 * lin2_v * *nonl_vy;
		*vx_mean_appr = lin_v * ( *vx_mean_old + dt * *nonl_vx );
		*vy_mean_appr = lin_v * ( *vy_mean_old + dt * *nonl_vy );
		
		// We calculate the last Runge-Kutta coefficient.
		nonl_func(w_appr, w_appr_comp, arr, arr_comp, vx, vx_comp, vy, vy_comp,
		          kx, ky, k_squared, cutoff, nonl_comp, size, dt, transf,
		          inv_transf, nonl_vx, nonl_vy, vx_mean_appr, vy_mean_appr);
		
		reality_func<<< BLOCK_NUM, BLOCK_SIZE >>>(w_new_comp, Nxh, Ny, size);
		cuda_error_func( cudaPeekAtLastError() );
		cuda_error_func( cudaDeviceSynchronize() );
		
		// We ad the 4rd coefficient to w_new_comp.
		integrate_func_4<<< BLOCK_NUM,BLOCK_SIZE >>>(w_new_comp, nonl_comp,
		                                             size, dt);
		cuda_error_func( cudaPeekAtLastError() );
		cuda_error_func( cudaDeviceSynchronize() );
		
		// Again, we do the same with the mean velocity.
		*vx_mean_new += dt/6 * *nonl_vx;
		*vy_mean_new += dt/6 * *nonl_vy;
		
		// We regulary save a snapshots to a binary file.
		if( m%snap == 0 )
		{
			// We stop the programm if it can't save the snapshots.
			if (save_func(w_new_comp, w_save_comp, vx_mean_new, vy_mean_new,
			              size, m, dt) )
			{
				cerr << "Error: unable to create snapshot file" << endl;
				unallocate_func(transf, inv_transf, vx, vy, w_appr, arr,
				                w_old_comp, w_new_comp, w_save_comp,
				                w_appr_comp, vx_comp, vy_comp, arr_comp, lin,
				                lin2, nonl_comp, cutoff, kx, ky, k_squared,
				                vx_mean_old, vy_mean_old, vx_mean_new,
				                vy_mean_new, vx_mean_appr, vy_mean_appr,
				                nonl_vx, nonl_vy );
				return 1;
			}
			// We calculate the elapsed time.
			auto current_time = chrono::high_resolution_clock::now();
			exe_time = current_time - starting_time;
			
			cout << "\r" << m << " / " << M << " timesteps done in "
				<< exe_time.count() << "s, w[0]=" << "("
				<< cuCrealf(w_new_comp[0]) << "," << cuCimagf(w_new_comp[0])
				<< ")    " << flush;
		}
		// We stop the programm if it produces nan:
		if( check_func( w_new_comp ) )
		{
			cerr << "\nError: array contains nan values after " << m
				<< " timesteps" << endl;
			unallocate_func(transf, inv_transf, vx, vy, w_appr, arr,
			                w_old_comp, w_new_comp, w_save_comp,
			                w_appr_comp, vx_comp, vy_comp, arr_comp, lin,
			                lin2, nonl_comp, cutoff, kx, ky, k_squared,
			                vx_mean_old, vy_mean_old, vx_mean_new, vy_mean_new,
			                vx_mean_appr, vy_mean_appr, nonl_vx, nonl_vy );
			return 1;
		}
	}
	
/******************************************************************************
 * clean-up                                                                   *
 ******************************************************************************/
	unallocate_func( transf, inv_transf, vx, vy, w_appr, arr, w_old_comp,
	                 w_new_comp, w_save_comp, w_appr_comp, vx_comp, vy_comp,
	                 arr_comp, lin, lin2, nonl_comp, cutoff, kx, ky, k_squared,
	                 vx_mean_old, vy_mean_old, vx_mean_new, vy_mean_new,
	                 vx_mean_appr, vy_mean_appr, nonl_vx, nonl_vy );
	
	auto ending_time = chrono::high_resolution_clock::now();
	exe_time = ending_time - starting_time;
	cout << "\nprogram finished successfully after " << exe_time.count()
		<< "s.\n" << endl;
	return 0;
}

/******************************************************************************
 * declarations                                                               *
 ******************************************************************************/

inline bool check_func(cuComplex w_new_comp[])
{
	// We copy the value of w_new_comp[0] from device to host to prevent the
	// unified memory system from copying the entire array.
	// This may produce errors if, for some unknown reason, the unified data
	// already is on the cpu.
	cuda_error_func( cudaMemcpy( &w_check, w_new_comp, 2*sizeof(float),
								cudaMemcpyDeviceToHost ) );
	// check for nan-values
	if( cuCrealf(w_check) != cuCrealf(w_check) )
		return 1;
	else
		return 0;
}

__global__
void cutoff_func(float cutoff[], int Nxh, int Ny)
{
	int index{ static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
	int stride{ static_cast<int>(blockDim.x * gridDim.x) };
	
	// We use only the lower 1/2 of the wavenumbers, since the rest would be
	// misinterpreted by the DFT algorythm.
	for( int c{index}; c<Ny*Nxh; c+=stride )
	{
		int x{c%Nxh};
		int y{c/Nxh};
		int R_sq{(Nxh-1)*(Nxh-1) * Ny*Ny/4};
		if( x*x + y*y <= R_sq ) 
			cutoff[c] = 1;
		else if( x*x + (Ny-y)*(Ny-y) <= R_sq )
			cutoff[c] = 1;
		else
			cutoff[c] = 0;
	}
}

__global__
void dealiasing_func(cuComplex w_new_comp[], cuComplex w_appr_comp[],
                     float cutoff[], int size)
{
	int index{ static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
	int stride{ static_cast<int>(blockDim.x * gridDim.x) };
	
	// We cut of the greater half of the frequencies to avoid misinterpretation
	// by the DFT algorithm. This is called de-aliasing.
	for (int c{index}; c<size; c+=stride)
		w_new_comp[c] = w_appr_comp[c] * cutoff[c];
}

inline bool existence_check(string directory, int i)
{
	ifstream check_stream( directory + "/ActTurb_snap" + to_string(i) + ".bin" );
	return check_stream.good();
}

__global__
void integrate_func_0(cuComplex w_old_comp[], cuComplex w_new_comp[],
                      cuComplex w_appr_comp[], float lin[], int size)
{
	int index{ static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
	int stride{ static_cast<int>(blockDim.x * gridDim.x) };
	
	// We copy the current vorticity (w_new_comp) to w_old_comp and w_appr and
	// start calculating a new vorticity.
	for( int c{index}; c<size; c+=stride )
	{
		w_old_comp[c] = w_new_comp[c];
		w_appr_comp[c] = w_new_comp[c];
		w_new_comp[c] = w_new_comp[c] * lin[c];
	}
}

__global__
void integrate_func_1(cuComplex w_old_comp[], cuComplex w_new_comp[],
                      cuComplex w_appr_comp[], float lin[], float lin2[],
                      cuComplex nonl_comp[], int size, float dt)
{
	int index{ static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
	int stride{ static_cast<int>(blockDim.x * gridDim.x) };
	
	// We add the result of the 1st Runge-Kutta step to w_new_comp and calculate
	// the vorticity for the 2nd.
	for( int c{index}; c<size; c+=stride )
	{
		w_new_comp[c] = w_new_comp[c] + dt/6*lin[c] * nonl_comp[c];
		w_appr_comp[c] = lin2[c] * ( w_old_comp[c] + dt/2*nonl_comp[c] );
	}
}

__global__
void integrate_func_2(cuComplex w_old_comp[], cuComplex w_new_comp[],
                      cuComplex w_appr_comp[], float lin2[],
                      cuComplex nonl_comp[], int size, float dt)
{
	int index{ static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
	int stride{ static_cast<int>(blockDim.x * gridDim.x) };
	
	// We add the result of the 2nd Runge-Kutta step to w_new_comp and calculate
	// the vorticity for the 3rd.
	for( int c{index}; c<size; c+=stride )
	{
		w_new_comp[c] = w_new_comp[c] + dt/3 * lin2[c] * nonl_comp[c];
		w_appr_comp[c] = lin2[c]*( w_old_comp[c] + dt/2 * nonl_comp[c] );
	}
}

__global__
void integrate_func_3(cuComplex w_old_comp[], cuComplex w_new_comp[],
                      cuComplex w_appr_comp[], float lin[], float lin2[],
                      cuComplex nonl_comp[], int size, float dt)
{
	int index{ static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
	int stride{ static_cast<int>(blockDim.x * gridDim.x) };
	
	//We add the result of the 3rd Runge-Kutta step to w_new_comp and calculate
	// the vorticity for the 4th.
	for( int c{index}; c<size; c+=stride )
		{
			w_new_comp[c] = w_new_comp[c] + dt/3 * lin2[c] * nonl_comp[c];
			w_appr_comp[c] = lin[c] * ( w_old_comp[c] + dt*nonl_comp[c] );
		}
}

__global__
void integrate_func_4(cuComplex w_new_comp[], cuComplex nonl_comp[],
                      int size, float dt)
{
	int index{ static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
	int stride{ static_cast<int>(blockDim.x * gridDim.x) };
	
	//We add the result of the 3rd Runge-Kutta step to w_new_comp.
	for( int c{index}; c<size; c+=stride )
		w_new_comp[c] = w_new_comp[c] + dt/6 * nonl_comp[c];
}

__global__
void lin_func(float k_squared[], float lin[], float lin2[], int size,
              float alpha, float dt)
{
	int index{ static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
	int stride{ static_cast<int>(blockDim.x * gridDim.x) };
	
	// We calculate the linear part of the integrating factor.
	for( int c{index}; c<size; c+=stride )
	{
		lin[c] = exp( (-(1-k_squared[c])*(1-k_squared[c]) -alpha) * dt );
		lin2[c] = exp( (-(1-k_squared[c])*(1-k_squared[c]) -alpha) * dt/2 );
	}
}

/******************************************************************************/
inline void nonl_func(float w_appr[], cuComplex w_appr_comp[], float arr[],
                      cuComplex arr_comp[], float vx[], cuComplex vx_comp[],
                      float vy[], cuComplex vy_comp[], float kx[], float ky[],
                      float k_squared[], float cutoff[],
                      cuComplex nonl_comp[], int size, float dt,
                      cufftHandle transf, cufftHandle inv_transf,
                      float *nonl_vx, float *nonl_vy, float *vx_mean_appr,
                      float *vy_mean_appr)
{
	// We calculate both components of the velocety v by utilizing a stream
	// function. (Unfortunately, we lose the mean velocity in the process, so we
	// have to seperately integrate it.)
	nonl_subfunc_0<<< BLOCK_NUM,BLOCK_SIZE >>>(w_appr_comp, vx_comp, vy_comp,
												kx, ky, k_squared, size);
	cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
	// We add the mean velocity (v_comp[0] currently is zero):
	vx_comp[0] = make_cuComplex( *vx_mean_appr, 0.f );
	vy_comp[0] = make_cuComplex( *vy_mean_appr, 0.f );
	
	// We go to real space.
	cufft_error_func( cufftExecC2R(inv_transf, vx_comp, vx) );
	cufft_error_func( cufftExecC2R(inv_transf, vy_comp, vy) );
	cufft_error_func( cufftExecC2R(inv_transf, w_appr_comp, w_appr) );
	cuda_error_func( cudaDeviceSynchronize() );
	
	// We calculate the 4 terms within the nonlinear part of the integrating
	// factor. They are then stored in arr, w_appr, vx, and vy.
	nonl_subfunc_1<<< BLOCK_NUM,BLOCK_SIZE >>>(w_appr, arr, vx, vy, size);
	cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
	
	// We go back to Fourier space.
	cufft_error_func( cufftExecR2C(transf, w_appr, w_appr_comp) );
	cufft_error_func( cufftExecR2C(transf, arr, arr_comp) );
	cufft_error_func( cufftExecR2C(transf, vx, vx_comp) );
	cufft_error_func( cufftExecR2C(transf, vy, vy_comp) );
	cuda_error_func( cudaDeviceSynchronize() );
	
	// We calculate the entire nonlinear part in fourier space, using the
	// terms previously calculated in real space.
	nonl_subfunc_2<<< BLOCK_NUM,BLOCK_SIZE >>>(w_appr_comp, arr_comp, vx_comp,
											vy_comp, nonl_comp, kx, ky, cutoff,
											lambda, beta, size);
	cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
	
	// We calculate the time derivative of the mean velocity. (Actually, we 
	// calculate the time derivative of the 0th Fourier mode, which yields the
	// identity: N^2*<v> = F[v](k+0) )
	*nonl_vx = -beta * cuCrealf(vx_comp[0]);
	*nonl_vy = -beta * cuCrealf(vy_comp[0]);
}

inline void modified_nonl_func(float w_appr[], cuComplex w_appr_comp[], float arr[],
						cuComplex arr_comp[], float vx[], cuComplex vx_comp[],
						float vy[], cuComplex vy_comp[], float kx[], float ky[],
						float k_squared[], float cutoff[],
						cuComplex nonl_comp[], int size, int Nx, int Ny, float dt,
						cufftHandle transf, cufftHandle inv_transf,
						float *nonl_vx, float *nonl_vy, float *vx_mean_appr,
						float *vy_mean_appr, int m)
{
	// We calculate both components of the velocety v by utilizing a stream
	// function. (Unfortunately, we lose the mean velocity in the process, so we
	// have to seperately integrate it.)
	nonl_subfunc_0<<< BLOCK_NUM,BLOCK_SIZE >>>(w_appr_comp, vx_comp, vy_comp,
												kx, ky, k_squared, size);
	cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
	// We add the mean velocity (v_comp[0] currently is zero):
	vx_comp[0] = make_cuComplex( *vx_mean_appr, 0.f );
	vy_comp[0] = make_cuComplex( *vy_mean_appr, 0.f );
	
	// We go to real space.
	cufft_error_func( cufftExecC2R(inv_transf, vx_comp, vx) );
	cufft_error_func( cufftExecC2R(inv_transf, vy_comp, vy) );
	cufft_error_func( cufftExecC2R(inv_transf, w_appr_comp, w_appr) );
	cuda_error_func( cudaDeviceSynchronize() );
	
	//##########################################################################
	// Here comes the modification.
	float E_mean{ 0 };
	for( int r{0}; r<2*size; r++)
		// We don't count the two columns with the padding.
		if( r%(Nx+2)==Nx || r%(Nx+2)==Nx+1 )
			continue;
		else
			E_mean += vx[r]*vx[r] + vy[r]*vy[r];
	// We divide by the number of grid points twice to account for the Fourier-
	// transform and the mean, and by 2 because of the definition of energy.
	E_mean /= (2.f*Nx)*Ny*Nx*Ny*Nx*Ny;
	// We save the mean energy
	if( save_energy_func(E_mean, m, dt) )
	{
		cerr << "Error: failed to save the mean energy" << endl;
		terminate();
	}
	
	// We calculate the 4 terms within the nonlinear part of the integrating
	// factor. They are then stored in arr, w_appr, vx, and vy.
	nonl_subfunc_1<<< BLOCK_NUM,BLOCK_SIZE >>>(w_appr, arr, vx, vy, size);
	cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
	
	// We go back to Fourier space.
	cufft_error_func( cufftExecR2C(transf, w_appr, w_appr_comp) );
	cufft_error_func( cufftExecR2C(transf, arr, arr_comp) );
	cufft_error_func( cufftExecR2C(transf, vx, vx_comp) );
	cufft_error_func( cufftExecR2C(transf, vy, vy_comp) );
	cuda_error_func( cudaDeviceSynchronize() );
	
	// We calculate the entire nonlinear part in fourier space, using the
	// terms previously calculated in real space.
	nonl_subfunc_2<<< BLOCK_NUM,BLOCK_SIZE >>>(w_appr_comp, arr_comp, vx_comp,
											vy_comp, nonl_comp, kx, ky, cutoff,
											lambda, beta, size);
	cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
	
	// We calculate the time derivative of the mean velocity. (Actually, we 
	// calculate the time derivative of the 0th Fourier mode, which yields the
	// identity: N^2*<v> = F[v](k+0) )
	*nonl_vx = -beta * cuCrealf(vx_comp[0]);
	*nonl_vy = -beta * cuCrealf(vy_comp[0]);
}

__global__
void nonl_subfunc_0(cuComplex w_appr_comp[], cuComplex vx_comp[],
					 cuComplex vy_comp[], float kx[], float ky[],
					 float k_squared[], int size)
{
	int index{ static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
	int stride{ static_cast<int>(blockDim.x * gridDim.x) };
	
	// We consider a stream function psi with: omega = laplace(psi)
	// => vx = -d_y psi, vy = d_x psi
	// In Fourier space: omega = -k_suqared * psi, vx = -ky * psi, vy = kx * psi
	// This can be rewrittem to get vx(omega) and vy(omega).
	// The case k=0 is handled seperately, since it would be v = 0 * 0 / 0.
	for( int c{index+1}; c<size; c+=stride )
	{
		vx_comp[c] = w_appr_comp[c] * i() * ky[c] / k_squared[c];
		vy_comp[c] = w_appr_comp[c] * i() * -kx[c] / k_squared[c];
	}
}

__global__
void nonl_subfunc_1(float w_appr[], float arr[], float vx[], float vy[],
					int size)
{
	int index{ static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
	int stride{ static_cast<int>(blockDim.x * gridDim.x) };
	
	// The 1st non-linear term is stored in arr[] and the 2nd term in w_appr[],
	// which we won't need it anymore.
	for( int r{index}; r<2*size; r+=stride )
	{
		arr[r] = w_appr[r] * vx[r];
		w_appr[r] = w_appr[r] * vy[r];
	}
	// For the same reason we use vx[] to store the 3rd nonlinear term and vy[]
	// to store the 4th nonlinear term.
	float vel_squared{ 0 };
	for( int r{index}; r<2*size; r+=stride )
	{
		vel_squared = vx[r]*vx[r] + vy[r]*vy[r];
		vx[r] *= vel_squared;
		vy[r] *= vel_squared;
	}
}

__global__
void nonl_subfunc_2(cuComplex w_appr_comp[], cuComplex arr_comp[],
					cuComplex vx_comp[], cuComplex vy_comp[],
					cuComplex nonl_comp[], float kx[], float ky[],
					float cutoff[], float lambda, float beta, int size)
{
	int index{ static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
	int stride{ static_cast<int>(blockDim.x * gridDim.x) };
	
	// We add all parts of the nonlinear term and multiply them with constants.
	// These constants contain physical properties as well as the normalization.
	for( int c{index}; c<size; c+=stride )
	{
		nonl_comp[c] = i() * ( -kx[c] * (lambda*arr_comp[c] + beta*vy_comp[c])
						-ky[c] * (lambda*w_appr_comp[c] - beta*vx_comp[c])
						)*cutoff[c];
	}
}
/******************************************************************************/

__global__
void reality_func(cuComplex w_new_comp[], int Nxh, int Ny, int size)
{
	int index{ static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
	int stride{ static_cast<int>(blockDim.x * gridDim.x) };
	
	// We make sure that the transformed array keeps the necessary symmetry for
	// RFT. (The 1st column should consist of pairs of complex conjugates.)
	cuComplex mean_value{ 0.f, 0.f };
	for( int y{(index+1)*Nxh}; y<(Ny/2*Nxh); y+=stride*Nxh )
	{
		mean_value = 0.5f * ( w_new_comp[y] + cuConjf(w_new_comp[size-y]) );
		w_new_comp[y] = mean_value;
		w_new_comp[size-y] = cuConjf(mean_value);
	}
}

bool save_func(cuComplex w_new_comp[], cuComplex w_save_comp[],
				float *vx_mean_new, float *vy_mean_new, int size, int m,
				float dt)
{
	// We save a snapshot of the (transformed) system and vx_mean, vy_mean as
	// last two values.
	string filename{ directory + "/ActTurb_snap" + to_string(m/snap) + ".bin" };
	ofstream data_file( filename, ios::out | ios::binary );
	if( !data_file.is_open() )
		return 1;
	// We copy the data from the GPU to the CPU. (If we let the unified memory
	// handle this, it will move the array to the CPU, print into the file and
	// then move everything back to the GPU. Here the data stays on the GPU
	// and we copy only once.)
	cuda_error_func( cudaMemcpy( w_save_comp, w_new_comp, 2*size*sizeof(float),
	                             cudaMemcpyDeviceToHost ) );
	for (int c{0}; c<size; ++c)
		data_file << w_save_comp[c].x << "," << w_save_comp[c].y << ",";
	data_file << *vx_mean_new << "," << *vy_mean_new;
	data_file.close();
	return 0;
}

bool save_energy_func( float E_mean, int m, float dt)
{
	// We save the current time and mean energy.
	string filename{ directory + "/energy.txt" };
	ofstream data_file( filename, ios::out | ios::app );
	if( !data_file.is_open() )
		return 1;
	data_file << m*dt << " " << E_mean << endl;
	data_file.close();
	return 0;
}

void unallocate_func( cufftHandle transf, cufftHandle inv_transf, float vx[],
                      float vy[], float w_appr[], float arr[],
                      cuComplex w_old_comp[], cuComplex w_new_comp[],
                      cuComplex w_save_comp[], cuComplex *w_appr_comp,
                      cuComplex *vx_comp, cuComplex *vy_comp,
                      cuComplex *arr_comp, float lin[], float lin2[],
                      cuComplex nonl_comp[], float cutoff[], float kx[],
                      float ky[], float k_squared[], float *vx_mean_old,
                      float *vy_mean_old, float *vx_mean_new,
                      float *vy_mean_new, float *vx_mean_appr,
                      float *vy_mean_appr, float *nonl_vx, float *nonl_vy )
{
	// We unallocate the transformation plans (arr2 has no transformations).
	cufft_error_func( cufftDestroy(transf) );
	cufft_error_func( cufftDestroy(inv_transf) );
	
	// W set all pointers to null before freeing the adress they point at.
	w_appr_comp = nullptr;
	vx_comp = nullptr;
	vy_comp = nullptr;
	arr_comp = nullptr;
	
	// We unallocate the arrays.
	cuda_error_func( cudaFree( vx ) );
	cuda_error_func( cudaFree( vy ) );
	cuda_error_func( cudaFree( w_appr ) );
	cuda_error_func( cudaFree( arr ) );
	cuda_error_func( cudaFree( w_old_comp ) );
	cuda_error_func( cudaFree( w_new_comp ) );
	cuda_error_func( cudaFree( w_save_comp ) );
	cuda_error_func( cudaFree( lin ) );
	cuda_error_func( cudaFree( lin2 ) );
	cuda_error_func( cudaFree( nonl_comp ) );
	cuda_error_func( cudaFree( kx ) );
	cuda_error_func( cudaFree( ky ) );
	cuda_error_func( cudaFree( k_squared ) );
	cuda_error_func( cudaFree( cutoff ) );
	
	cuda_error_func( cudaFree( vx_mean_old ) );
	cuda_error_func( cudaFree( vy_mean_old ) );
	cuda_error_func( cudaFree( vx_mean_new ) );
	cuda_error_func( cudaFree( vy_mean_new ) );
	cuda_error_func( cudaFree( vx_mean_appr ) );
	cuda_error_func( cudaFree( vy_mean_appr ) );
	cuda_error_func( cudaFree( nonl_vx ) );
	cuda_error_func( cudaFree( nonl_vy ) );
}


__global__
void wavenumber_func(float kx[], float ky[], float k_squared[], float Lx,
						float Ly, int Nxh, int Ny)
{
	int index{ static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) };
	int stride{ static_cast<int>(blockDim.x * gridDim.x) };
	
	// The wavenumbers are stored as [k0, k1, k2, ..., k(N/2)] in x-direction
	// and as [k0, k1, k2, ..., k(N/2), k(-N/2+1), ... , k(-1)] in y-direction.
	// ky increases along the first axis, kx along the second, k_squared = |k|^2
	for( int c{index}; c<Ny*Nxh; c+=stride )
	{
		if( (c/Nxh)<(Ny/2+1))
			ky[c] = 2*_pi/Ly * (c/Nxh);
		else
			ky[c] = 2*_pi/Ly * (c/Nxh - Ny);
		
		kx[c] = 2*_pi/Lx * (c%Nxh);
		k_squared[c] = ky[c]*ky[c] + kx[c]*kx[c];
	}
}

/******************************************************************************
 * THE END                                                                    *
 ******************************************************************************/
