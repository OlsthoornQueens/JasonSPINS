/* Script for the formation of a Kelvin-Helmholtz billow
 * without topography */

/* ------------------ Top matter --------------------- */

// Required headers
#include "../../BaseCase.hpp"      // Support file containing default implementations of several functions
#include "../../Options.hpp"       // config-file parser
#include <random/normal.h>         // Blitz random number generator


using namespace ranlib;

// Tensor variables for indexing
blitz::firstIndex ii;
blitz::secondIndex jj;
blitz::thirdIndex kk;

/* ------------------ Define parameters --------------------- */

// Grid scales
double Lx, Ly, Lz;              // Grid lengths (m)
int    Nx, Ny, Nz;              // Number of points in x, y, z
double MinX, MinY, MinZ;        // Minimum x/y/z points (m)
// Grid types
DIMTYPE intype_x, intype_y, intype_z;
string grid_type[3];

// Physical parameters
double g, rho_0;                // gravity accel (m/s^2), reference density (kg/m^3)
double visco;                   // viscosity (m^2/s)
double mu;                      // dynamic viscosity (kg/(m·s))
double kappa;               // diffusivity of density (m^2/s)
// helpful constants
const int Num_tracers = 1;      // number of tracers (density and dyes)
const int T = 0;              // index for rho

// Problem parameters
double T_offset;               // Initial Temperature of the lower layer 
double sig_T0;                 // Initial interface width
double Tz_bot;                 // Bottom Gradient
double T_top;                  // Top T value


// Temporal parameters
double final_time;              // Final time (s)
double plot_interval;           // Time between field writes (s)
double havg_interval;           // Time between avg writes (s)
double slice_interval;           // Time between slice writes (s)
int xy_slice_height;            // vertical position of the XY slice in grid index
double dt_max;                  // maximum time step (s)

// Restarting options
bool restarting;                // are you restarting?
double initial_time;            // initial start time of simulation
int restart_sequence;           // output number to restart from

// Dump parameters
bool restart_from_dump;         // restarting from dump?
double compute_time;            // requested computation time
double avg_write_time;          // average time to write all output fields at one output
double real_start_time;         // real (clock) time when simulation begins
double compute_start_time;      // real (clock) time when computation begins (after initialization)

// other options
double perturb;                 // Initial velocity perturbation

bool compute_enstrophy;         // Compute enstrophy?
bool compute_dissipation;       // Compute dissipation?
bool compute_BPE;               // Compute background potential energy?
bool compute_internal_to_BPE;   // Compute BPE gained from internal energy?
bool compute_stresses_top;      // Compute top surface stresses?
bool compute_stresses_bottom;   // Compute bottom surface stresses?
bool write_pressure;            // Write the pressure field?
int iter = 0;                   // Iteration counter

// Maximum squared buoyancy frequency
double N2_max;


////////////////////////////////////
/////////// Import Files ///////////
////////////////////////////////////

// Input file names
string xgrid_filename,
       ygrid_filename,
       zgrid_filename,
       u_filename,
       v_filename,
       w_filename,
       rho_filename,
       tracer_filename;




////////////////////////////////////
/////// Horiz. Avg. and Slices//////
////////////////////////////////////

// Horizontally Averaged Files 
int prev_chain_write_count = 0;
int chain_write_count;
int slice_write_count = 0; 
int offset_from, offset_to;
MPI_Status status;

const int num_h_saves = 16; 
const int D           = 14; 
const int Xi          = 15; 
MPI_File temp_file[num_h_saves];
MPI_File final_file[num_h_saves];
char file_names_ordered[num_h_saves][20] 
    = {"Ubar.bin","Vbar.bin","Wbar.bin","Tbar.bin",
    "Urms.bin","Vrms.bin","Wrms.bin","Trms.bin",
    "UVp.bin","UWp.bin","VWp.bin",
    "UTp.bin","VTp.bin","WTp.bin",
    "Dbar.bin","XiBar.bin",
    };
char file_temp_names_ordered[num_h_saves][20]  
    = {"Ubar_temp.bin","Vbar_temp.bin","Wbar_temp.bin","Tbar_temp.bin",
    "Urms_temp.bin","Vrms_temp.bin","Wrms_temp.bin","Trms_temp.bin",
    "UVp_temp.bin","UWp_temp.bin","VWp_temp.bin",
    "UTp_temp.bin","VTp_temp.bin","WTp_temp.bin",
    "Dbar_temp.bin","XiBar_temp.bin",
    };

//Slice Files
const int num_XZslices = 5; 
const int num_XYslices = 5; 
const int U_SLICE = 0; 
const int V_SLICE = 1; 
const int W_SLICE = 2; 
const int T_SLICE = 3; 
const int D_SLICE = 4; 
int slice_bin_count = 0;
const int SLICES_PER_BIN = 100;

MPI_File slice_temp_file_XZ[num_XZslices];
MPI_File slice_temp_file_XY[num_XYslices];
char slice_names_XZ[num_XZslices][20] = {"u_slice","v_slice","w_slice","T_slice","D_slice"};
char slice_names_XY[num_XYslices][20] = {"u_slice","v_slice","w_slice","T_slice","D_slice"};


////////////////////////////////////
////////////////////////////////////


/* ------------------ Adjust the class --------------------- */

class userControl : public BaseCase {
    public:
        // Grid and topography arrays
        Array<double,1> xx, yy, zz;     // 1D grid vectors
        DTArray *Hprime;                // derivative of topography vector

        // Arrays and operators for derivatives
        Grad * gradient_op;
        DTArray *temp1,*diss,*density, *dxdydz;

        // Timing variables (for outputs and measuring time steps)
        int plot_number;        // plot output number
        double next_plot;       // time of next output write
        double havg_next_plot;  // time of next h-avg write
        double slice_next_plot; // time of next slice write

        double comp_duration;   // clock time since computation began
        double clock_time;      // current clock time

        // Size of domain
        double length_x() const { return Lx; }
        double length_y() const { return Ly; }
        double length_z() const { return Lz; }

        // Resolution in x, y, and z
        int size_x() const { return Nx; }
        int size_y() const { return Ny; }
        int size_z() const { return Nz; }

        // Set expansions (FREE_SLIP, NO_SLIP (in vertical) or PERIODIC)
        DIMTYPE type_x() const { return intype_x; }
        DIMTYPE type_y() const { return intype_y; }
        DIMTYPE type_z() const { return intype_z; }

        // Record the gradient-taking object
        void set_grad(Grad * in_grad) { gradient_op = in_grad; }

        // Coriolis parameter, viscosity, and diffusivities
        double get_visco() const { return visco; }
        double get_diffusivity(int t_num) const {
            return kappa;
        }

        // Temporal parameters
        double init_time() const { return initial_time; }
        int get_restart_sequence() const { return restart_sequence; }
        double get_dt_max() const { return dt_max; }
        double get_next_plot() { return havg_next_plot; }

        // Number of tracers (the first is density)
        int numtracers() const { return Num_tracers; }

        /* Initialize velocities */
        void init_vels(DTArray & u, DTArray & v, DTArray & w) {
            if (master()) fprintf(stdout,"Initializing velocities\n");
            // if restarting
            if (restarting and !restart_from_dump) {
                init_vels_restart(u, v, w);
 
            } else if (restarting and restart_from_dump) {
                init_vels_dump(u, v, w);
            } else{
                

                // Add a random perturbation to trigger any 3D instabilities
                int myrank;
                MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
                Normal<double> rnd(0,1);
                for (int i = u.lbound(firstDim); i <= u.ubound(firstDim); i++) {
                    rnd.seed(i);
                    for (int j = u.lbound(secondDim); j <= u.ubound(secondDim); j++) {
                        for (int k = u.lbound(thirdDim); k <= u.ubound(thirdDim); k++) {
                            u(i,j,k) += perturb*rnd.random();
                            w(i,j,k) += perturb*rnd.random();
                            if (Ny > 1) 
                                    v(i,j,k) += perturb*rnd.random();
                            
                        }
                    }
                }

                // Write the arrays
                write_array(u,"u",plot_number);
                write_array(w,"w",plot_number);
                if (Ny > 1) {
                    write_array(v,"v",plot_number);
                }
            }
        }

        /* Initialize the tracers (density and dyes) */
        void init_tracers(vector<DTArray *> & tracers) {
            if (master()) fprintf(stdout,"Initializing tracers\n");
            DTArray & temperature = *tracers[T];
            
            if (restarting and !restart_from_dump) {
                init_tracer_restart("T",temperature);
            } else if (restarting and restart_from_dump) {
                init_tracer_dump("T",temperature);
            } else {
                // Density configuration
                temperature = T_offset;
                // Write the arrays
                write_array(*tracers[T],"T",plot_number);
            }
        }


        /*
        Density is defined with a quadtratic equation of state. 
        To avoid duplicate computations, we are defining the public variable 
        -- density -- 
        which will be used in the analysis function.
        */
        void get_density(DTArray & temperature,DTArray & density ){
            // DTArray *rho = alloc_array(Nx,Ny,Nz);
            // // rho *= -1*rho;
            // density = rho;
            

            // density = -pow2(temperature);
            density = -pow2(temperature); 


        }

        /* Forcing in the momentum equations */
        void forcing(double t, const DTArray & u, DTArray & u_f,
                const DTArray & v, DTArray & v_f, const DTArray & w, DTArray & w_f,
                vector<DTArray *> & tracers, vector<DTArray *> & tracers_f) {
            
            


            // DTArray & density = *tracers[T];
            // get_density(*tracers[T],density);
            get_density(*tracers[T],*density);


            u_f = 0;
            v_f = 0;
            w_f = -g*(*density);   // tracers[T] is defined as rho/rho_0
            
            blitz::Range all = blitz::Range::all();
            *tracers_f[T] = 0;
            (*tracers_f[T])(all,all,0) = Tz_bot;    // Bottom
            (*tracers_f[T])(all,all,Nz-1) = T_top;  // Top
        }



        ////////////////////////
        /////// Tracer Forcing //////
        ////////////////////////

        
        // Turn on Different BCs
        bool diffBCs() const{
            return true; 
        }
        // Setup Neumann BCs
        void tracer_top_bc_z(int t_num, double & dir, double & neu) const {
            dir = 1;
            neu = 0;
        }
        // Setup Neumann BCs
        void tracer_bottom_bc_z(int t_num, double & dir, double & neu) const {
            dir = 0;
            neu = 1;
        }

        // void tracer_bc_z(int t_num, double & dir, double & neu) const {
        //     // Set up Robin-type BCs
        //     dir = 1;
        //     neu = 0;
        // }

        // Turn on Forcing
        bool tracer_bc_forcing() const {
            return true;
        }

        



    

        ////////////////////////
        /////// Analysis //////
        ////////////////////////

        /* Basic analysis: compute secondary variables, and save fields and diagnostics */
        void analysis(double time, DTArray & u, DTArray & v, DTArray & w,
                vector<DTArray *> & tracers, DTArray & pressure) {
            // Set-up
            if ( iter == 0 ) {

                density = alloc_array(Nx,Ny,Nz);
            

                if ( compute_enstrophy or compute_dissipation or
                        compute_stresses_top or compute_stresses_bottom ) {
                    temp1 = alloc_array(Nx,Ny,Nz);
                }
                if (compute_dissipation){
                    diss = alloc_array(Nx,Ny,Nz);

                }
                if ( compute_stresses_top or compute_stresses_bottom ) {
                    // initialize the vector of the bottom slope (Hprime)
                    Hprime = alloc_array(Nx,Ny,1);
                    *Hprime = 0*ii + 0*jj;
                }
                // Determine last plot if restarting from the dump file
                if (restart_from_dump) {
                    next_plot = (restart_sequence+1)*plot_interval;
                }
                // initialize the size of each voxel
                dxdydz = alloc_array(Nx,Ny,Nz);
                *dxdydz = (*get_quad_x())(ii)*(*get_quad_y())(jj)*(*get_quad_z())(kk);
            }
            // update clocks
            if (master()) {
                clock_time = MPI_Wtime();
                comp_duration = clock_time - compute_start_time;
            }

            /* Calculate and write out useful information */

            // Energy (PE assumes density is density anomaly)
            double ke_x = 0, ke_y = 0, ke_z = 0;
            if ( Nx > 1 ) {
                ke_x = pssum(sum(0.5*rho_0*(u*u)*(*dxdydz)));
            }
            if ( Ny > 1 ) {
                ke_y = pssum(sum(0.5*rho_0*(v*v)*(*dxdydz)));
            }
            if ( Nz > 1 ) {
                ke_z = pssum(sum(0.5*rho_0*(w*w)*(*dxdydz)));
            }

            // DTArray & density = *tracers[T];
            // get_density(*tracers[T],density);
            get_density(*tracers[T],*density);

            double pe_tot;
            pe_tot = pssum(sum(rho_0*(1+(*density))*g*(zz(kk) - MinZ)*(*dxdydz)));
            double BPE_tot = 0;
            if (compute_BPE) {
                compute_Background_PE(BPE_tot,*density, *dxdydz, Nx, Ny, Nz, Lx, Ly, Lz,
                        g, rho_0, iter);
            }
            // Conversion from internal energy to background potential energy
            double phi_i = 0;
            if (compute_internal_to_BPE) {
                compute_BPE_from_internal(phi_i, *density, kappa, rho_0, g, Nz);
            }
            // viscous dissipation
            double diss_tot = 0;
            double max_diss = 0;
            if (compute_dissipation) {
                dissipation(*diss, u, v, w, gradient_op, grid_type, Nx, Ny, Nz, mu);

                max_diss = psmax(max(*diss));
                diss_tot = pssum(sum((*diss)*(*dxdydz)));
            }
            // Vorticity / Enstrophy
            double max_vort_x = 0, enst_x_tot = 0;
            double max_vort_y = 0, enst_y_tot = 0;
            double max_vort_z = 0, enst_z_tot = 0;
            if (compute_enstrophy) {
                // x-vorticity
                if (Ny > 1 and Nz > 1) {
                    compute_vort_x(*temp1, v, w, gradient_op, grid_type);
                    max_vort_x = psmax(max(abs(*temp1)));
                    enst_x_tot = pssum(sum(0.5*pow(*temp1,2)*(*dxdydz)));
                }
                // y-vorticity
                if (Nx > 1 and Nz > 1) {
                    compute_vort_y(*temp1, u, w, gradient_op, grid_type);
                    max_vort_y = psmax(max(abs(*temp1)));
                    enst_y_tot = pssum(sum(0.5*pow(*temp1,2)*(*dxdydz)));
                }
                // z-vorticity
                if (Nx > 1 and Ny > 1) {
                    compute_vort_z(*temp1, u, v, gradient_op, grid_type);
                    max_vort_z = psmax(max(abs(*temp1)));
                    enst_z_tot = pssum(sum(0.5*pow(*temp1,2)*(*dxdydz)));
                }
            }
            // max of fields
            double max_u = psmax(max(abs(u)));
            double max_v = psmax(max(abs(v)));
            double max_w = psmax(max(abs(w)));
            double max_vel = psmax(max(sqrt(u*u + v*v + w*w)));

            // Maximum abs density
            double max_rho = psmax(max(abs(*density)));
            double max_temperature = psmax(max(abs(*tracers[T])));
            double mass = pssum(sum(rho_0*(1+(*density))*(*dxdydz)));

            if (master()) {
                // add diagnostics to buffers
                string header, line;
                add_diagnostic("Iter", iter,            header, line);
                add_diagnostic("Clock_time", comp_duration, header, line);
                add_diagnostic("Time", time,            header, line);
                add_diagnostic("Max_vel", max_vel,      header, line);
                add_diagnostic("Max_density", max_rho,  header, line);
                add_diagnostic("Mass", mass,            header, line);
                add_diagnostic("PE_tot", pe_tot,        header, line);
                if (compute_BPE) {
                    add_diagnostic("BPE_tot", BPE_tot,  header, line);
                }
                if (compute_internal_to_BPE) {
                    add_diagnostic("BPE_from_int", phi_i,   header, line);
                }
                if (compute_dissipation) {
                    add_diagnostic("Max_diss", max_diss,    header, line);
                    add_diagnostic("Diss_tot", diss_tot,    header, line);
                }
                if (Nx > 1) {
                    add_diagnostic("Max_u", max_u,  header, line);
                    add_diagnostic("KE_x", ke_x,    header, line);
                }
                if (Ny > 1) {
                    add_diagnostic("Max_v", max_v,  header, line);
                    add_diagnostic("KE_y", ke_y,    header, line);
                }
                if (Nz > 1) {
                    add_diagnostic("Max_w", max_w,  header, line);
                    add_diagnostic("KE_z", ke_z,    header, line);
                }
                if (Ny > 1 && Nz > 1 && compute_enstrophy) {
                    add_diagnostic("Enst_x_tot", enst_x_tot, header, line);
                    add_diagnostic("Max_vort_x", max_vort_x, header, line);
                }
                if (Nx > 1 && Nz > 1 && compute_enstrophy) {
                    add_diagnostic("Enst_y_tot", enst_y_tot, header, line);
                    add_diagnostic("Max_vort_y", max_vort_y, header, line);
                }
                if (Nx > 1 && Ny > 1 && compute_enstrophy) {
                    add_diagnostic("Enst_z_tot", enst_z_tot, header, line);
                    add_diagnostic("Max_vort_z", max_vort_z, header, line);
                }

                // Write to file
                if (!(restarting and iter==0))
                    write_diagnostics(header, line, iter, restarting);
                // and to the log file
                fprintf(stdout,"[%d] (%.4g) %.4f: "
                        "%.4g %.4g %.4g %.4g\n",
                        iter,comp_duration,time,
                        max_u,max_v,max_w,max_temperature);
            }

            // Top Surface Stresses
            if ( compute_stresses_top ) {
                stresses_top(u, v, w, *Hprime, *temp1, gradient_op, grid_type, mu, time, iter, restarting);
            }
            // Bottom Surface Stresses
            if ( compute_stresses_bottom ) {
                stresses_bottom(u, v, w, *Hprime, *temp1, gradient_op, grid_type, mu, time, iter, restarting);
            }



            ////////////////////////
            /////// Write out //////
            ////////////////////////

            


            /* SLICE --  Write slices to disk if at correct time */
            if ((time - slice_next_plot) > -1e-6) {
                
                verify_slice_files();           // Ensure the the MPI files are open correctly; 
                write_slice_XZ(u,slice_temp_file_XZ[U_SLICE],0);
                write_slice_XZ(v,slice_temp_file_XZ[V_SLICE],0);
                write_slice_XZ(w,slice_temp_file_XZ[W_SLICE],0);
                write_slice_XZ(*tracers[T],slice_temp_file_XZ[T_SLICE],0);
                write_slice_XZ(*diss,slice_temp_file_XZ[D_SLICE],0);
        
                write_slice_XY(u,slice_temp_file_XY[U_SLICE],xy_slice_height);
                write_slice_XY(v,slice_temp_file_XY[V_SLICE],xy_slice_height);
                write_slice_XY(w,slice_temp_file_XY[W_SLICE],xy_slice_height);
                write_slice_XY(*tracers[T],slice_temp_file_XY[T_SLICE],xy_slice_height);
                write_slice_XY(*diss,slice_temp_file_XY[D_SLICE],xy_slice_height);

                // Increment the slice index
                incr_slice_count();


                slice_next_plot += slice_interval; 
                if (master()) {
                    // in log file
                    fprintf(stdout,"*S*");
                }

            }

            /* H_AVG --  Write horizontal averages to disk if at correct time */
            if ((time - havg_next_plot) > -1e-6) {
                

                
                write_havg_variables(*diss,temp_file[D]);
                compute_xi(*diss, *tracers[T], gradient_op, grid_type, Nx, Ny, Nz, kappa);
                write_havg_variables(*diss,temp_file[Xi]);

                write_havg_ReStress(u,v,w,*tracers[T],temp_file);

                // Increment the havg index
                incr_chain_count();

                havg_next_plot += havg_interval; 
                 if (master()) {
                    // in log file
                    fprintf(stdout,"*H*");
                 }
            }



            /* Write to disk if at correct time */
            if ((time - next_plot) > -1e-6) {
                plot_number++;
                comp_duration = MPI_Wtime(); // time just before write (for dump)
                // Write the arrays
                write_array(u,"u",plot_number);
                write_array(w,"w",plot_number);
                if (Ny > 1)
                    write_array(v,"v",plot_number);
                // write the perturbation density
                *temp1 = (*tracers[T]);
                write_array(*temp1,"T",plot_number);
                if (write_pressure)
                    write_array(pressure,"p",plot_number);
                // update next plot time
                next_plot = next_plot + plot_interval;

                // Find average time to write (for dump)
                clock_time = MPI_Wtime(); // time just after write
                avg_write_time = (avg_write_time*(plot_number-restart_sequence-1) 
                        + (clock_time - comp_duration))/(plot_number-restart_sequence);
                // Print information about plot outputs
                write_plot_times(time, clock_time, comp_duration, avg_write_time, plot_number, restarting);




                // H_AVG -- Stitch the Horizontal Averages together
                for (int ifile = 0; ifile<num_h_saves; ifile++){
                    write_havg_stich(temp_file[ifile],final_file[ifile]);
                    // ### Need to close/re-open the temp after stitching
                    MPI_File_close(&(temp_file[ifile]));
                    MPI_File_open(MPI_COMM_WORLD, file_temp_names_ordered[ifile],
                        MPI_MODE_RDWR | MPI_MODE_CREATE | MPI_MODE_DELETE_ON_CLOSE,
                        MPI_INFO_NULL, &(temp_file[ifile]));
                }

                prev_chain_write_count += chain_write_count;
                chain_write_count = 0;

                

            }

            // see if close to end of compute time and dump
            check_and_dump(clock_time, real_start_time, compute_time, time, avg_write_time,
                    plot_number, iter, u, v, w, tracers);
            // Change dump log file if successfully reached final time
            successful_dump(plot_number, final_time, plot_interval);
            // increase counter
            iter++;
        }


        // User specified variables to dump
        void write_variables(DTArray & u,DTArray & v, DTArray & w,
                vector<DTArray *> & tracers) {
            write_array(u,"u.dump");
            write_array(v,"v.dump");
            write_array(w,"w.dump");
            *temp1 = (*tracers[T]);
            write_array(*temp1,"T.dump");
        }

        //////////////////////////////////////
        //////////  Horz. Avg. Func //////////
        //////////////////////////////////////
        
        // compute the horizontal average of a given variable. 
        void compute_havg(DTArray & q, double *qbar_vec){
           double * temp_qbar_vec = new double[Nz]();
            
            // Compute the horizontal average
            for (int Iz = q.lbound(thirdDim);  Iz <= q.ubound(thirdDim); Iz++) {
                for (int Ix = q.lbound(firstDim);  Ix <= q.ubound(firstDim); Ix++) {
                    for (int Iy = q.lbound(secondDim); Iy <= q.ubound(secondDim); Iy++) {
                        temp_qbar_vec[Iz] += q(Ix,Iy,Iz) / Nx / Ny;
                    }
                }
            }

            // Ensure it's the same on all processes
            MPI_Allreduce(temp_qbar_vec, qbar_vec, Nz, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
 
            delete[] temp_qbar_vec; 

        }

        // compute the horizontal rms of two variables. 
        void compute_hrms(DTArray & q1, DTArray & q2, double *q1bar, double *q2bar,double *qoutput){
           double * temp_qbar_vec = new double[Nz]();
            // Compute (u - <u>)*(v-<v>)


            // Compute the horizontal average
            for (int Iz = q1.lbound(thirdDim);  Iz <= q1.ubound(thirdDim); Iz++) {
                for (int Ix = q1.lbound(firstDim);  Ix <= q1.ubound(firstDim); Ix++) {
                    for (int Iy = q1.lbound(secondDim); Iy <= q1.ubound(secondDim); Iy++) {
                        temp_qbar_vec[Iz] += ((q1(Ix,Iy,Iz) - q1bar[Iz])*(q2(Ix,Iy,Iz) - q2bar[Iz])) / Nx / Ny;
                    }
                }
            }

            // Ensure it's the same on all processes
            MPI_Allreduce(temp_qbar_vec, qoutput, Nz, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            
            // qoutput = sqrt(qoutput);


            delete[] temp_qbar_vec; 

        }

        // Write out the horizontal average of the requested variable to the designated file  
        void write_havg_variables(DTArray & q,MPI_File qbar_temp_file){
            //## When doing a diagnostic output, write the mean
            double * qbar_vec = new double[Nz](); 
            compute_havg(q, qbar_vec);


            if (master()) {
                MPI_File_seek(qbar_temp_file, Nz*chain_write_count*sizeof(double), MPI_SEEK_SET);
                MPI_File_write(qbar_temp_file, qbar_vec, Nz, MPI_DOUBLE, &status);
            }

            delete[] qbar_vec; 
        }

        // Write out the double vector to the designated file  
        void write_chain_vec(double *qbar_vec,MPI_File qbar_temp_file){


            if (master()) {
                MPI_File_seek(qbar_temp_file, Nz*chain_write_count*sizeof(double), MPI_SEEK_SET);
                MPI_File_write(qbar_temp_file, qbar_vec, Nz, MPI_DOUBLE, &status);
            }

        }


        // Write out the horizontal average of the requested variable to the designated file  
        void write_havg_ReStress(DTArray & u,DTArray & v,DTArray & w,
                DTArray & rho,MPI_File *qbar_temp_files){

            // Note that q_bar_temp_files is ordered and should be 
            // reflected in file_names_ordered.


            int output_index = 0;

            //## When doing a diagnostic output, write the mean
            double * temp_vec = new double[Nz](); 
            double * ubar_vec = new double[Nz](); 
            double * vbar_vec = new double[Nz](); 
            double * wbar_vec = new double[Nz](); 
            double * rhobar_vec = new double[Nz](); 
            compute_havg(u, ubar_vec);
            compute_havg(v, vbar_vec);
            compute_havg(w, wbar_vec);
            compute_havg(rho, rhobar_vec);

            write_chain_vec(ubar_vec,qbar_temp_files[output_index]); output_index++;   // Write out the umean
            write_chain_vec(vbar_vec,qbar_temp_files[output_index]); output_index++;  // Write out the vmean
            write_chain_vec(wbar_vec,qbar_temp_files[output_index]); output_index++;   // Write out the wmean
            write_chain_vec(rhobar_vec,qbar_temp_files[output_index]); output_index++; // Write out the rho-mean


            //////////// Velocity Means////////////////
            // Write out urms
            compute_hrms(u,u,ubar_vec,ubar_vec,temp_vec);
            write_chain_vec(temp_vec,qbar_temp_files[output_index]);output_index++;

            // Write out vrms
            compute_hrms(v,v,vbar_vec,vbar_vec,temp_vec);
            write_chain_vec(temp_vec,qbar_temp_files[output_index]); output_index++;

            // Write out wrms
            compute_hrms(w,w,wbar_vec,wbar_vec,temp_vec);
            write_chain_vec(temp_vec,qbar_temp_files[output_index]); output_index++; 

            // Write out rhorms
            compute_hrms(rho,rho,rhobar_vec,rhobar_vec,temp_vec);
            write_chain_vec(temp_vec,qbar_temp_files[output_index]); output_index++; 



            //////// Re Stress Terms //////////////
            // Write out uvrms
            compute_hrms(u,v,ubar_vec,vbar_vec,temp_vec);
            write_chain_vec(temp_vec,qbar_temp_files[output_index]); output_index++; 

            // Write out uwrms
            compute_hrms(u,w,ubar_vec,wbar_vec,temp_vec);
            write_chain_vec(temp_vec,qbar_temp_files[output_index]); output_index++; 

            // Write out vwrms
            compute_hrms(v,w,vbar_vec,wbar_vec,temp_vec);
            write_chain_vec(temp_vec,qbar_temp_files[output_index]); output_index++; 


            //////////// Buoyancy Fluxes /////////////////
            // Write out urhorms
            compute_hrms(u,rho,ubar_vec,rhobar_vec,temp_vec);
            write_chain_vec(temp_vec,qbar_temp_files[output_index]); output_index++; 

            // Write out vrhorms
            compute_hrms(v,rho,vbar_vec,rhobar_vec,temp_vec);
            write_chain_vec(temp_vec,qbar_temp_files[output_index]); output_index++; 

            // Write out wrhorms
            compute_hrms(w,rho,wbar_vec,rhobar_vec,temp_vec);
            write_chain_vec(temp_vec,qbar_temp_files[output_index]); output_index++; 


            delete[] temp_vec ; 
            delete[] ubar_vec ; 
            delete[] vbar_vec ; 
            delete[] wbar_vec ; 
            delete[] rhobar_vec ; 
        }


        // Stich the temp horizontally averages together. 
        void write_havg_stich(MPI_File qbar_temp_file,MPI_File qbar_final_file){
            // ## When doing a major write, need to 'stitch' together the temp into the main
            double* temp_bar = new double[Nz](); 
            int my_rank = -1; int num_procs = -1;
            MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
            for (int JJ = my_rank; JJ < chain_write_count; JJ += num_procs) {
                offset_from = JJ*Nz;
                offset_to   = (JJ + prev_chain_write_count)*Nz;
                MPI_File_seek( qbar_temp_file,  offset_from*sizeof(double), MPI_SEEK_SET);
                MPI_File_read( qbar_temp_file,  temp_bar, Nz, MPI_DOUBLE, &status);
                MPI_File_seek( qbar_final_file, offset_to*sizeof(double), MPI_SEEK_SET);
                MPI_File_write(qbar_final_file, temp_bar, Nz, MPI_DOUBLE, &status);
            }

            delete[] temp_bar;
        }

        // Increase chain count -- how many writes have been written since last Stitch
        void incr_chain_count(){
            chain_write_count += 1;
        }

        //////////////////////////////////////
        //////////////////////////////////////
        //////////////////////////////////////

        //////////////////////////////////////
        //////  Streamwise Slice. Func ///////
        //////////////////////////////////////
        

        // Write out the horizontal average of the requested variable to the designated file  
        // Here take the XZ slice --- Parallelized
        // 
        void write_slice_XZ(DTArray & q,MPI_File q_file, int Iy = 0){
            // q - data field to output
            // q_file - MPI_FILE for output
            // Iy - y-plane to output in grid-coordinates (Ny)

            int offset, my_rank, num_procs;
            MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

            // Constuct output array
            offset = Nx*Nz*get_slice_count() + Nx*Nz/num_procs*my_rank;
            double *xz_data_buffer = new double[Nz*Nx/num_procs];
            
            int cnt = 0;
            for (int Ix = q.lbound(firstDim); Ix <= q.ubound(firstDim); Ix++) {
                for (int Iz = 0; Iz < Nz; Iz++) {
                    xz_data_buffer[cnt] = q(Ix,Iy,Iz);
                    cnt++;
                }
            }

            MPI_File_seek(q_file, offset*sizeof(double), MPI_SEEK_SET);
            MPI_File_write(q_file, xz_data_buffer, Nz*Nx/num_procs, MPI_DOUBLE, &status);
            
            delete[] xz_data_buffer;



        }


        // Write out the horizontal average of the requested variable to the designated file  
        // Here take the XY slice --- Parallelized
        // 
        void write_slice_XY(DTArray & q,MPI_File q_file, int Iz = 0){
            // q - data field to output
            // q_file - MPI_FILE for output
            // Iz - z-plane to output in grid-coordinates (Nz)

            int offset, my_rank, num_procs;
            MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

            // Construct output array 
            offset = Nx*Ny*get_slice_count() + Nx*Ny/num_procs*my_rank;
            double *xy_data_buffer = new double[Nx*Ny/num_procs];
            
            int cnt = 0;
            for (int Ix = q.lbound(firstDim); Ix <= q.ubound(firstDim); Ix++) {
                for (int Iy = 0; Iy < Ny; Iy++) {
                    xy_data_buffer[cnt] = q(Ix,Iy,Iz);
                    cnt++;
                }
            }

            MPI_File_seek(q_file, offset*sizeof(double), MPI_SEEK_SET);
            MPI_File_write(q_file, xy_data_buffer, Ny*Nx/num_procs, MPI_DOUBLE, &status);
            
            delete[] xy_data_buffer;

        }


        // // Write out the horizontal average of the requested variable to the designated file  
        // // Here take the YZ slice --- Not-Parallelized
        // // NOTE WORKING!!! Code originally written for XZ and was not properly converted
        // void write_slice_YZ(DTArray & q,MPI_File q_file){

        //     // ## Grab an array for slice data 
        //     double * q_slice = new double[Nz*Nx](); 
        //     double * q_slice_final = new double[Nz*Nx](); 

        //     // Cut out the ymin plane
        //     for (int Iz = q.lbound(thirdDim);  Iz <= q.ubound(thirdDim); Iz++) {
        //         for (int Ix = q.lbound(firstDim);  Ix <= q.ubound(firstDim); Ix++) {
        //                 q_slice[Iz*Nx + Ix] = q(Ix,0,Iz);
        //         }
        //     }

        //     //Pass to all Processes
        //     MPI_Allreduce(q_slice, q_slice_final, Nz*Nx, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        //     // Write out 
        //     if (master()) {
        //         //          printf("%f", q_slice_final[q.lbound(thirdDim)*Nx + q.lbound(firstDim)]);
        //         //          printf("Double Count: %d",sizeof(double));
        //         MPI_File_seek(q_file, Nx*Nz*get_slice_count()*sizeof(double), MPI_SEEK_SET);
        //         MPI_File_write(q_file, q_slice_final, Nz*Nx, MPI_DOUBLE, &status);
        //     }

        //     // Close the File 
        //     delete[] q_slice;
        //     delete[] q_slice_final;

        // }

        // Save the various slices 
        void verify_slice_files(){

            if (get_slice_count() % SLICES_PER_BIN == 0){

                    // Set the slice count to 0
                    reset_slice_count();

                    for (int ss = 0 ; ss<num_XZslices ; ss++){
                        
                        //if the Files are already Open, close them
                        if (slice_bin_count != 0){                
                            MPI_File_close(&(slice_temp_file_XZ[ss]));
                        }
                        
                        // Get the Name of the  XZ slice 
                        string filenameXZ;
                        std::stringstream convXZ; /* For converting sequence number to string */
                        convXZ << slice_names_XZ[ss];
                        convXZ << "_XZslices";
                        convXZ << SLICES_PER_BIN;
                        convXZ << ".";
                        convXZ << slice_bin_count;
                        filenameXZ = convXZ.str();
                        
                        // Open the File 
                        MPI_File_open(MPI_COMM_WORLD, filenameXZ.c_str(),
                            MPI_MODE_RDWR | MPI_MODE_CREATE ,
                            MPI_INFO_NULL, &(slice_temp_file_XZ[ss]));
                    }


                    for (int ss = 0 ; ss<num_XYslices ; ss++){
                        
                        //if the Files are already Open, close them
                        if (slice_bin_count != 0){                
                            MPI_File_close(&(slice_temp_file_XY[ss]));
                        }
                        
                        // Get the Name of the  XY slice 
                        string filenameXY;
                        std::stringstream convXY; /* For converting sequence number to string */
                        convXY << slice_names_XY[ss];
                        convXY << "_XYslices";
                        convXY << SLICES_PER_BIN;
                        convXY << ".";
                        convXY << slice_bin_count;
                        filenameXY = convXY.str();
                        
                        // Open the File 
                        MPI_File_open(MPI_COMM_WORLD, filenameXY.c_str(),
                            MPI_MODE_RDWR | MPI_MODE_CREATE ,
                            MPI_INFO_NULL, &(slice_temp_file_XY[ss]));
                    }

                    // Increment the bin counter.
                    slice_bin_count++; 
            }
        }

        void incr_slice_count(){
            slice_write_count += 1;
        }

        int get_slice_count(){
            return slice_write_count;
        }

        void reset_slice_count(){
            slice_write_count = 0;
        }

        //////////////////////////////////////
        //////////////////////////////////////
        //////////////////////////////////////

        // Constructor: Initialize local variables
        userControl():
            xx(split_range(Nx)), yy(Ny), zz(Nz),
            gradient_op(0),
            plot_number(restart_sequence),
            next_plot(initial_time + plot_interval),
            havg_next_plot(initial_time ),
            slice_next_plot(initial_time)
    {   compute_quadweights(
            size_x(),   size_y(),   size_z(),
            length_x(), length_y(), length_z(),
            type_x(),   type_y(),   type_z());
        // Create one-dimensional arrays for the coordinates
        automatic_grid(MinX, MinY, MinZ, &xx, &yy, &zz);
    }
};

/* The ''main'' routine */
int main(int argc, char ** argv) {
    /* Initialize MPI.  This is required even for single-processor runs,
       since the inner routines assume some degree of parallelization,
       even if it is trivial. */
    MPI_Init(&argc, &argv);

    real_start_time = MPI_Wtime();     // start of simulation (for dump)
    /* ------------------ Define parameters from spins.conf --------------------- */
    options_init();

    option_category("Grid Options");
    add_option("Lx",&Lx,"Length of tank");
    add_option("Ly",&Ly,1.0,"Width of tank");
    add_option("Lz",&Lz,"Height of tank");
    add_option("Nx",&Nx,"Number of points in X");
    add_option("Ny",&Ny,1,"Number of points in Y");
    add_option("Nz",&Nz,"Number of points in Z");
    add_option("min_x",&MinX,0.0,"Minimum X-value");
    add_option("min_y",&MinY,0.0,"Minimum Y-value");
    add_option("min_z",&MinZ,0.0,"Minimum Z-value");

    option_category("Grid expansion options");
    string xgrid_type, ygrid_type, zgrid_type;
    add_option("type_x",&xgrid_type,
            "Grid type in X.  Valid values are:\n"
            "   FOURIER: Periodic\n"
            "   FREE_SLIP: Cosine expansion\n"
            "   NO_SLIP: Chebyhsev expansion");
    add_option("type_y",&ygrid_type,"FOURIER","Grid type in Y");
    add_option("type_z",&zgrid_type,"Grid type in Z");

    // option_category("Input data");
    // string datatype;
    // add_option("file_type",&datatype,
    //         "Format of input data files, including that for the mapped grid."
    //         "Valid options are:\n"
    //         "   MATLAB: \tRow-major 2D arrays of size Nx x Nz\n"
    //         "   CTYPE:  \tColumn-major 2D arrays (including that output by 2D SPINS runs)\n"
    //         "   FULL:   \tColumn-major 3D arrays; implies CTYPE for grid mapping if enabled");

    // add_option("u_file",&u_filename,"U-velocity filename");
    // add_option("v_file",&v_filename,"","V-velocity filename");
    // add_option("w_file",&w_filename,"W-velocity filename");
    // add_option("rho_file",&rho_filename,"Rho (density) filename");
    // add_option("tracer_file",&tracer_filename,"","Tracer filename");

    option_category("Physical parameters");
    add_option("g",&g,9.81,"Gravitational acceleration");
    add_option("rho_0",&rho_0,1000.0,"Reference density");
    add_option("visco",&visco,"Viscosity");
    add_option("kappa",&kappa,"Diffusivity of density");

    option_category("Problem parameters");
    add_option("T_offset",&T_offset,"Lower Layer Temp");
    add_option("sig_T0",&sig_T0,"Initial T width");
    add_option("Tz_bot",&Tz_bot,0.0,"Bottom T Gradient Value");
    add_option("T_top",&T_top,"Top T value");
    

    
    option_category("Temporal options");
    add_option("final_time",&final_time,"Final time");
    add_option("plot_interval",&plot_interval,"Time between writes");
    add_option("havg_interval",&havg_interval,"Time between havg writes");
    add_option("slice_interval",&slice_interval,"Time between havg writes");
    add_option("dt_max",&dt_max,0.0,"Maximum time step. Zero value results in the default");
    add_option("xy_slice_height",&xy_slice_height,int(double(Nz)/2.0 - 1),"vertical position of the XY slice in grid index");


    option_category("Restart options");
    add_option("restart",&restarting,false,"Restart from prior output time.");
    add_option("restart_time",&initial_time,0.0,"Time to restart from");
    add_option("restart_sequence",&restart_sequence,-1,"Sequence number to restart from");


    option_category("Dumping options");
    add_option("restart_from_dump",&restart_from_dump,false,"If restart from dump");
    add_option("compute_time",&compute_time,-1.0,"Time permitted for computation");

    option_category("Other options");
    add_option("perturb",&perturb,"Initial perturbation in velocity");
    add_option("compute_enstrophy",&compute_enstrophy,true,"Calculate enstrophy?");
    add_option("compute_dissipation",&compute_dissipation,true,"Calculate dissipation?");
    add_option("compute_BPE",&compute_BPE,true,"Calculate BPE?");
    add_option("compute_internal_to_BPE",&compute_internal_to_BPE,true,
            "Calculate BPE gained from internal energy?");
    add_option("compute_stresses_top",&compute_stresses_top,false,"Calculate top surfaces stresses?");
    add_option("compute_stresses_bottom",&compute_stresses_bottom,false,"Calculate bottom surfaces stresses?");
    add_option("write_pressure",&write_pressure,false,"Write the pressure field?");

    option_category("Filter options");
    add_option("f_cutoff",&f_cutoff,0.6,"Filter cut-off frequency");
    add_option("f_order",&f_order,2.0,"Filter order");
    add_option("f_strength",&f_strength,20.0,"Filter strength");

    // Parse the options from the command line and config file
    options_parse(argc,argv);


    /* ------------------ Create Horizontal Average Output File --------------------- */
    // ## Open the files for writing
    for (int ifile = 0; ifile<num_h_saves; ifile++){
        MPI_File_open(MPI_COMM_WORLD, file_temp_names_ordered[ifile],
        MPI_MODE_RDWR | MPI_MODE_CREATE | MPI_MODE_DELETE_ON_CLOSE,
        MPI_INFO_NULL, &(temp_file[ifile]));

        MPI_File_open(MPI_COMM_WORLD, file_names_ordered[ifile], MPI_MODE_WRONLY | MPI_MODE_CREATE,
        MPI_INFO_NULL, &(final_file[ifile]));

    }


    /* ------------------ Adjust and check parameters --------------------- */
    /* Now, make sense of the options received.  Many of these
     * can be directly used, but the ones of string-type need further procesing. */

    // adjust temporal values when restarting from dump
    if (restart_from_dump) {
        adjust_for_dump(restarting, initial_time, restart_sequence,
                final_time, compute_time, avg_write_time, Num_tracers, Nx, Ny, Nz);
    }

    // check restart sequence
    check_restart_sequence(restarting, restart_sequence, initial_time, plot_interval);

    // parse expansion types
    parse_boundary_conditions(xgrid_type, ygrid_type, zgrid_type, intype_x, intype_y, intype_z);
    // vector of expansion types
    grid_type[0] = xgrid_type;
    grid_type[1] = ygrid_type;
    grid_type[2] = zgrid_type;

    // adjust Ly for 2D
    if (Ny==1 and Ly!=1.0) {
        Ly = 1.0;
        if (master())
            fprintf(stdout,"Simulation is 2 dimensional, "
                    "Ly has been changed to 1.0 for normalization.\n");
    }

    /* ------------------ Derived parameters --------------------- */

    // Dynamic viscosity
    mu = visco*rho_0;
    // Maximum buoyancy frequency (squared) if the initial stratification was stable
    // Maximum time step
    if (dt_max == 0.0) {
        // if dt_max not given in spins.conf, use the buoyancy frequency
        dt_max = 0.5/sqrt(N2_max);
    }

    /* ------------------ Print some parameters --------------------- */

    if (master()) {
        fprintf(stdout,"Kelvin-Helmholtz billow problem\n");
        fprintf(stdout,"Using a %f x %f x %f grid of %d x %d x %d points\n",Lx,Ly,Lz,Nx,Ny,Nz);
        fprintf(stdout,"g = %f, rho_0 = %f\n",g,rho_0);
        fprintf(stdout,"Time between plots: %g s\n",plot_interval);
        fprintf(stdout,"Initial velocity perturbation: %g\n",perturb);
        fprintf(stdout,"Filter cutoff = %f, order = %f, strength = %f\n",f_cutoff,f_order,f_strength);
        fprintf(stdout,"Max time step: %g\n",dt_max);
    }

    /* ------------------ Do stuff --------------------- */

    // Create an instance of the above class
    userControl mycode;
    // Create a flow-evolver that takes its settings from the above class
    FluidEvolve<userControl> do_stuff(&mycode);
    // Initialize
    do_stuff.initialize();
    compute_start_time = MPI_Wtime(); // beginning of simulation (after reading in data)
    double startup_time = compute_start_time - real_start_time;
    if (master()) fprintf(stdout,"Start-up time: %.6g s.\n",startup_time);
    // Run until the end of time
    do_stuff.do_run(final_time);



    // ## Close upon completion
    for (int ifile = 0; ifile<num_h_saves; ifile++){
        MPI_File_close(&(temp_file[ifile]));
        MPI_File_close(&(final_file[ifile]));
    }

    for (int ss = 0 ; ss<num_XZslices ; ss++){
        MPI_File_close(&(slice_temp_file_XZ[ss]));
    }

    for (int ss = 0 ; ss<num_XYslices ; ss++){
        MPI_File_close(&(slice_temp_file_XY[ss]));
    }




    MPI_Finalize();

    return 0;

}
