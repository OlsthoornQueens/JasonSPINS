#include "../Science.hpp"
#include "math.h"
#include "../Par_util.hpp"
#include "stdio.h"
#include "../Split_reader.hpp"
#include "../T_util.hpp"
#include "../Parformer.hpp"
#include "../Sorter.hpp"
#include <numeric>

using blitz::Array;
using blitz::cos;
using namespace TArrayn;
using namespace NSIntegrator;
using namespace Transformer;


// Viscous dissipation: 2*mu*e_ij*e_ij
void compute_xi(TArrayn::DTArray & xi, TArrayn::DTArray & rho, TArrayn::Grad * gradient_op, const string * grid_type,
        const int Nx, const int Ny, const int Nz, const double diffusivity) {
    // Set-up
    static DTArray *temp = alloc_array(Nx,Ny,Nz);
    S_EXP expan[3];
    assert(gradient_op);

    // 1st term: xi_11^2 = (d rho/dx)^2
    find_expansion(grid_type, expan, "rho");
    gradient_op->setup_array(&rho,expan[0],expan[1],expan[2]);
    gradient_op->get_dx(temp,false);
    xi = pow(*temp,2);
    // 2nd term: xi_22^2 = (d rho /dy)^2
    gradient_op->get_dy(temp,false);
    xi += pow(*temp,2);
    // 3rd term: xi_33^2 = (d rho/dz)^2
    gradient_op->get_dz(temp,false);
    xi += pow(*temp,2);
    
    // multiply by 2*kappa
    xi *= 2.0*diffusivity;
}
