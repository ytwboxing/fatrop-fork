#ifndef FATROPPARAMSINCLUDED
#define FATROPPARAMSINCLUDED
#include "aux/SmartPtr.hpp"
namespace fatrop
{
    class FatropParams
    {
    public:
        int max_watchdog_steps = 20;
        // bool linear_solver_improved_accuracy = true;
        // bool first_try_watchdog = true;
        int maxiter = 1000;
        double smax = 100.0;
        double lammax = 1e3;
        double tol = 1e-8;
        double mu0 = 1e2;
        // double mu0 = 2e5;
        // double mu0 = 1e;
        double kappa_eta = 10;
        double kappa_mu = 0.2;
        double theta_mu = 1.5;
        double delta_w0 = 1e-4;
        double delta_wmin = 1e-20;
        double kappa_wmin = 1.0 / 3.0;
        double kappa_wplus = 8;
        double kappa_wplusem = 100;
        double kappa_sigma = 1e10;
        double s_phi = 2.3;
        double delta = 1.0;
        double s_theta = 1.1;
        double theta_min = 1e-4;
        // double gamma_theta = 1e-8;
        double gamma_theta = 1e-12;
        // double gamma_theta = 1e-5; // todo check!!
        double gamma_phi = 1e-8;
        double gamma_alpha = 0.05;
        double eta_phi = 1e-4;
        double delta_c_stripe = 1e-8;
        double kappa_c = 0.25;
        double kappa1 = 1e-2;
        double kappa2 = 1e-2;
        double kappa_d = 1e-5;
        double bound_relax_factor = 1e-8;
        double constr_viol_tol = 1e-4; // currently only used to relax bounds
    };

} // namespace fatrop
#endif // FatropParams