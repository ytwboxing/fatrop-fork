#include "fatrop/spectool/spectool.hpp"
#include "fatrop/fatrop.hpp"
#include <casadi/casadi.hpp>
#include "Timer.hpp"

int main()
{
    const int N = 100;
    const int n = 4;
    const int m = 1;
    const double dt = 0.05;
    const double T = N * dt;
    const double m_c = 1.0;
    const double m_p = 1.0;
    const double l_p = 1.0;
    const double gravity_ = 9.81;
    
    // Timer fatrop_timer;
    
    fatrop::spectool::Ocp ocp = fatrop::spectool::Ocp();
    casadi::MX x = ocp.state(n);
    casadi::MX u = ocp.control();
    // casadi::MX p = ocp.parameter();

    casadi::MX xnext = casadi::MX::zeros(n);

    casadi::MX sin_theta = casadi::MX::sin(x(2));
    casadi::MX cos_theta = casadi::MX::cos(x(2));
    casadi::MX force = u;
    casadi::MX theta_dot = x(3);
    
    xnext(0) = x(0) + dt * x(1);
    xnext(1) = x(1) + dt * (1.0 / (m_c + m_p * sin_theta * sin_theta) * 
                            (force + m_p * sin_theta * (l_p * theta_dot * theta_dot + 
                            gravity_ * cos_theta)));
    xnext(2) = x(2) + dt * x(3);
    xnext(3) = x(3) + dt * (1.0 / (l_p * (m_c + m_p * sin_theta * sin_theta)) *
                            (-force * cos_theta - m_p * l_p * theta_dot * theta_dot * 
                            cos_theta * sin_theta - (m_c + m_p) * gravity_ * sin_theta));

    fatrop::spectool::Stage stage = ocp.new_stage(N);

    stage.add_objective(u * u 
                        + 100. * (x(0) - 1.) * (x(0) - 1.)
                        + 10. * x(1) * x(1)
                        + 100. * (x(2) - 2.) * (x(2) - 2.)
                        + 10. * x(3) * x(3), 
                        fatrop::spectool::at::t0, 
                        fatrop::spectool::at::mid
    );
    stage.add_objective(1000. * 100. * (x(0) - 1.) * (x(0) - 1.)
                        + 1000. * 10. * x(1) * x(1)
                        + 1000. * 100. * (x(2) - 2.) * (x(2) - 2.)
                        + 1000. * 10. * x(3) * x(3),
                        fatrop::spectool::at::tf);

    stage.set_next(x, xnext);

    stage.subject_to(-1.5 <= (x(0) <= 1.5), 
                     fatrop::spectool::at::t0, 
                     fatrop::spectool::at::mid
    );
    stage.subject_to(-10. <= (u <= 5.), 
                     fatrop::spectool::at::t0, 
                     fatrop::spectool::at::mid
    );
    stage.subject_to(-1.5 <= (x(0) <= 1.5), 
                     fatrop::spectool::at::tf
    );

    ocp.solver("fatrop", 
               {
                    {"jit", false}
               },
               {
                    {"max_iter", 100},
                    {"tol", 1e-2},
                    {"constr_viol_tol", 1e-2},
                    {"acceptable_tol", 1e-2},
                    {"mu_init", 1e2}
               }
    );

    Timer fatrop_timer;

    stage.at_t0().subject_to(x(0) == 0.);
    stage.at_t0().subject_to(x(1) == 0.);
    stage.at_t0().subject_to(x(2) == 0.);
    stage.at_t0().subject_to(x(3) == 0.);

    casadi::Function ocp_func = ocp.to_function("cartpole_test", 
        {}, 
        {ocp.sample(x).second, ocp.sample(u).second}
    );
    auto ret = ocp_func(std::vector<casadi::DM>{});

    std::cout << "fatrop solve time(ms): " << fatrop_timer.getMs() << std::endl;

    std::cout << ret << std::endl; 

  return 0;
}