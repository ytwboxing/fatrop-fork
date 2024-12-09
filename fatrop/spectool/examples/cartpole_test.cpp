#include "fatrop/spectool/spectool.hpp"
#include "fatrop/fatrop.hpp"
#include <casadi/casadi.hpp>
#include "Timer.hpp"
#include <matplotlib-cpp/matplotlibcpp.h>
namespace plt = matplotlibcpp;

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

     // column vector
     casadi::DM Q = casadi::DM({{100.}, {10.}, {100.}, {10.}});
     casadi::DM Q_f = 1000. * Q;
     casadi::DM R = casadi::DM({{1.}});
     casadi::DM x_ref = casadi::DM({{1.}, {0.}, {2.}, {0.}});

     Timer construct_ocp_timer;

     fatrop::spectool::Ocp ocp = fatrop::spectool::Ocp();
     casadi::MX x = ocp.state(n, 1);
     casadi::MX u = ocp.control(m, 1);
     casadi::MX p = ocp.parameter(n, 1);

     casadi::MX xnext = casadi::MX::zeros(n, 1);

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

     stage.add_objective(R(0) * u * u 
                         + Q(0) * (x(0) - x_ref(0)) * (x(0) - x_ref(0))
                         + Q(1) * (x(1) - x_ref(1)) * (x(1) - x_ref(1))
                         + Q(2) * (x(2) - x_ref(2)) * (x(2) - x_ref(2))
                         + Q(3) * (x(3) - x_ref(3)) * (x(3) - x_ref(3)), 
                         fatrop::spectool::at::t0, 
                         fatrop::spectool::at::mid
     );

     stage.add_objective(Q_f(0) * (x(0) - x_ref(0)) * (x(0) - x_ref(0))
                         + Q_f(1) * (x(1) - x_ref(1)) * (x(1) - x_ref(1))
                         + Q_f(2) * (x(2) - x_ref(2)) * (x(2) - x_ref(2))
                         + Q_f(3) * (x(3) - x_ref(3)) * (x(3) - x_ref(3)),
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

     stage.at_t0().subject_to(x == p);

     ocp.solver("fatrop", 
               {
                    {"expand", true},
                    {"jit", false}
               },
               {
                    {"max_iter", 200},
                    {"tol", 1e-2},
                    {"constr_viol_tol", 1e-2},
                    {"acceptable_tol", 1e-2},
                    {"mu_init", 1e2}
               }
     );
     std::cout << "construct ocp time(ms): " << construct_ocp_timer.getMs() << std::endl;

     Timer construct_casadi_func_timer;
     casadi::Function ocp_func = ocp.to_function("cartpole_test", 
          {p}, 
          {ocp.sample(x).second, ocp.sample(u).second}
     );
     std::cout << "construct casadi func time(ms): " << construct_casadi_func_timer.getMs() << std::endl;

     const int bench_num = 100;
     int bench_iter = 0;
     std::vector<double> solve_time_vec;
     while (bench_iter < bench_num) {
          Timer fatrop_solve_timer;

          casadi::DMVector ret = ocp_func(casadi::DM({{0.0}, {0.0}, {0.0}, {0.0}}));
          casadi::DM states = ret[0];
          casadi::DM inputs = ret[1];
          // std::cout << states << std::endl;
          // std::cout << std::endl;
          // std::cout << inputs << std::endl;
          std::vector<double> x_fatrop, vel_fatrop, theta_fatrop, theta_dot_fatrop, u_fatrop;
          for (int i = 0; i < N + 1; ++i) {
               x_fatrop.emplace_back(states(0, i));
               vel_fatrop.emplace_back(states(1, i));
               theta_fatrop.emplace_back(states(2, i));
               theta_dot_fatrop.emplace_back(states(3, i));
               if (i < N) {
                    u_fatrop.emplace_back(inputs(0, i));
               }
          }

          solve_time_vec.emplace_back(fatrop_solve_timer.getMs());
          std::cout << "fatrop solve time(ms): " << fatrop_solve_timer.getMs() << std::endl;

          bench_iter++;
          if (bench_iter == bench_num) {
               std::cout << "\n\n";
               std::cout << "Total run " << bench_num << " benchmarks" << std::endl;
               std::cout << "fatrop" 
                         << "\n    max time(ms): " << *std::max_element(solve_time_vec.begin(), solve_time_vec.end())
                         << "\n    min time(ms): " << *std::min_element(solve_time_vec.begin(), solve_time_vec.end())
                         << "\n    avg time(ms): " << (std::accumulate(solve_time_vec.begin(), solve_time_vec.end(), 0.0) / solve_time_vec.size());
               std::cout << std::endl;

               plt::subplot(5, 1, 1);
               plt::plot(x_fatrop, {{"label", "fatrop"}, {"linestyle", "-"}});
               plt::title("x");
               plt::legend();
               plt::subplot(5, 1, 2);
               plt::plot(vel_fatrop, {{"label", "fatrop"}, {"linestyle", "-"}});
               plt::title("vel");
               plt::legend();
               plt::subplot(5, 1, 3);
               plt::plot(theta_fatrop, {{"label", "fatrop"}, {"linestyle", "-"}});
               plt::title("theta");
               plt::legend();
               plt::subplot(5, 1, 4);
               plt::plot(theta_dot_fatrop, {{"label", "fatrop"}, {"linestyle", "-"}});
               plt::title("theta-dot");
               plt::legend();
               plt::subplot(5, 1, 5);
               plt::plot(u_fatrop, {{"label", "fatrop"}, {"linestyle", "-"}});
               plt::title("u");
               plt::legend();

               plt::show();
          }
     }

     return 0;
}