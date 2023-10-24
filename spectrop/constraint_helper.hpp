
#pragma once
#include <casadi/casadi.hpp>
#include <string>
#include <cassert>
#include "casadi_utilities.hpp"
namespace fatrop
{
    namespace spectrop
    {
        namespace cs = casadi;
        class ConstraintHelper
        {
        public:
            static void process(const casadi::MX &constr,
                                casadi::DM &lb,
                                casadi::DM &ub,
                                casadi::MX &g_ineq,
                                casadi::MX &g)
            {
                // reset lb, ub, g_ineq, g
                lb = casadi::DM();
                ub = casadi::DM();
                g_ineq = casadi::MX();
                g = casadi::MX();
                assert(constr.size2() == 1);
                auto opti = casadi::Opti();
                auto syms = casadi::MX::symvar(constr);
                std::vector<casadi::MX> syms_opti;
                for (auto &sym : syms)
                {
                    syms_opti.push_back(opti.variable(sym.size1(), sym.size2()));
                }
                auto canon = opti.advanced().canon_expr(casadi::MX::substitute({constr}, syms, syms_opti)[0]);
                casadi::DM lb_temp = casadi::MX::evalf(canon.lb);
                casadi::DM ub_temp = casadi::MX::evalf(canon.ub);
                auto g_temp = casadi::MX::substitute({canon.canon}, syms_opti, syms)[0];
                for (int i = 0; i < lb_temp.size1(); i++)
                {
                    if ((double)lb_temp(i) == (double)ub_temp(i))
                    {
                        g = casadi::MX::veccat({g, g_temp(i) - lb_temp(i)});
                    }
                    else
                    {
                        g_ineq = casadi::MX::veccat({g_ineq, g_temp(i)});
                        lb = casadi::DM::veccat({lb, lb_temp(i)});
                        ub = casadi::DM::veccat({ub, ub_temp(i)});
                    }
                }
            }
            static void process(const std::vector<casadi::MX> &constr,
                                casadi::DM &lb,
                                casadi::DM &ub,
                                casadi::MX &g_ineq,
                                casadi::MX &g)
            {
                lb = casadi::DM();
                ub = casadi::DM();
                g_ineq = casadi::MX();
                g = casadi::MX();
                for (auto &constr_i : constr)
                {
                    auto lb_i = casadi::DM();
                    auto ub_i = casadi::DM();
                    auto g_ineq_i = casadi::MX();
                    auto g_i = casadi::MX();
                    process(constr_i, lb_i, ub_i, g_ineq_i, g_i);
                    lb = casadi::DM::veccat({lb, lb_i});
                    ub = casadi::DM::veccat({ub, ub_i});
                    g_ineq = casadi::MX::veccat({g_ineq, g_ineq_i});
                    g = casadi::MX::veccat({g, g_i});
                }
            }
        };
    }
}