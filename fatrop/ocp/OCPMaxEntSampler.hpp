#ifndef OCPMAXENTSAMPLERINCLUDED
#define OCPMAXENTSAMPLERINCLUDED
#include "OCPLSRiccati.hpp"
#include <random>
namespace fatrop
{
    class OCPMaxEntSampler : public OCPLSRiccati
    {
        using OCPLSRiccati::OCPLSRiccati;

    public:
        std::default_random_engine generator;
        std::normal_distribution<double> distribution = std::normal_distribution<double>(0.0, 1.0);
        int Sample(
            OCPKKTMemory *OCP,
            const FatropVecBF &ux,
            const FatropVecBF &lam,
            const FatropVecBF &delta_s,
            const FatropVecBF &sigma_total,
            const FatropVecBF &rhs_rq,
            const FatropVecBF &rhs_b,
            const FatropVecBF &rhs_g,
            const FatropVecBF &rhs_g_ineq,
            const FatropVecBF &rhs_gradb,
            double alpha)
        {
            double inertia_correction_w = lastused_.inertia_correction_w;
            double inertia_correction_c = lastused_.inertia_correction_c;
            assert(inertia_correction_c == 0.0); // not implemented yet
            bool increased_accuracy = true;
            //     // blasfeo_timer timer;
            //     // blasfeo_tic(&timer);
            //     // define compiler macros for notational convenience
            // #define OCPMACRO(type, name, suffix) type name##suffix = ((type)OCP->name)
            // #define AUXMACRO(type, name, suffix) type name##suffix = ((type)OCP->aux.name)
            // #define SOLVERMACRO(type, name, suffix) type name##suffix = ((type)name)
            int K = OCP->K;
            //     // make variables local for efficiency
            // OCPMACRO(MAT *, RSQrqt, _p);
            OCPMACRO(MAT *, BAbt, _p);
            // OCPMACRO(MAT *, Ggt, _p);
            OCPMACRO(MAT *, Ggt_ineq, _p);
            SOLVERMACRO(MAT *, Ppt, _p);
            SOLVERMACRO(MAT *, Hh, _p);
            SOLVERMACRO(MAT *, AL, _p);
            SOLVERMACRO(MAT *, RSQrqt_tilde, _p);
            SOLVERMACRO(MAT *, Ggt_stripe, _p);
            SOLVERMACRO(MAT *, Ggt_tilde, _p);
            SOLVERMACRO(PMAT *, Pl, _p);
            SOLVERMACRO(PMAT *, Pr, _p);
            // SOLVERMACRO(MAT *, GgLt, _p);
            // SOLVERMACRO(MAT *, RSQrqt_hat, _p);
            SOLVERMACRO(MAT *, Llt, _p);
            SOLVERMACRO(MAT *, Llt_shift, _p);
            SOLVERMACRO(MAT *, GgIt_tilde, _p);
            // SOLVERMACRO(MAT *, GgLIt, _p);
            SOLVERMACRO(MAT *, HhIt, _p);
            // SOLVERMACRO(MAT *, PpIt_hat, _p);
            SOLVERMACRO(MAT *, LlIt, _p);
            // SOLVERMACRO(MAT *, Ggt_ineq_temp, _p);
            SOLVERMACRO(PMAT *, PlI, _p);
            SOLVERMACRO(PMAT *, PrI, _p);
            SOLVERMACRO(VEC *, ux, _p);
            SOLVERMACRO(VEC *, lam, _p);
            SOLVERMACRO(VEC *, delta_s, _p);
            SOLVERMACRO(VEC *, sigma_total, _p);
            SOLVERMACRO(VEC *, rhs_rq, _p);
            SOLVERMACRO(VEC *, rhs_b, _p);
            SOLVERMACRO(VEC *, rhs_g, _p);
            SOLVERMACRO(VEC *, rhs_g_ineq, _p);
            SOLVERMACRO(VEC *, rhs_gradb, _p);
            OCPMACRO(int *, nu, _p);
            OCPMACRO(int *, nx, _p);
            OCPMACRO(int *, ng, _p);
            OCPMACRO(int *, ng_ineq, _p);
            SOLVERMACRO(int *, gamma, _p);
            SOLVERMACRO(int *, rho, _p);
            VEC *v_RSQrq_hat_curr_p;
            // MAT *RSQrq_hat_curr_p;
            int rank_k;
            int *offs_ineq_p = (int *)OCP->aux.ineq_offs.data();
            int *offs_g_ineq_p = (int *)OCP->aux.g_ineq_offs.data();
            int *offs_ux = (int *)OCP->aux.ux_offs.data();
            int *offs_g = (int *)OCP->aux.g_offs.data();
            int *offs_dyn_eq = (int *)OCP->aux.dyn_eq_offs.data();
            int *offs_dyn = (int *)OCP->aux.dyn_offs.data();

            // todo make this member variables

            VEC *v_Ppt_p = (VEC *)v_Ppt[0];
            VEC *v_Hh_p = (VEC *)v_Hh[0];
            VEC *v_AL_p = (VEC *)v_AL[0];
            VEC *v_RSQrqt_tilde_p = (VEC *)v_RSQrqt_tilde[0];
            VEC *v_Ggt_stripe_p = (VEC *)v_Ggt_stripe[0];
            VEC *v_Ggt_tilde_p = (VEC *)v_Ggt_tilde[0];
            VEC *v_GgLt_p = (VEC *)v_GgLt[0];
            VEC *v_RSQrqt_hat_p = (VEC *)v_RSQrqt_hat[0];
            VEC *v_Llt_p = (VEC *)v_Llt[0];
            VEC *v_Llt_shift_p = (VEC *)v_Llt_shift[0];
            VEC *v_GgIt_tilde_p = (VEC *)v_GgIt_tilde[0];
            VEC *v_GgLIt_p = (VEC *)v_GgLIt[0];
            VEC *v_HhIt_p = (VEC *)v_HhIt[0];
            VEC *v_PpIt_hat_p = (VEC *)v_PpIt_hat[0];
            VEC *v_LlIt_p = (VEC *)v_LlIt[0];
            VEC *v_Ggt_ineq_temp_p = (VEC *)v_Ggt_ineq_temp[0];

            /////////////// recursion ///////////////

            // last stage
            {
                const int nx = nx_p[K - 1];
                const int nu = nu_p[K - 1]; // this should be zero but is included here in case of misuse
                const int ng = ng_p[K - 1];
                const int ng_ineq = ng_ineq_p[K - 1];
                const int offs_ineq_k = offs_ineq_p[K - 1];
                const int offs_g_k = offs_g[K - 1];
                const int offs = offs_ux[K - 1];
                //         GECP(nx + 1, nx, RSQrqt_p + (K - 1), nu, nu, Ppt_p + K - 1, 0, 0);
                VECCP(nx, rhs_rq_p, offs + nu, v_Ppt_p + K - 1, 0);
                //         DIARE(nx, inertia_correction, Ppt_p + K - 1, 0, 0);
                //         //// inequalities
                if (ng_ineq > 0)
                {
                    //             GECP(nx, ng_ineq, Ggt_ineq_p + K - 1, nu, 0, Ggt_ineq_temp_p, 0, 0);
                    for (int i = 0; i < ng_ineq; i++)
                    {
                        //                 // kahan sum
                        double scaling_factor = VECEL(sigma_total_p, offs_ineq_k + i) + inertia_correction_w;
                        double grad_barrier = VECEL(rhs_gradb_p, offs_ineq_k + i);
                        //                 COLSC(nx, scaling_factor, Ggt_ineq_temp_p, 0, i);
                        //                 MATEL(Ggt_ineq_temp_p, nx, i) = grad_barrier + (scaling_factor)*MATEL(Ggt_ineq_p + K - 1, nu + nx, i);
                        VECEL(v_Ggt_ineq_temp_p, i) = grad_barrier + (scaling_factor)*VECEL(rhs_g_ineq_p, offs_ineq_k + i);
                    }
                    //             // add the penalty
                    //             SYRK_LN_MN(nx + 1, nx, ng_ineq, 1.0, Ggt_ineq_temp_p, 0, 0, Ggt_ineq_p + K - 1, nu, 0, 1.0, Ppt_p + K - 1, 0, 0, Ppt_p + K - 1, 0, 0);
                    GEMV_N(nx, ng_ineq, 1.0, Ggt_ineq_p + K - 1, 0, 0, v_Ggt_ineq_temp_p, 0, 1.0, v_Ppt_p + K - 1, 0, v_Ppt_p + K - 1, 0);
                    //             TRTR_L(nx, Ppt_p + K - 1, 0, 0, Ppt_p + K - 1, 0, 0);
                }
                //         // Hh_Km1 <- Gg_Km1
                //         GETR(nx + 1, ng, Ggt_p + (K - 1), nu, 0, Hh_p + (K - 1), 0, 0);
                VECCP(ng, rhs_g_p, offs_g_k, v_Hh_p + (K - 1), 0);
                //         gamma_p[K - 1] = ng;
                //         rho_p[K - 1] = 0;
            }
            for (int k = K - 2; k >= 0; --k)
            {
                const int nu = nu_p[k];
                const int nx = nx_p[k];
                const int nxp1 = nx_p[k + 1];
                const int ng = ng_p[k];
                const int ng_ineq = ng_ineq_p[k];
                const int offs_ineq_k = offs_ineq_p[k];
                const int offs_dyn_k = offs_dyn[k];
                const int offs_g_k = offs_g[k];
                const int offs = offs_ux[k];
                // calculate the size of H_{k+1} matrix
                const int Hp1_size = gamma_p[k + 1] - rho_p[k + 1];
                // if (Hp1_size + ng > nu + nx)
                //     return -1;
                // gamma_k <- number of eqs represented by Ggt_stripe
                const int gamma_k = Hp1_size + ng;
                // if (k==0) blasfeo_print_dvec(nxp1, v_Ppt_p+k+1, 0);
                //         //////// SUBSDYN
                {
                    //             // AL <- [BAb]^T_k P_kp1
                    //             GEMM_NT(nu + nx + 1, nxp1, nxp1, 1.0, BAbt_p + k, 0, 0, Ppt_p + k + 1, 0, 0, 0.0, AL_p, 0, 0, AL_p, 0, 0);
                    GEMV_N(nxp1, nxp1, 1.0, Ppt_p + k + 1, 0, 0, rhs_b_p, offs_dyn_k, 0.0, v_AL_p, 0, v_AL_p, 0);
                    //             // AL[-1,:] <- AL[-1,:] + p_kp1^T
                    //             GEAD(1, nxp1, 1.0, Ppt_p + (k + 1), nxp1, 0, AL_p, nx + nu, 0);
                    AXPY(nxp1, 1.0, v_Ppt_p + (k + 1), 0, v_AL_p, 0, v_AL_p, 0);
                    // if (k==K-2) blasfeo_print_dvec(nxp1, v_AL_p, 0);
                    //             // RSQrqt_stripe <- AL[BA] + RSQrqt
                    //             SYRK_LN_MN(nu + nx + 1, nu + nx, nxp1, 1.0, AL_p, 0, 0, BAbt_p + k, 0, 0, 1.0, RSQrqt_p + k, 0, 0, RSQrqt_tilde_p + k, 0, 0);
                    GEMV_N(nu + nx, nxp1, 1.0, BAbt_p + k, 0, 0, v_AL_p, 0, 1.0, rhs_rq_p, offs, v_RSQrqt_tilde_p + k, 0);
                    // if (k==K-2) blasfeo_print_dvec(nu+nx, v_RSQrqt_tilde_p+k, 0);
                    //             //// inequalities
                    if (ng_ineq > 0)
                    {
                        //                 GECP(nu + nx , ng_ineq, Ggt_ineq_p + k, 0, 0, Ggt_ineq_temp_p, 0, 0);
                        for (int i = 0; i < ng_ineq; i++)
                        {
                            double scaling_factor = VECEL(sigma_total_p, offs_ineq_k + i) + inertia_correction_w;
                            double grad_barrier = VECEL(rhs_gradb_p, offs_ineq_k + i);
                            //                     COLSC(nu + nx, scaling_factor, Ggt_ineq_temp_p, 0, i);
                            //                     MATEL(Ggt_ineq_temp_p, nu + nx, i) = grad_barrier + (scaling_factor)*MATEL(Ggt_ineq_p + k, nu + nx, i);
                            VECEL(v_Ggt_ineq_temp_p, i) = grad_barrier + (scaling_factor)*VECEL(rhs_g_ineq_p, offs_ineq_k + i);
                        }
                        //                 // add the penalty
                        //                 SYRK_LN_MN(nu + nx + 1, nu + nx, ng_ineq, 1.0, Ggt_ineq_temp_p, 0, 0, Ggt_ineq_p + k, 0, 0, 1.0, RSQrqt_tilde_p + k, 0, 0, RSQrqt_tilde_p + k, 0, 0);
                        GEMV_N(nu + nx, ng_ineq, 1.0, Ggt_ineq_p + k, 0, 0, v_Ggt_ineq_temp_p, 0, 1.0, v_RSQrqt_tilde_p + k, 0, v_RSQrqt_tilde_p + k, 0);
                    }
                    //             DIARE(nu + nx, inertia_correction, RSQrqt_tilde_p + k, 0, 0);
                    //             gamma_p[k] = gamma_k;
                    //             // if ng[k]>0
                    if (gamma_k > 0)
                    {
                        //                 // if Gk nonempty
                        if (ng > 0)
                        {
                            //                     // Ggt_stripe  <- Ggt_k
                            //                     GECP(nu + nx + 1, ng, Ggt_p + k, 0, 0, Ggt_stripe_p, 0, 0);
                            VECCP(ng, rhs_g_p, offs_g_k, v_Ggt_stripe_p, 0);
                        }
                        //                 // if Hkp1 nonempty
                        if (Hp1_size > 0)
                        {
                            //                     // Ggt_stripe <- [Ggt_k [BAb_k^T]H_kp1]
                            //                     GEMM_NT(nu + nx + 1, Hp1_size, nxp1, 1.0, BAbt_p + k, 0, 0, Hh_p + (k + 1), 0, 0, 0.0, Ggt_stripe_p, 0, ng, Ggt_stripe_p, 0, ng);
                            GEMV_N(Hp1_size, nxp1, 1.0, Hh_p + (k + 1), 0, 0, rhs_b_p, offs_dyn_k, 0.0, v_Ggt_stripe_p, ng, v_Ggt_stripe_p, ng);
                            //                     // Ggt_stripe[-1,ng:] <- Ggt_stripe[-1,ng:] + h_kp1^T
                            //                     GEADTR(1, Hp1_size, 1.0, Hh_p + (k + 1), 0, nxp1, Ggt_stripe_p, nu + nx, ng);
                            AXPY(Hp1_size, 1.0, v_Hh_p + (k + 1), 0, v_Ggt_stripe_p, ng, v_Ggt_stripe_p, ng);
                        }
                    }
                    else
                    {
                        //                 rho_p[k] = 0;
                        rank_k = 0;
                        v_RSQrq_hat_curr_p = v_RSQrqt_tilde_p + k;
                    }
                }
                //         //////// TRANSFORM_AND_SUBSEQ
                {
                    //             // symmetric transformation, done a little different than in paper, in order to fuse LA operations
                    //             // LU_FACT_TRANSPOSE(Ggtstripe[:gamma_k, nu+nx+1], nu max)
                    //             LU_FACT_transposed(gamma_k, nu + nx + 1, nu, rank_k, Ggt_stripe_p, Pl_p + k, Pr_p + k);
                    // TODO GET RID OF THIS OPERATION
                    rank_k = rho_p[k];
                    // if (k==K-2) blasfeo_print_dvec(gamma_k, v_Ggt_stripe_p, 0);
                    GECP(rank_k, gamma_k, Ggt_tilde_p + k, nu - rank_k + nx + 1, 0, Ggt_stripe_p, 0, 0);
                    (Pl_p + k)->PV(rank_k, v_Ggt_stripe_p, 0);
                    // L1^-1 g_stipe[:rho]
                    TRSV_UTU(rank_k, Ggt_stripe_p, 0, 0, v_Ggt_stripe_p, 0, v_Ggt_stripe_p, 0);
                    // -L2 L1^-1 g_stripe[:rho] + g_stripe[rho:]
                    GEMV_T(rank_k, gamma_k - rank_k, -1.0, Ggt_stripe_p, 0, rank_k, v_Ggt_stripe_p, 0, 1.0, v_Ggt_stripe_p, rank_k, v_Ggt_stripe_p, rank_k);

                    //             rho_p[k] = rank_k;
                    if (gamma_k - rank_k > 0)
                    {
                        //                 // transfer eq's to next stage
                        //                 GETR(nx + 1, gamma_k - rank_k, Ggt_stripe_p, nu, rank_k, Hh_p + k, 0, 0);
                        VECCP(gamma_k - rank_k, v_Ggt_stripe_p, rank_k, v_Hh_p + k, 0);
                    }
                    if (rank_k > 0)
                    {
                        //                 // Ggt_tilde_k <- Ggt_stripe[rho_k:nu+nx+1, :rho] L-T (note that this is slightly different from the implementation)
                        //                 TRSM_RLNN(nu - rank_k + nx + 1, rank_k, -1.0, Ggt_stripe_p, 0, 0, Ggt_stripe_p, rank_k, 0, Ggt_tilde_p + k, 0, 0);
                        VECCPSC(rank_k, -1.0, v_Ggt_stripe_p, 0, v_Ggt_tilde_p + k, 0);
                        TRSV_LTN(rank_k, Ggt_stripe_p, 0, 0, v_Ggt_tilde_p + k, 0, v_Ggt_tilde_p + k, 0);
                        //                 // the following command copies the top block matrix (LU) to the bottom because it it needed later
                        //                 GECP(rank_k, gamma_k, Ggt_stripe_p, 0, 0, Ggt_tilde_p + k, nu - rank_k + nx + 1, 0);
                        //                 // permutations
                        //                 TRTR_L(nu + nx, RSQrqt_tilde_p + k, 0, 0, RSQrqt_tilde_p + k, 0, 0); // copy lower part of RSQ to upper part
                        //                 (Pr_p + k)->PM(rank_k, RSQrqt_tilde_p + k);                          // TODO make use of symmetry
                        (Pr_p + k)->PV(rank_k, v_RSQrqt_tilde_p + k, 0);
                        //                 (Pr_p + k)->MPt(rank_k, RSQrqt_tilde_p + k);
                        //                 // GL <- Ggt_tilde_k @ RSQ[:rho,:nu+nx] + RSQrqt[rho:nu+nx+1, rho:] (with RSQ[:rho,:nu+nx] = RSQrqt[:nu+nx,:rho]^T)
                        //                 // GEMM_NT(nu - rank_k + nx + 1, nu + nx, rank_k, 1.0, Ggt_tilde_p + k, 0, 0, RSQrqt_tilde_p + k, 0, 0, 1.0, RSQrqt_tilde_p + k, rank_k, 0, GgLt_p, 0, 0);
                        //                 // split up because valgrind was giving invalid read errors when C matrix has nonzero row offset
                        //                 // GgLt[0].print();
                        //                 GECP(nu - rank_k + nx + 1, nu + nx, RSQrqt_tilde_p + k, rank_k, 0, GgLt_p, 0, 0);
                        VECCP(nu + nx, v_RSQrqt_tilde_p + k, 0, v_GgLt_p, 0);
                        //                 GEMM_NT(nu - rank_k + nx + 1, nu + nx, rank_k, 1.0, Ggt_tilde_p + k, 0, 0, RSQrqt_tilde_p + k, 0, 0, 1.0, GgLt_p, 0, 0, GgLt_p, 0, 0);
                        GEMV_N(nu + nx, rank_k, 1.0, RSQrqt_tilde_p + k, 0, 0, v_Ggt_tilde_p + k, 0, 1.0, v_GgLt_p, 0, v_GgLt_p, 0);
                        //                 // RSQrqt_hat = GgLt[nu-rank_k + nx +1, :rank_k] * G[:rank_k, :nu+nx] + GgLt[rank_k:, :]  (with G[:rank_k,:nu+nx] = Gt[:nu+nx,:rank_k]^T)
                        //                 SYRK_LN_MN(nu - rank_k + nx + 1, nu + nx - rank_k, rank_k, 1.0, GgLt_p, 0, 0, Ggt_tilde_p + k, 0, 0, 1.0, GgLt_p, 0, rank_k, RSQrqt_hat_p, 0, 0);
                        GEMV_N(nu + nx - rank_k, rank_k, 1.0, Ggt_tilde_p + k, 0, 0, v_GgLt_p, 0, 1.0, v_GgLt_p, rank_k, v_RSQrqt_hat_p, 0);
                        //                 // GEMM_NT(nu - rank_k + nx + 1, nu + nx - rank_k, rank_k, 1.0, GgLt_p, 0, 0, Ggt_tilde_p + k, 0, 0, 1.0, GgLt_p, 0, rank_k, RSQrqt_hat_p, 0, 0);
                        v_RSQrq_hat_curr_p = v_RSQrqt_hat_p;
                        //    RSQrq_hat_curr_p = RSQrqt_hat_p;
                    }
                    else
                    {
                        v_RSQrq_hat_curr_p = v_RSQrqt_tilde_p + k;
                        //    RSQrq_hat_curr_p = RSQrqt_tilde_p + k;
                    }
                    // if (k==K-2) blasfeo_print_dvec(nu+nx-rank_k, v_RSQrq_hat_curr_p, 0);
                }
                //         //////// SCHUR
                {
                    if (nu - rank_k > 0)
                    {
                        //                 // DLlt_k = [chol(R_hatk); Llk@chol(R_hatk)^-T]
                        //                 POTRF_L_MN(nu - rank_k + nx + 1, nu - rank_k, RSQrq_hat_curr_p, 0, 0, Llt_p + k, 0, 0);

                        TRSV_LNN(nu - rank_k, Llt_p + k, 0, 0, v_RSQrq_hat_curr_p, 0, v_Llt_p + k, 0);
                        //                 if (!check_reg(nu - rank_k, Llt_p + k, 0, 0))
                        //                     return 1;
                        //                 // Pp_k = Qq_hatk - L_k^T @ Ll_k
                        //                 // SYRK_LN_MN(nx+1, nx, nu-rank_k, -1.0,Llt_p+k, nu-rank_k,0, Llt_p+k, nu-rank_k,0, 1.0, RSQrq_hat_curr_p, nu-rank_k, nu-rank_k,Pp+k,0,0); // feature not implmented yet
                        ///// TODO get rid ot this operation
                        GECP(nx + 1, nu - rank_k, Llt_p + k, nu - rank_k, 0, Llt_shift_p, 0, 0); // needless operation because feature not implemented yet
                        VECCP(nu - rank_k, v_Llt_p + k, 0, v_Llt_shift_p, 0);
                        //                 // SYRK_LN_MN(nx + 1, nx, nu - rank_k, -1.0, Llt_shift_p, 0, 0, Llt_shift_p, 0, 0, 1.0, RSQrq_hat_curr_p, nu - rank_k, nu - rank_k, Ppt_p + k, 0, 0);
                        //                 GECP(nx + 1, nx, RSQrq_hat_curr_p, nu - rank_k, nu - rank_k, Ppt_p + k, 0, 0);
                        VECCP(nx, v_RSQrq_hat_curr_p, nu - rank_k, v_Ppt_p + k, 0);
                        //                 SYRK_LN_MN(nx + 1, nx, nu - rank_k, -1.0, Llt_shift_p, 0, 0, Llt_shift_p, 0, 0, 1.0, Ppt_p + k, 0, 0, Ppt_p + k, 0, 0);
                        GEMV_N(nx, nu - rank_k, -1.0, Llt_shift_p, 0, 0, v_Llt_shift_p, 0, 1.0, v_Ppt_p + k, 0, v_Ppt_p + k, 0);
                        //                 // next steps are for better accuracy
                        if (increased_accuracy)
                        {
                            //                     // copy eta
                            //                     GETR(nu - rank_k, gamma_k - rank_k, Ggt_stripe_p, rank_k, rank_k, Ggt_stripe_p, 0, 0);
                            //                     // blasfeo_print_dmat(gamma_k-rank_k, nu-rank_k, Ggt_stripe_p, 0,0);
                            //                     // eta L^-T
                            //                     TRSM_RLTN(gamma_k - rank_k, nu - rank_k, 1.0, Llt_p + k, 0, 0, Ggt_stripe_p, 0, 0, Ggt_stripe_p, 0, 0);
                            //                     // ([S^T \\ r^T] L^-T) @ (L^-1 eta^T)
                            //                     // (eta L^-T) @ ([S^T \\ r^T] L^-T)^T
                            //                     GEMM_NT(gamma_k - rank_k, nx + 1, nu - rank_k, -1.0, Ggt_stripe_p, 0, 0, Llt_p + k, nu - rank_k, 0, 1.0, Hh_p + k, 0, 0, Hh_p + k, 0, 0);
                            //                     // keep (L^-1 eta^T) for forward recursion
                            //                     GETR(gamma_k - rank_k, nu - rank_k, Ggt_stripe_p, 0, 0, Ggt_tilde_p + k, 0, rank_k);
                            GEMV_T(nu - rank_k, gamma_k - rank_k, -1.0, Ggt_tilde_p + k, 0, rank_k, v_Llt_p + k, 0, 1.0, v_Hh_p + k, 0, v_Hh_p + k, 0);
                        }
                    }
                    else
                    {
                        //                 GECP(nx + 1, nx, RSQrq_hat_curr_p, 0, 0, Ppt_p + k, 0, 0);
                        VECCP(nx, v_RSQrq_hat_curr_p, 0, v_Ppt_p + k, 0);
                    }
                    //             TRTR_L(nx, Ppt_p + k, 0, 0, Ppt_p + k, 0, 0);
                }
            }
            //     rankI = 0;
            //     //////// FIRST_STAGE
            {
                const int nx = nx_p[0];
                int gamma_I = gamma_p[0] - rho_p[0];
                //         if (gamma_I > nx)
                //         {
                //             return -3;
                //         }
                if (gamma_I > 0)
                {
                    //             GETR(gamma_I, nx + 1, Hh_p + 0, 0, 0, HhIt_p, 0, 0); // transposition may be avoided
                    VECCP(gamma_I, v_Hh_p + 0, 0, v_HhIt_p, 0);

                    //             // HhIt[0].print();
                    //             LU_FACT_transposed(gamma_I, nx + 1, nx, rankI, HhIt_p, PlI_p, PrI_p);
                    PlI_p->PV(rankI, v_HhIt_p, 0);
                    // L1^-1 g_stipe[:rho]
                    TRSV_UTU(rankI, HhIt_p, 0, 0, v_HhIt_p, 0, v_HhIt_p, 0);
                    // -L2 L1^-1 g_stripe[:rho] + g_stripe[rho:]
                    GEMV_T(rankI, gamma_I - rankI, -1.0, HhIt_p, 0, rankI, v_HhIt_p, 0, 1.0, v_HhIt_p, rankI, v_HhIt_p, rankI);
                    //             if (rankI < gamma_I)
                    //                 return -2;
                    //             // PpIt_tilde <- Ggt[rankI:nx+1, :rankI] L-T (note that this is slightly different from the implementation)
                    //             TRSM_RLNN(nx - rankI + 1, rankI, -1.0, HhIt_p, 0, 0, HhIt_p, rankI, 0, GgIt_tilde_p, 0, 0);
                    VECCPSC(rankI, -1.0, v_HhIt_p, 0, v_GgIt_tilde_p, 0);
                    TRSV_LTN(rankI, HhIt_p, 0, 0, v_GgIt_tilde_p, 0, v_GgIt_tilde_p, 0);
                    //             // permutations
                    //             (PrI_p)->PM(rankI, Ppt_p); // TODO make use of symmetry
                    (PrI_p)->PV(rankI, v_Ppt_p, 0);
                    //             (PrI_p)->MPt(rankI, Ppt_p);
                    //             // // GL <- GgIt_tilde @ Pp[:rankI,:nx] + Ppt[rankI:nx+1, rankI:] (with Pp[:rankI,:nx] = Ppt[:nx,:rankI]^T)
                    //             // GEMM_NT(nx - rankI + 1, nx, rankI, 1.0, GgIt_tilde_p, 0, 0, Ppt_p, 0, 0, 1.0, Ppt_p, rankI, 0, GgLIt_p, 0, 0);
                    //             // split up because valgrind was giving invalid read errors when C matrix has nonzero row offset
                    //             GECP(nx - rankI + 1, nx, Ppt_p, rankI, 0, GgLIt_p, 0, 0);
                    VECCP(nx, v_Ppt_p, 0, v_GgLIt_p, 0);
                    //             GEMM_NT(nx - rankI + 1, nx, rankI, 1.0, GgIt_tilde_p, 0, 0, Ppt_p, 0, 0, 1.0, GgLIt_p, 0, 0, GgLIt_p, 0, 0);
                    GEMV_N(nx, rankI, 1.0, Ppt_p, 0, 0, v_GgIt_tilde_p, 0, 1.0, v_GgLIt_p, 0, v_GgLIt_p, 0);
                    //             // // RSQrqt_hat = GgLt[nu-rank_k + nx +1, :rank_k] * G[:rank_k, :nu+nx] + GgLt[rank_k:, :]  (with G[:rank_k,:nu+nx] = Gt[:nu+nx,:rank_k]^T)
                    //             SYRK_LN_MN(nx - rankI + 1, nx - rankI, rankI, 1.0, GgLIt_p, 0, 0, GgIt_tilde_p, 0, 0, 1.0, GgLIt_p, 0, rankI, PpIt_hat_p, 0, 0);
                    GEMV_N(nx - rankI, rankI, 1.0, GgIt_tilde_p, 0, 0, v_GgLIt_p, 0, 1.0, v_GgLIt_p, rankI, v_PpIt_hat_p, 0);
                    //             // TODO skipped if nx-rankI = 0
                    //             POTRF_L_MN(nx - rankI + 1, nx - rankI, PpIt_hat_p, 0, 0, LlIt_p, 0, 0);
                    TRSV_LNN(nx - rankI, LlIt_p, 0, 0, v_PpIt_hat_p, 0, v_LlIt_p, 0);
                    //             if (!check_reg(nx - rankI, LlIt_p, 0, 0))
                    //                 return 2;
                }
                else
                {
                    //             rankI = 0;
                    //             POTRF_L_MN(nx + 1, nx, Ppt_p, 0, 0, LlIt_p, 0, 0);
                    TRSV_LNN(nx, LlIt_p, 0, 0, v_LlIt_p, 0, v_LlIt_p, 0);
                    //             if (!check_reg(nx, LlIt_p, 0, 0))
                    //                 return 2;
                }
            }
            //     ////// FORWARD_SUBSTITUTION:
            //     // first stage
            {
                const int nx = nx_p[0];
                const int nu = nu_p[0];
                //         // calculate xIb
                //         ROWEX(nx - rankI, -1.0, LlIt_p, nx - rankI, 0, ux_p, nu + rankI);
                VECCPSC(nx - rankI, -1.0, v_LlIt_p, 0, ux_p, nu + rankI);
                //         // assume TRSV_LTN allows aliasing, this is the case in normal BLAS
                TRSV_LTN(nx - rankI, LlIt_p, 0, 0, ux_p, nu + rankI, ux_p, nu + rankI);
                //         // calculate xIa
                //         ROWEX(rankI, 1.0, GgIt_tilde_p, nx - rankI, 0, ux_p, nu);
                VECCP(rankI, v_GgIt_tilde_p, 0, ux_p, nu);
                //         // assume aliasing is possible for last two elements
                GEMV_T(nx - rankI, rankI, 1.0, GgIt_tilde_p, 0, 0, ux_p, nu + rankI, 1.0, ux_p, nu, ux_p, nu);
                //         //// lag
                // ROWEX(rankI, -1.0, Ppt_p, nx, 0, lam_p, 0);
                VECCPSC(rankI, -1.0, v_Ppt_p, 0, lam_p, 0);
                //         // assume aliasing is possible for last two elements
                GEMV_T(nx, rankI, -1.0, Ppt_p, 0, 0, ux_p, nu, 1.0, lam_p, 0, lam_p, 0);
                //         // U^-T
                TRSV_LNN(rankI, HhIt_p, 0, 0, lam_p, 0, lam_p, 0);
                //         // L^-T
                TRSV_UNU(rankI, rankI, HhIt_p, 0, 0, lam_p, 0, lam_p, 0);
                (PlI_p)->PtV(rankI, lam_p, 0);
                (PrI_p)->PtV(rankI, ux_p, nu);
                // blasfeo_print_dvec(rankI, lam_p, 0);
            }
            //     int *offs_ux = (int *)OCP->aux.ux_offs.data();
            //     int *offs_g = (int *)OCP->aux.g_offs.data();
            //     int *offs_dyn_eq = (int *)OCP->aux.dyn_eq_offs.data();
            //     // other stages
            //     // for (int k = 0; k < K - 1; k++)
            //     // int dyn_eqs_ofs = offs_g[K - 1] + ng_p[K - 1]; // this value is incremented at end of recursion
            for (int k = 0; k < K - 1; k++)
            {
                const int nx = nx_p[k];
                const int nu = nu_p[k];
                const int nxp1 = nx_p[k + 1];
                const int nup1 = nu_p[k + 1];
                const int offsp1 = offs_ux[k + 1];
                const int offs = offs_ux[k];
                const int rho_k = rho_p[k];
                const int numrho_k = nu - rho_k;
                const int offs_g_k = offs_g[k];
                const int offs_dyn_eq_k = offs_dyn_eq[k];
                const int offs_dyn_k = offs_dyn[k];
                const int offs_g_kp1 = offs_g[k + 1];
                const int gammamrho_k = gamma_p[k] - rho_p[k];
                const int gamma_k = gamma_p[k];
                const int gammamrho_kp1 = gamma_p[k + 1] - rho_p[k + 1];
                if (numrho_k > 0)
                {
                    //             /// calculate ukb_tilde
                    //             // -Lkxk - lk
                    //             ROWEX(numrho_k, -1.0, Llt_p + k, numrho_k + nx, 0, ux_p, offs + rho_k);
                    VECCPSC(numrho_k, -1.0, v_Llt_p + k, 0, ux_p, offs + rho_k);
                    if (increased_accuracy)
                    {
                        GEMV_N(nu - rho_k, gamma_k - rho_k, -1.0, Ggt_tilde_p + k, 0, rho_k, lam_p, offs_g_k, 1.0, ux_p, offs + rho_k, ux_p, offs + rho_k);
                    }
                    //             // assume aliasing of last two eliments is allowed
                    GEMV_T(nx, numrho_k, -1.0, Llt_p + k, numrho_k, 0, ux_p, offs + nu, 1.0, ux_p, offs + rho_k, ux_p, offs + rho_k);
                    for (int ii = 0; ii < numrho_k; ii++)
                    {
                        double noise = alpha * distribution(generator);
                        VECEL(ux_p, offs + rho_k + ii) += noise;
                    }
                    TRSV_LTN(numrho_k, Llt_p + k, 0, 0, ux_p, offs + rho_k, ux_p, offs + rho_k);
                }
                //         /// calcualate uka_tilde
                if (rho_k > 0)
                {
                    // ROWEX(rho_k, 1.0, Ggt_tilde_p + k, numrho_k + nx, 0, ux_p, offs);
                    VECCP(rho_k, v_Ggt_tilde_p + k, 0, ux_p, offs);
                    GEMV_T(nx + numrho_k, rho_k, 1.0, Ggt_tilde_p + k, 0, 0, ux_p, offs + rho_k, 1.0, ux_p, offs, ux_p, offs);
                    //             // calculate lamda_tilde_k
                    //             // copy vk to right location
                    //             // we implemented a version of vector copy that starts with copy of last element, to avoid aliasing error
                    VECCPR(gammamrho_k, lam_p, offs_g_k, lam_p, offs_g_k + rho_k);
                    // ROWEX(rho_k, -1.0, RSQrqt_tilde_p + k, nu + nx, 0, lam_p, offs_g_k);
                    VECCPSC(rho_k, -1.0, v_RSQrqt_tilde_p + k, 0, lam_p, offs_g_k);
                    //             // assume aliasing of last two eliments is allowed
                    GEMV_T(nu + nx, rho_k, -1.0, RSQrqt_tilde_p + k, 0, 0, ux_p, offs, 1.0, lam_p, offs_g_k, lam_p, offs_g_k);
                    //             // nu-rank_k+nx,0
                    //             // needless copy because feature not implemented yet in trsv_lnn
                    GECP(rho_k, gamma_k, Ggt_tilde_p + k, nu - rho_k + nx + 1, 0, AL_p, 0, 0);
                    //             // U^-T
                    TRSV_LNN(rho_k, AL_p, 0, 0, lam_p, offs_g_k, lam_p, offs_g_k);
                    //             // L^-T
                    TRSV_UNU(rho_k, gamma_k, AL_p, 0, 0, lam_p, offs_g_k, lam_p, offs_g_k);
                    (Pl_p + k)->PtV(rho_k, lam_p, offs_g_k);
                    (Pr_p + k)->PtV(rho_k, ux_p, offs);
                }
                //         // calculate xkp1
                //         ROWEX(nxp1, 1.0, BAbt_p + k, nu + nx, 0, ux_p, offsp1 + nup1);
                VECCP(nxp1, rhs_b_p, offs_dyn_k, ux_p, offsp1 + nup1);
                GEMV_T(nu + nx, nxp1, 1.0, BAbt_p + k, 0, 0, ux_p, offs, 1.0, ux_p, offsp1 + nup1, ux_p, offsp1 + nup1);
                //         // calculate lam_dyn xp1
                //         ROWEX(nxp1, 1.0, Ppt_p + (k + 1), nxp1, 0, lam_p, offs_dyn_eq_k);
                VECCP(nxp1, v_Ppt_p + (k + 1), 0, lam_p, offs_dyn_eq_k);
                GEMV_T(nxp1, nxp1, 1.0, Ppt_p + (k + 1), 0, 0, ux_p, offsp1 + nup1, 1.0, lam_p, offs_dyn_eq_k, lam_p, offs_dyn_eq_k);
                GEMV_T(gammamrho_kp1, nxp1, 1.0, Hh_p + (k + 1), 0, 0, lam_p, offs_g_kp1, 1.0, lam_p, offs_dyn_eq_k, lam_p, offs_dyn_eq_k);
            }
            for (int k = 0; k < K; k++)
            {
                const int nx = nx_p[k];
                const int nu = nu_p[k];
                const int ng_ineq = ng_ineq_p[k];
                const int offs = offs_ux[k];
                const int offs_g_ineq_k = offs_g_ineq_p[k];
                const int offs_ineq_k = offs_ineq_p[k];
                if (ng_ineq > 0)
                {
                    //             // calculate delta_s
                    //             ROWEX(ng_ineq, 1.0, Ggt_ineq_p + k, nu + nx, 0, delta_s_p, offs_ineq_k);
                    VECCP(ng_ineq, rhs_g_ineq_p, offs_ineq_k, delta_s_p, offs_ineq_k);
                    //             // GEMV_T(nu + nx, ng_ineq, 1.0, Ggt_ineq_p + k, 0, 0, ux_p, offs, 1.0, delta_s_p, offs_ineq_k, delta_s_p, offs_ineq_k);
                    GEMV_T(nu + nx, ng_ineq, 1.0, Ggt_ineq_p + k, 0, 0, ux_p, offs, 1.0, delta_s_p, offs_ineq_k, delta_s_p, offs_ineq_k);
                    //             // calculate lamineq
                    for (int i = 0; i < ng_ineq; i++)
                    {
                        double scaling_factor = VECEL(sigma_total_p, offs_ineq_k + i) + inertia_correction_w;
                        double grad_barrier = VECEL(rhs_gradb_p, offs_ineq_k + i);
                        double ds = VECEL(delta_s_p, offs_ineq_k + i);
                        VECEL(lam_p, offs_g_ineq_k + i) = scaling_factor * ds + grad_barrier;
                    }
                }
            }
            return 0;
        };
    };
} // namespace fatrop

#endif