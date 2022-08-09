#ifndef OCPBUILDERINCLUDED
#define OCPBUILDERINCLUDED
#include "ocp/BFOCPBasic.hpp"
#include "ocp/BFOCPAdapter.hpp"
#include "ocp/OCPLSRiccati.hpp"
#include "ocp/OCPNoScaling.hpp"
#include "solver/FatropParams.hpp"
#include "solver/Filter.hpp"
#include "ocp/FatropOCP.hpp"
#include "solver/FatropAlg.hpp"
#include <string>
#include <fstream>
#include <iostream>
#include "json/json.h"
#include <sstream>
namespace fatrop
{
    class OCPBuilder
    {
    public:
        OCPBuilder(const string &functions, const string &json_spec_file)
        {
            std::ifstream t(json_spec_file);
            std::stringstream buffer;
            buffer << t.rdbuf();
            json::jobject json_spec = json::jobject::parse(buffer.str());
            K = json_spec["K"];
            const int nx = json_spec["nx"];
            const int nu = json_spec["nu"];
            const int ngI = json_spec["ngI"];
            const int ngF = json_spec["ngF"];
            const int ng_ineq = json_spec["ng_ineq"];
            const int ng_ineqF = json_spec["ng_ineqF"];
            const int n_stage_params = json_spec["n_stage_params"];
            const int n_global_params = json_spec["n_global_params"];
            shared_ptr<DLHandler> handle = make_shared<DLHandler>(functions);
            EvalCasGen BAbtf(handle, "BAbt");
            EvalCasGen bkf(handle, "bk");
            EvalCasGen RSQrqtIf(handle, "RSQrqtI");
            EvalCasGen rqIf(handle, "rqI");
            EvalCasGen RSQrqtf(handle, "RSQrqt");
            EvalCasGen rqf(handle, "rqk");
            EvalCasGen RSQrqtFf(handle, "RSQrqtF");
            EvalCasGen rqFf(handle, "rqF");
            EvalCasGen GgtIf(handle, "GgtI");
            EvalCasGen gIf(handle, "gI");
            EvalCasGen GgtFf(handle, "GgtF");
            EvalCasGen gFf(handle, "gF");
            EvalCasGen Lkf(handle, "Lk");
            EvalCasGen LFf(handle, "LF");
            EvalCasGen Ggineqtf(handle, "Ggineqt");
            EvalCasGen gineqf(handle, "gineq");
            EvalCasGen GgineqFtf(handle, "GgineqFt");
            EvalCasGen gineqFf(handle, "gineqF");
            shared_ptr<BFOCP> ocptemplatebasic = make_shared<BFOCPBasic>(nu, nx, ngI, ngF, ng_ineq, ng_ineqF, n_stage_params, n_global_params, K,
                                                                 BAbtf,
                                                                 bkf,
                                                                 RSQrqtIf,
                                                                 rqIf,
                                                                 RSQrqtf,
                                                                 rqf,
                                                                 RSQrqtFf,
                                                                 rqFf,
                                                                 GgtIf,
                                                                 gIf,
                                                                 GgtFf,
                                                                 gFf,
                                                                 Ggineqtf,
                                                                 gineqf,
                                                                 GgineqFtf,
                                                                 gineqFf,
                                                                 Lkf,
                                                                 LFf);
            ocptempladapter = make_shared<BFOCPAdapter>(ocptemplatebasic);
            ocptempladapter->SetParams(json_spec["stage_params"].get_number_array<double>("%lf"), json_spec["global_params"].get_number_array<double>("%lf"));
            ocplsriccati = make_shared<OCPLSRiccati>(ocptempladapter->GetOCPDims());
            params = make_shared<FatropParams>();
            ocpscaler = make_shared<OCPNoScaling>(params);
            fatropocp = make_shared<FatropOCP>(ocptempladapter, ocplsriccati, ocpscaler);
            fatropdata = make_shared<FatropData>(fatropocp->GetNLPDims(), params);
            initial_u = json_spec["initial_u"].get_number_array<double>("%lf");
            initial_x = json_spec["initial_x"].get_number_array<double>("%lf");
            lower = json_spec["lower"].get_number_array<double>("%lf");
            upper = json_spec["upper"].get_number_array<double>("%lf");
            lowerF = json_spec["lowerF"].get_number_array<double>("%lf");
            upperF = json_spec["upperF"].get_number_array<double>("%lf");
            lower.insert(lower.end(), lowerF.begin(), lowerF.end());
            upper.insert(upper.end(), upperF.begin(), upperF.end());
            SetBounds();
            SetInitial();
            // vector<double> upper = vector<double>(lower.size(), INFINITY);
            filter = make_shared<Filter>(params->maxiter + 1);
            journaller = make_shared<Journaller>(params->maxiter + 1);
            linesearch = make_shared<BackTrackingLineSearch>(params, fatropocp, fatropdata, filter, journaller);
            fatropalg = make_shared<FatropAlg>(fatropocp, fatropdata, params, filter, linesearch, journaller);
            // blasfeo_timer timer;
            // blasfeo_tic(&timer);
            // fatropalg->Optimize();
            // double el = blasfeo_toc(&timer);
            // cout << "el time " << el << endl;
        }
        void SetBounds()
        {
            ocptempladapter->SetInitial(K, fatropdata, initial_u, initial_x);
        }
        void SetInitial()
        {
            fatropdata->SetBounds(lower, upper);
        }
        int K;
        shared_ptr<OCP> ocptempladapter;
        shared_ptr<OCPLinearSolver> ocplsriccati;
        shared_ptr<FatropParams> params;
        shared_ptr<OCPScalingMethod> ocpscaler;
        shared_ptr<FatropNLP> fatropocp;
        shared_ptr<FatropData> fatropdata;
        vector<double> initial_u;
        vector<double> initial_x;
        vector<double> lower;
        vector<double> upper;
        vector<double> lowerF;
        vector<double> upperF;
        shared_ptr<Filter> filter;
        shared_ptr<Journaller> journaller;
        shared_ptr<LineSearch> linesearch;
        shared_ptr<FatropAlg> fatropalg;
    };
}
#endif // OCPBUILDERINCLUDED