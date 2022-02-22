#include "ocp/BFOCPBasic.hpp"
// #include "OCP/BFOCP.hpp"
#include "ocp/BFOCPAdapter.hpp"
// #include "AUX/SmartPtr.hpp"
// #include "AUX/FatropMemory.hpp"
// #include "OCP/FatropOCP.hpp"
// #include "OCP/OCPLinearSolver.hpp"
#include "ocp/OCPLSRiccati.hpp"
// #include "AUX/FatropVector.hpp"
// #include "BLASFEO_WRAPPER/LinearAlgebraBlasfeo.hpp"
#include "ocp/OCPNoScaling.hpp"
#include "solver/FatropParams.hpp"
// #include "SOLVER/FatropAlg.hpp"
// #ioclude "SPARSE/SparseOCP.hpp"
#include "solver/Filter.hpp"
#include "ocp/FatropOCP.hpp"
#include "solver/FatropAlg.hpp"
using namespace fatrop;
int main()
{
    RefCountPtr<BFOCP> ocptemplatebasic =
        new BFOCPBasic(BFOCPBasic::from_shared_lib("./../../ocp_specification/f.so", 100));
    RefCountPtr<OCP> ocptempladapter = new BFOCPAdapter(ocptemplatebasic);
    RefCountPtr<OCPLinearSolver> ocplsriccati = new OCPLSRiccati(ocptempladapter->GetOCPDims());
    RefCountPtr<FatropParams> params = new FatropParams();
    RefCountPtr<OCPScalingMethod> ocpscaler = new OCPNoScaling(params);
    RefCountPtr<FatropNLP> fatropocp = new FatropOCP(ocptempladapter, ocplsriccati, ocpscaler);
    RefCountPtr<FatropData> fatropdata = new FatropData(fatropocp->GetNLPDims(), params);
    RefCountPtr<Filter> filter(new Filter(params->maxiter + 1));
    RefCountPtr<Journaller> journaller(new Journaller(params->maxiter + 1));
    RefCountPtr<LineSearch> linesearch = new BackTrackingLineSearch(params, fatropocp, fatropdata, filter, journaller);
    RefCountPtr<FatropAlg> fatropalg = new FatropAlg(fatropocp, fatropdata, params, filter, linesearch, journaller);
    blasfeo_timer timer;
    VECSE(fatropdata->x_curr.nels(), 1.0, (VEC *)fatropdata->x_curr, 0);
    VECSE(fatropdata->s_lower.nels(), 0.0, (VEC *)fatropdata->s_lower, 0);
    VECSE(fatropdata->s_upper.nels(), INFINITY, (VEC *)fatropdata->s_upper, 0);
    // VECSE(fatropdata->s_lower.nels(), -INFINITY, (VEC *) fatropdata->s_lower, 0);
    // VECSE(fatropdata->s_upper.nels(), 0.0, (VEC *) fatropdata->s_upper, 0);
    double el = 0.0;
    const int N = 1;
    for (int i = 0; i < N; i++)
    {
        VECSE(fatropdata->x_curr.nels(), 1.0, (VEC *)fatropdata->x_curr, 0);
        fatropdata->Initialize();
        blasfeo_tic(&timer);
        fatropalg->Optimize();
        el += blasfeo_toc(&timer);
    }
    // fatropalg->journaller_->PrintIterations();
    cout << "el time " << el/N << endl;
    // journaller->PrintIterations();
}