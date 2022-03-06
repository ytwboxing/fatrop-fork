from RobotSpecification import *
import numpy as np
import numpy.matlib
from FatropOCPSpecification import *
K = 50
robotspecification = RobotSpecification()
optibuilder = OptiBuilder(robotspecification)
opti = optibuilder.set_up_Opti(K)
dt = 2.0/K
stage_params = np.matlib.repmat(np.array(dt), 1, K)
global_params = np.array([0.80,0.00,0.50])
alpha = 0.5
inits_x = np.matlib.repmat((alpha*robotspecification.joint_lower + (1-alpha)*robotspecification.joint_upper)[:, np.newaxis], 1, K)
# inits_x = np.matlib.repmat((np.array(7*[0.0]))[:, np.newaxis], 1, K)
inits_u = np.zeros((robotspecification.nu, K-1))
opti.set_value(optibuilder.stage_params_in, stage_params)
opti.set_value(optibuilder.global_params_in, global_params)
opti.set_initial(optibuilder.x_vars, inits_x)
opti.set_initial(optibuilder.u_vars, inits_u)
# opti.solver("ipopt", {"expand":True}, {'print_level':5, 'max_soc':0, 'tol':1e-5})
# opti.solver("ipopt", {"expand":True}, {'print_level':8, 'tol':1e-5, 'max_iter':500, "max_soc":0, "min_refinement_steps":0, 'bound_relax_factor':0.0, 'nlp_scaling_method':None})
opti.solver("ipopt", {"expand":True}, {'print_level':5, 'tol':1e-5, 'max_iter':500, "max_soc":0, "min_refinement_steps":0, 'bound_relax_factor':0.0, 'nlp_scaling_method':'none', 'kappa_sigma':0.1})
opti.solve()
# opti.solve()
lower = np.matlib.repmat(robotspecification.lower[:,np.newaxis], 1,K-1)
upper = np.matlib.repmat(robotspecification.upper[:,np.newaxis], 1,K-1)
jsongen = JSONGenerator(robotspecification)
jsongen.generate_JSON('test.json', K, stage_params, global_params, inits_x, inits_u, lower, upper)
codegen = FatropOCPCodeGenerator(robotspecification)
codegen.generate_code("f.c")