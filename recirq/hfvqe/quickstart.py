##
"""
This code tutorial shows how to estimate a 1-RDM and perform variational optimization
"""
# Import library functions and define a helper function
import numpy as np
import cirq

from recirq.hfvqe.gradient_hf import rhf_func_generator
from recirq.hfvqe.opdm_functionals import OpdmFunctional
from recirq.hfvqe.analysis import (compute_opdm,
                            mcweeny_purification,
                            resample_opdm,
                            fidelity_witness,
                            fidelity)
from recirq.hfvqe.third_party.higham import fixed_trace_positive_projection
from recirq.hfvqe.molecular_example import make_h6_1_3
##
"""
Generate the input files, set up quantum resources, and set up the OpdmFunctional to make measurements.
"""
rhf_objective, molecule, parameters, obi, tbi = make_h6_1_3()
ansatz, energy, gradient = rhf_func_generator(rhf_objective)

# settings for quantum resources
qubits = [cirq.GridQubit(0, x) for x in range(molecule.n_orbitals)]
sampler = cirq.Simulator(dtype=np.complex128)  # this can be a QuantumEngine

# OpdmFunctional contains an interface for running experiments
opdm_func = OpdmFunctional(qubits=qubits,
                           sampler=sampler,
                           constant=molecule.nuclear_repulsion,
                           one_body_integrals=obi,
                           two_body_integrals=tbi,
                           num_electrons=molecule.n_electrons // 2,  # only simulate spin-up electrons
                           clean_xxyy=True,
                           purification=True
                           )
##
"""
The displayed text is the output of the gradient based restricted Hartree-Fock. 
We define the gradient in rhf_objective and use the conjugate-gradient optimizer to optimize 
the basis rotation parameters. This is equivalent to doing Hartree-Fock theory 
from the canonical transformation perspective.

Next, we will do the following:
1. Do measurements for a given set of parameters
2. Compute 1-RDM, variances, and purification
3. Compute energy, fidelities, and errorbars
"""
# 1.
# default to 250_000 shots for each circuit.
# 7 circuits total, printed for your viewing pleasure
# return value is a dictionary with circuit results for each permutation
measurement_data = opdm_func.calculate_data(parameters)

# 2.
opdm, var_dict = compute_opdm(measurement_data,
                              return_variance=True)
opdm_pure = mcweeny_purification(opdm)

# 3.
raw_energies = []
raw_fidelity_witness = []
purified_eneriges = []
purified_fidelity_witness = []
purified_fidelity = []
true_unitary = ansatz(parameters)
nocc = molecule.n_electrons // 2
nvirt = molecule.n_orbitals - nocc
initial_fock_state = [1] * nocc + [0] * nvirt
for _ in range(1000):  # 1000 repetitions of the measurement
    new_opdm = resample_opdm(opdm, var_dict)
    raw_energies.append(opdm_func.energy_from_opdm(new_opdm))
    raw_fidelity_witness.append(
        fidelity_witness(target_unitary=true_unitary,
                         omega=initial_fock_state,
                         measured_opdm=new_opdm)
    )
    # fix positivity and trace of sampled 1-RDM if strictly outside
    # feasible set
    w, v = np.linalg.eigh(new_opdm)
    if len(np.where(w < 0)[0]) > 0:
        new_opdm = fixed_trace_positive_projection(new_opdm, nocc)

    new_opdm_pure = mcweeny_purification(new_opdm)
    purified_eneriges.append(opdm_func.energy_from_opdm(new_opdm_pure))
    purified_fidelity_witness.append(
        fidelity_witness(target_unitary=true_unitary,
                         omega=initial_fock_state,
                         measured_opdm=new_opdm_pure)
    )
    purified_fidelity.append(
        fidelity(target_unitary=true_unitary,
                 measured_opdm=new_opdm_pure)
    )
print('\n\n\n\n')
print("Canonical Hartree-Fock energy ", molecule.hf_energy)
print("True energy ", energy(parameters))
print("Raw energy ", opdm_func.energy_from_opdm(opdm),
      "+- ", np.std(raw_energies))
print("Raw fidelity witness ", np.mean(raw_fidelity_witness).real,
      "+- ", np.std(raw_fidelity_witness))
print("purified energy ", opdm_func.energy_from_opdm(opdm_pure),
      "+- ", np.std(purified_eneriges))
print("Purified fidelity witness ", np.mean(purified_fidelity_witness).real,
      "+- ", np.std(purified_fidelity_witness))
print("Purified fidelity ", np.mean(purified_fidelity).real,
      "+- ", np.std(purified_fidelity))
"""
This should print out the various energies estimated from the 1-RDM along with error bars. 
Generated from resampling the 1-RDM based on the estimated covariance.
"""
##
"""
Optimization:
We use the sampling functionality to variationally relax the parameters of my ansatz such that the energy is decreased.
For this we will need the augmented Hessian optimizer
The optimizerer code we have takes: rhf_objective object, initial parameters, 
a function that takes a n x n unitary and returns an opdm maximum iterations, 
hassian_update which indicates how much of the hessian to use rtol which is the gradient stopping condition.
A natural thing that we will want to save is the variance dictionary of the non-purified 1-RDM. 
This is accomplished by wrapping the 1-RDM estimation code in another object that keeps track of the variance dictionaries.
"""
from recirq.hfvqe.mfopt import moving_frame_augmented_hessian_optimizer
from recirq.hfvqe.opdm_functionals import RDMGenerator
import matplotlib.pyplot as plt
rdm_generator = RDMGenerator(opdm_func, purification=True)
opdm_generator = rdm_generator.opdm_generator

result = moving_frame_augmented_hessian_optimizer(
    rhf_objective=rhf_objective,
    initial_parameters=parameters + 1.0E-1,
    opdm_aa_measurement_func=opdm_generator,
    verbose=True, delta=0.03,
    max_iter=20,
    hessian_update='diagonal',
    rtol=0.50E-2)
"""
Each interation prints out a variety of information that the user might find useful. 
Watching energies go down is known to be one of the best forms of entertainment during a shelter-in-place order.

After the optimization we can print the energy as a function of iteration number to see 
close the energy gets to the true minium."""
##
plt.semilogy(range(len(result.func_vals)),
             np.abs(np.array(result.func_vals) - energy(parameters)),
             'C0o-')
plt.xlabel("Optimization Iterations",  fontsize=18)
plt.ylabel(r"$|E  - E^{*}|$", fontsize=18)
plt.tight_layout()
plt.show()