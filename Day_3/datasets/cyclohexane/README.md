This data repository contains initial conformers and molecular dynamics (MD) trajectories for cyclohexane.

MD was computed using Quantum Espresso (https://www.quantum-espresso.org/ V7.0)
and pseudopotentials determined by the standard solid-state pseudopotentials
library (SSSP, https://archive.materialscloud.org/record/2023.65), assuming
molecules were in a dilute gas phase through Martyna-Tuckerman isolation.
We assumed a self-consistent convergence of 1.D-6 and a kinetic energy cutoff
of 60.0D0. Trajectories were run for 4000 steps at 300K using an Andersen thermostat.
Of these, we have here included every tenth frame.

A typical QE input file consisted of the following:

    &CONTROL
        calculation = 'md'
        verbosity = 'high'
        nstep = 4000
    /
    &SYSTEM
	    ibrav = 0,
	    nat = 18,
	    ntyp = 2,
 	    ecutwfc = 60.0D0,
	    nosym=.true.
	    assume_isolated="mt",
    /
    &ELECTRONS
	    conv_thr = 1.D-6,
	    electron_maxstep = 300,
    /
    &IONS
        ion_temperature = 'andersen',
        tempw = 300,
    /
    &CELL
    /
    ATOMIC_SPECIES
    H	1.0079	H.pbe-rrkjus_psl.1.0.0.UPF
    C	12.0107	C.pbe-n-kjpaw_psl.1.0.0.UPF

    ATOMIC_POSITIONS angstrom

followed by the appropriate starting positions and unit cell.

In this folder, we have the results of the MD trajectories stored in the main tree and the initial frames stored in `conformers`.


We also have some featurizations stored, namely the Smooth Overlap of Atomic Positions (SOAP),
computed with [rascaline](https://luthaf.fr/rascaline/latest/) and the following hyperparameters:

``` python
from rascaline.calculators import SoapPowerSpectrum
representation = SoapPowerSpectrum(
    **{
        "atomic_gaussian_width": 0.3,
        "max_angular": 4,
        "max_radial": 6,
        "cutoff": 3.5,
        "radial_basis": {"Gto": {}},
        "cutoff_function": {"ShiftedCosine": {"width": 0.8}},
        "center_atom_weight": 1.0,
    }
)
```

We computed the SOAP parameters for every carbon atom using the following (dirty) code:

``` python
from equistore import Labels
import numpy as np
from skmatter.preprocessing import StandardFlexibleScaler

values = []
for i, _ in enumerate(traj):
    for j in range(6):
        values.append([i,j])

selection = Labels(
    names=["structure", "center"],
    values=np.array(values),
    )

# traj is our set of ase-type frames
rep = representation.compute(traj, selected_samples=selection)

rep = rep.keys_to_samples('species_center')
rep = rep.keys_to_properties(["species_neighbor_1", "species_neighbor_2"])
x = StandardFlexibleScaler(column_wise=False).fit_transform(rep.block().values)
split_soaps = np.array(np.split(x, len(traj)))

# split_soaps is a len(traj) x 6 x n_features array
np.save('cyclohexane_descriptors.npy', split_soaps)
```

We also have a best-match cosine kernel stored under `normalized_kernel.npy`, computed with the (again, dirty) code:

``` python
from sklearn.metrics.pairwise import pairwise_kernels
from skmatter.preprocessing import KernelNormalizer
K_block = pairwise_kernels(x, x, metric='cosine', n_jobs=4)

K_raw = np.array(
    [
        [
            np.mean([K_block[6 * i + m][6 * j + m] for m in range(6)])
            for j in range(len(traj))
        ]
        for i in range(len(traj))
    ]
)

kn = KernelNormalizer(with_trace=False).fit(K_raw)
K = kn.transform(K_raw)

np.save('normalized_kernel.npy', K)
```