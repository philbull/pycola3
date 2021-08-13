"""
Use simple COLABox interface to generate a density field realisation.
"""
import numpy as np
import pylab as plt
import pycola3
import os

# Use the CAMB matter power spectrum file in the same directory as this script
dirname = os.path.dirname(__file__)
pspec_file = os.path.join(dirname, "camb_matterpower_z0.dat")

# Initialise COLABox object
# (note that the input matter power spectrum should be evaluated at z=0)
box = pycola3.COLABox(
    ngrid=128,
    nparticles=128,
    box_size=1000.0,
    z_init=10.0,
    z_final=0.0,
    omega_m=0.316,
    h=0.67,
    pspec=pspec_file,
)

# Initialise initial particle displacements
box.generate_initial_conditions(seed=42)

# Evolve the initial displacements until the final redshift
px, py, pz, vx, vy, vz = box.evolve(n_steps=int(1 + box.z_init))

# Deposit the final particle displacements on a regular grid to make a density field
density = box.cic_deposit()
delta = density - 1.0  # this is valid if nparticles == ngrid

# Plot a slice through the density field
plt.matshow(density[:, :, 0], cmap="cividis", vmin=0.0, vmax=5.0)
plt.colorbar()
plt.show()
