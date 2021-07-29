"""
Use pycola to output 2LPT-like snapshots of velocities and particle positions 
over a series of redshift slices.
"""
import numpy as np
import pycola

# Cosmological parameters
omega_m = 0.316
omega_lam = 1. - omega_m
h = 0.67

# Simulation parameters
BOXSIZE = 150. # Mpc?
NGRID = 32
NPART = 32
zi = 14.
zf = 0.
SNAPSHOT_FILE = "snapshot.particles.npz"
SEED = 10 # Define random seed

def particle_mass(z):
    """Calculate the particle mass in units of M_sun."""
    # Mean density in pycola units should be 1 on average
    G = 6.674e-11 # m^3 / kg / s^2
    rhom = 3. * (h * 100.)**2. * omega_m * (1. + z)**3. / (8. * np.pi * G)
    # Convert (km/s/Mpc)^2/(m^3/kg/s^2) => Msun/Mpc^3
    rhom *= 1e6 * 3.086e22 / 1.9886e30
    
    # Get mean particle mass from volume, density, and no. of particles
    pmass = rhom * (float(BOXSIZE) / float(NPART))**3.
    return pmass

# Calculate minimum halo mass
HALO_NPART_MIN = 50
halo_mmin = HALO_NPART_MIN * particle_mass(zf) # Minimum halo mass
print("Min. halo mass: %3.2e Msun" % halo_mmin)

# Calculate ZA initial conditions
sx, sy, sz = pycola.ic.ic_za("camb_matterpower_z0.dat", 
                             boxsize=BOXSIZE, 
                             npart=NPART, 
                             init_seed=SEED)
cellsize = BOXSIZE / float(sx.shape[0])
gridcellsize = BOXSIZE / float(NGRID)
print("cellsize:", cellsize)
print("gridcellsize:", gridcellsize)

# Calculate 2LPT positions
sx2, sy2, sz2 = pycola.ic.ic_2lpt(cellsize, sx, sy, sz, boxsize=BOXSIZE)

# Evolve particles
px, py, pz, vx, vy, vz, *_ = pycola.evolve.evolve(cellsize, 
                                              sx, sy, sz, sx2, sy2, sz2, 
                                              FULL=True, 
                                              gridcellsize=gridcellsize,
                                              ngrid_x=NGRID, 
                                              ngrid_y=NGRID, 
                                              ngrid_z=NGRID,
                                              Om=omega_m, 
                                              Ol=omega_lam, 
                                              a_initial=1./(1.+zi), 
                                              a_final=1./(1.+zf),
                                              save_to_file=True,
                                              file_npz_out=SNAPSHOT_FILE
                                              )
                                              #snapshot_root=SNAPSHOT_ROOT,
                                              #snapshot_zmax=3.)
                                              
print("Finished.")
