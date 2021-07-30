"""
Simple wrapper around pycola functions to provide a simplified interface for
generating simulation boxes.
"""
import numpy as np

from .ic import ic_za, ic_2lpt
from .cic import CICDeposit_3
from .evolve import evolve


class COLABox(object):
    """Class to handle metadata for a COLA simulation box."""

    def __init__(
        self,
        ngrid,
        nparticles,
        box_size,
        z_init,
        z_final=0.0,
        omega_m=0.316,
        h=0.67,
        pspec="camb_matterpower_z0.dat",
    ):
        """Class to handle metadata for a COLA simulation box.

        This class can be used to generate initial conditions, evolve the
        particle field, and deposit it onto a regular grid for example.

        Parameters
        ----------
        ngrid : int
            Number of grid cells in each dimension.
        nparticles : int
            Number of particles in each dimension.
        box_size : float
            Linear dimension of the cubic box, in Mpc/h units.
        z_init : float
            Initial redshift of the COLA evolution.
        z_final : float, optional
            Final redshift of the COLA evolution. Default: 0.
        omega_m : float, optional
            Fractional matter density today. A flat LCDM cosmology is assumed.
            Default: 0.316.
        h : float, optional
            Dimensionless Hubble parameter. Default: 0.67.
        pspec : callable or str, optional
            Either a callable function that return the matter power spectrum as
            a function of k (in h/Mpc units), or the filename of a data file
            containing the matter power spectrum at z=0, in Mpc/h units.
            Default: "camb_matterpower_z0.dat"
        """
        assert isinstance(box_size, (float, np.float)), "box_size must be a float"
        self.box_size = box_size
        self.ngrid = ngrid
        self.nparticles = nparticles
        self.z_init = z_init
        self.z_final = z_final
        self.a_init = 1.0 / (1.0 + z_init)
        self.a_final = 1.0 / (1.0 + z_final)

        # Calculate linear grid sizes
        self.cellsize = self.box_size / self.ngrid  # FIXME
        self.gridcellsize = self.box_size / self.ngrid

        # Cosmological parameters
        self.omega_m = omega_m
        self.omega_lam = 1.0 - omega_m
        self.h = h

        # Matter power spectrum at z=0
        self.pspec = pspec

        # Particle displacement/velocities
        self.sx = self.sy = self.sz = None
        self.sx2 = self.sy2 = self.sz2 = None
        self.px = self.py = self.pz = None
        self.vx = self.vy = self.vz = None

    def particle_mass(self, z):
        """Calculate the particle mass in units of M_sun.

        Parameters
        ----------
        z : float
            Redshift to evaluate the mean matter density at.

        Returns
        -------
        particle_mass : float
            Inferred mass of each particle, in M_sun.
        """
        # Mean density in pycola units, should be 1 on average
        G = 6.674e-11  # m^3 / kg / s^2
        rhom = (
            3.0
            * (self.h * 100.0) ** 2.0
            * self.omega_m
            * (1.0 + z) ** 3.0
            / (8.0 * np.pi * G)
        )

        # Convert (km/s/Mpc)^2/(m^3/kg/s^2) => Msun/Mpc^3
        rhom *= 1e6 * 3.086e22 / 1.9886e30

        # Get mean particle mass from volume, density, and no. of particles
        return rhom * (self.box_size / self.nparticles) ** 3.0

    def generate_initial_conditions(self, seed):
        """
        Generate a set of random initial conditions as 1LPT and 2LPT particle
        displacements. The initial power spectrum is derived from the matter
        power spectrum at z=0.

        Parameters
        ----------
        seed : int
            Random seed to use when generating random initial conditions.

        Returns
        -------
        sx, sy, sz : array_like
            Initial 1LPT particle displacements, from the Zel'dovich approx.
            These data are also stored in ``self.sx, self.sy, self.sz``.

        sx2, sy2, sz2 : array_like
            2LPT particle displacements, derived from the 1LPT displacements.
            These data are also stored in ``self.sx2, self.sy2, self.sz2``.
        """
        self.seed = seed

        # Calculate ZA initial conditions
        sx, sy, sz = ic_za(
            self.pspec,
            boxsize=self.box_size,
            npart=self.nparticles,
            init_seed=self.seed,
        )

        # Calculate 2LPT positions
        sx2, sy2, sz2 = ic_2lpt(self.cellsize, sx, sy, sz, boxsize=self.box_size)

        # Store results
        self.sx, self.sy, self.sz = sx, sy, sz
        self.sx2, self.sy2, self.sz2 = sx2, sy2, sz2
        return sx, sy, sz, sx2, sy2, sz2

    def evolve(
        self,
        s_vec=None,
        s2_vec=None,
        n_steps=15,
        ncola=-2.5,
    ):
        """
        Evolve an initial set of particle displacements from the initial to
        final redshift, using the COLA scheme.

        Parameters
        ----------
        s_vec : tuple of array_like
            If specified as a tuple ``(sx, sy, sz)``, use these 1LPT particle
            displacements for the initial particle positions. Otherwise, the
            stored ``(self.sx, self.sy, self.sz)`` positions are used.
            Default: None (use stored values).
        s2_vec : tuple of array_like
            If specified as a tuple ``(sx2, sy2, sz2)``, use these 2LPT
            particle displacements for the initial particle positions.
            Otherwise, the stored ``(self.sx2, self.sy2, self.sz2)`` positions
            are used. Default: None (use stored values).
        n_steps : int, optional
            The total number of timesteps that the COLA algorithm should make.
            For good results, the number of steps should be near to the inverse
            of the starting scale factor, ``n_steps ~ 1/a_init``. Default: 15.
        ncola : float, optional
            The spectral index for the time-domain COLA evolution. Reasonable
            values lie in the range ``(-4, 3.5)``. Can't be 0 exactly, but can
            be near 0. See Section A.3 of [temporalCOLA]_. Default: -2.5.

        Returns
        -------
        px, py, pz : array_like
            Return final particle displacements at z_final, in Mpc/h. These
            data are also stored in ``(self.px, self.py, self.pz)``.
        vx, vy, vz : array_like
            Return final particle velocities, in Mpc/h units. These data are
            also stored in ``(self.vx, self.vy, self.vz)``.
        """
        # Check initial particle displacements
        if s_vec is not None:
            assert len(s_vec) == 3, "s_vec must be a tuple of arrays (sx, sy, sz)"
            for s in s_vec:
                assert s.dtype == "float32", "s_vec arrays must be float32"
            sx, sy, sz = s_vec
        else:
            sx, sy, sz = self.sx, self.sy, self.sz

        if s2_vec is not None:
            assert len(s2_vec) == 3, "s_vec2 must be a tuple of arrays (sx2, sy2, sz2)"
            for s2 in s2_vec:
                assert s2.dtype == "float32", "s2_vec arrays must be float32"
            sx2, sy2, sz2 = s2_vec
        else:
            sx2, sy2, sz2 = self.sx2, self.sy2, self.sz2

        # Check that fields exist
        if sx is None or sy is None or sz is None:
            raise ValueError(
                "sx, sy, sz fields are not initialised; have you called generate_initial_conditions()?"
            )

        # Evolve particles
        px, py, pz, vx, vy, vz, *_ = evolve(
            self.cellsize,
            sx,
            sy,
            sz,
            sx2,
            sy2,
            sz2,
            covers_full_box=True,
            gridcellsize=self.gridcellsize,
            ngrid_x=self.ngrid,
            ngrid_y=self.ngrid,
            ngrid_z=self.ngrid,
            Om=self.omega_m,
            Ol=self.omega_lam,
            a_initial=self.a_init,
            a_final=self.a_final,
            n_steps=n_steps,
            ncola=ncola,
        )

        # Store results
        self.px, self.py, self.pz = px, py, pz
        self.vx, self.vy, self.vz = vx, vy, vz

        return px, py, pz, vx, vy, vz

    def cic_deposit(self, px=None, py=None, pz=None):
        """
        Deposit particles onto a gridded density field using CIC.

        Parameters
        ----------
        px, py, pz : array_like, optional
            If specified, use these particle displacements when performing the
            CIC operation. Otherwise, use the stored values, ``self.px`` etc.

        Returns
        -------
        density : array_like
            Density field on a regular grid of shape ``(ngrid, ngrid, ngrid)``,
            where ``ngrid = self.ngrid``. If ``self.nparticles == self.ngrid``,
            the output can be interpreted as ``density == (1 + delta)``, where
            ``delta`` is the total matter density contrast.
        """
        # Handle input data
        if px is None:
            px = self.px
        if py is None:
            py = self.py
        if pz is None:
            pz = self.pz

        # Check that fields exist
        if px is None or py is None or pz is None:
            raise ValueError(
                "px, py, pz fields are not initialised; have you called evolve()?"
            )

        # Initialise empty density field grid
        density = np.zeros((self.ngrid, self.ngrid, self.ngrid), dtype="float32")

        # Set extra arguments to default values
        bbox_zoom = np.array([[0, 0], [0, 0], [0, 0]], dtype="int32")
        offset = np.array([0.0, 0.0, 0.0], dtype="float32")
        dummy = 0.0

        # Do CIC deposit to get density field on regular grid
        CICDeposit_3(
            px,
            py,
            pz,
            px,
            py,
            pz,  # dummy vars
            density,
            self.cellsize,
            self.gridcellsize,
            0,
            dummy,
            dummy,
            bbox_zoom,
            offset,
            1,
        )
        return density

    def bounding_box(self):
        """Return bounding box of the cube in format that ``yt`` can use."""
        return np.array(
            [[0.0, self.box_size], [0.0, self.box_size], [0.0, self.box_size]]
        )

    def particle_data(self):
        """
        Return a dictionary containing the particle displacement, velocity, and
        mass data in a format suitable for ``yt``.

        Returns
        -------
        data : dict
            Dictionary with flattened arrays with keys 'particle_position_x',
            'particle_velocity_x', 'particle_mass' etc.
        """
        data = {
            "particle_position_x": self.px.flatten(),
            "particle_position_y": self.py.flatten(),
            "particle_position_z": self.pz.flatten(),
            "particle_velocity_x": self.vx.flatten(),
            "particle_velocity_y": self.vy.flatten(),
            "particle_velocity_z": self.vz.flatten(),
            "particle_mass": self.particle_mass(zf) * np.ones(self.px.shape).flatten(),
        }
        return data
