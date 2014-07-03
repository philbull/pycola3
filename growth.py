########################################################################
########################################################################
#    Copyright (c) 2013,2014       Svetlin Tassev
#                       Princeton University,Harvard University
#
#   This file is part of pyCOLA.
#
#   pyCOLA is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   pyCOLA is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with pyCOLA.  If not, see <http://www.gnu.org/licenses/>.
#
########################################################################
########################################################################



##################################################
##################################################
##################################################
##################################################

#   This solves for the linear growth factor and its derivative
#   The notation follows eq. (A.1,A.2) of arXiv:1301.0322


def _q_factor(a,Om,Ol):  # this is Q(a)
    from math import sqrt
    return a**3*sqrt(Om/a**3 + Ol + (1.0-Om-Ol)/a**2)


def _growth_derivs(f,a,Om,Ol):
    d=f[0]
    y=f[1]
    q=_q_factor(a,Om,Ol)
    dDda=y/q 
    dyda=1.5*Om*a*d/q
    return [dDda,dyda]


def growth_factor_solution(Om,Ol): # returns a,D(a),T[D] = Q(a)D'(a) where Q(a) = _q_factor()
    """
    :math:`\\vspace{-1mm}`
    
    Calculate the linear growth factor evolution for a given cosmology.
    
    **Arguments**:
    
    * ``Om`` -- a float, giving the matter density, :math:`\Omega_m`, today.
    
    * ``Ol`` -- a float, giving the vacuum density, :math:`\Omega_\Lambda`, today.
    
    **Return**:
    
    * an :math:`n\\times 3` array containing
      :math:`[a_i,D(a_i),T[D](a_i)]` for :math:`i=1\dots n` in 
      order of increasing scale factor :math:`a`. Here, the linear growth 
      factor is given by :math:`D(a)`, while :math:`T[D](a)` is given by 
      equation (A.1) of [temporalCOLA]_. These arrays can be further interpolated if needed.
      
    
    
    """
    from scipy import integrate
    from numpy import append,array
    a=[float(x)/1000. for x in range(1,1101)] # go to slightly later times so that no problems with interpolation occur
    amin=a[0]
    v=integrate.odeint(_growth_derivs, [amin,_q_factor(amin,Om,Ol)], a,args=(Om,Ol))
    
    v/=v[1001,0] # divide by growth factor at a=1. index=1000+1 depend on the a=[...] above
    
    return append(array([a]).transpose(),v,1)


def growth_2lpt(a,d,Om):
    """
    :math:`\\vspace{-1mm}`
    
    Return the second order growth factor for a given scale factor and 
    respective linear growth factor. One needs to precompute the latter. 
    :math:`\Lambda\mathrm{CDM}` is assumed for this calculation.
    
    **Arguments**:

    * ``a`` -- a float, giving the scale factor.
    
    * ``d`` -- a float, giving the linear growth factor at `a`.

    * ``Om`` -- a float, giving the matter density, :math:`\Omega_m`, today.
    
    **Return**:
    
    * A float giving the second order growth factor.
    
    **Example**::
    
        >>> Om=0.275
        >>> Ol=1.0-Om
        >>> from growth import growth_factor_solution,growth_2lpt
        >>> darr=growth_factor_solution(Om,Ol)
        >>> from scipy import interpolate
        >>> growth = interpolate.interp1d(darr[:,0].tolist(),darr[:,1].tolist(),
        ...                               kind='linear')
        >>> a=0.3
        >>> d=growth(a)
        >>> growth_2lpt(a,d,Om)/d/d
        0.99148941733187124
    
    """
    #omega=Om/(Om+(1.0-Om)*a*a*a)
    omega=1.0/(Om+(1.0-Om)*a*a*a) #normalized
    return d*d*omega**(-1./143.)

def d_growth2(a,d,Om,Ol):
    """
    :math:`\\vspace{-1mm}`
    
    Return :math:`T[D_2](a)` for the second order growth factor, :math:`D_2`, for a given scale factor and 
    respective linear growth factor. Here :math:`T` is given by 
    equation (A.1) of [temporalCOLA]_. One needs to precompute the linear growth factor. 
    :math:`\Lambda\mathrm{CDM}` is assumed for this calculation.
    
    **Arguments**:

    * ``a`` -- a float, giving the scale factor.
    
    * ``d`` -- a float, giving the linear growth factor at `a`.

    * ``Om`` -- a float, giving the matter density, :math:`\Omega_m`, today.
    
    **Return**:
    
    * A float giving :math:`T[D_2](a)`.
    
    """
    
    d2= growth_2lpt(a,d,Om)
    omega = Om/(Om+(1.0-Om)*a*a*a)
    return _q_factor(a,Om,Ol)*(d2/a)*2.0*omega**(6./11.)



##################################################
##################################################
##################################################
##################################################


#   These are routines calculating the displacement and velocity
#   coefficients for the COLA timestepping.
#   See eq. (A.15) of arXiv:1301.0322


def _u_ansatz(a,nCola):
    return a**nCola
    
def _du_ansatz(a,nCola): # this must be _du_ansatz/da
    return a**(nCola-1.0)*nCola

def _vel_coef(ai,af,ac,nCola,Om,Ol): 
    """
    Needed in implementing (A.15) of http://arxiv.org/pdf/1301.0322v1.pdf 
    """
    coef  =  _u_ansatz(af,nCola)  -  _u_ansatz(ai,nCola)
    q=_q_factor(ac,Om,Ol)
    coef /= q * _du_ansatz(ac,nCola)
    return coef

def _displ_coef_integral(a,nCola,Om,Ol):
    return _u_ansatz(a,nCola) / _q_factor(a,Om,Ol)

def _displ_coef(ai,af,ac,nCola,Om,Ol):
    """
    Needed in implementing (A.15) of http://arxiv.org/pdf/1301.0322v1.pdf 
    """
    from scipy import integrate
    coef  = 1.0/_u_ansatz(ac,nCola)
    coef *= integrate.quad(_displ_coef_integral,float(ai),float(af),args=(nCola,Om,Ol))[0]
    return coef


########################
########################
########################
########################
