"""
This module provides suscpetibilities and polarisabilities for noble gases
(He, Ne, Ar, Kr, Xe) given by the analytic formulae from
https://royalsocietypublishing.org/doi/epdf/10.1098/rspa.1960.0237

The functions are:
    susc_ref, polarisability
    
  

-------
Jan Vabek   
ELI-Beamlines, CELIA, CTU in Prague (FNSPE) (2021 - 2022)   
ELI ERIC (2023)  
"""
import units

susc_Ar = lambda x: (5.547e-4)*(1.0 + (5.15e5)/(x**2) + (4.19e11)/(x**4) + (4.09e17)/(x**6) + (4.32e23)/(x**8))
susc_Kr = lambda x: (8.377e-4)*(1.0 + (6.7e5)/(x**2) + (8.84e11)/(x**4) + (1.49e18)/(x**6) + (2.74e24)/(x**8) + (5.1e30)/(x**10))
susc_He = lambda x: (6.927e-5)*(1.0 + (2.24e5)/(x**2) + (5.94e10)/(x**4) + (1.72e16)/(x**6))
susc_Ne = lambda x: (1.335e-4)*(1.0 + (2.24e5)/(x**2) + (8.09e10)/(x**4) + (3.56e16)/(x**6))
susc_Xe = lambda x: (1.366e-3)*(1.0 + (9.02e5)/(x**2) + (1.81e12)/(x**4) + (4.89e18)/(x**6) + (1.45e25)/(x**8) + (4.34e31)/(x**10))

N_ref_default = 1e5/(units.Boltzmann_constant*(273.15+20.))

susc_funct = {
    'Ar': susc_Ar,
    'Kr': susc_Kr,
    'He': susc_Kr,
    'Ne': susc_Kr,
    'Xe': susc_Kr
}

def getsusc(g,lambd,p_ref=1.,T_ref=273.15+20):
    """
    Returns susceptibility for 1 bar & 20 °C

    Parameters
    ----------
    g : string
        available gases: 'He', 'Ne', 'Ar', 'Kr', 'Xe'  
    lambd : laser wavelength [m]
    p_ref : float, optional, reference pressure in [bar]
    T_ref: float, optional, reference pressure in [Kelvin]


    Returns
    -------
    susceptibility

    Note: The scaling assumes the ideal-gas law.
    """

    # https://royalsocietypublishing.org/doi/epdf/10.1098/rspa.1960.0237
    # provides the result in atmospheres, convert this value to 1 bar & 20 °C
    return (273.15/T_ref)*(p_ref/1.01325)*susc_funct[g](1e10*lambd)

susc_ref = getsusc

def getpol(g,lambd):
    """
    Returns polarisability.
    
    Parameters
    ----------
    g : string
        available gases: 'He', 'Ne', 'Ar', 'Kr', 'Xe'  
    lambd : laser wavelength [m]   

    Returns
    -------
    polarisability
    
    Notes
    -------
    The original paper tabulates susceptibilities at 'standard pressure'.

    """
    # https://royalsocietypublishing.org/doi/epdf/10.1098/rspa.1960.0237
    # provides the result in atmospheres, convert this value to the correct
    # SI polarisability
    N_atm = 101325./(units.Boltzmann_constant*(273.15))
    return susc_funct[g](1e10*lambd)/N_atm

polarisability = getpol