"""
This module implements the computation of a harmonic signal from a periodic
medium driven by a Bessel-Gauss beams using numerical approach. The main function is:   
    'hh_signal_dk'  

-------
The corresponding theory is described in 'XXX'.

-------
The fixed parameters of the scheme are specified by

parameters = {
              'XUV_table_type_dispersion': 'Henke' or 'NIST',
              'XUV_table_type_absorption': 'Henke' or 'NIST',
              'gas_type': available gases 'He', 'Ne', 'Ar', 'Kr', 'Xe'  
              'omegaSI': fundamental laser frequency [rad/s],
              'Aq' : Amplitude of the harmonic response}

-------
Ondrej Finke - ELI ERIC (2023)
"""

# imports
# import medium as md
import numpy as np
import scipy.integrate as spi
import mynumerics as mn
from scipy.signal import square
import IR_refractive_index as ir
import XUV_refractive_index as xuv
import units

p_ref = 1. # reference pressure for scaling [bar]
epsilonZero = 8.854187817e-12 #[F/m]
electronCharge = units.elcharge #[C] 
electronMass = units.elmass #[kg]
speedLight = units.c_light #[m/s]
numberDensity = 1e5/(units.Boltzmann_constant*(273.15+20.)) # [m^-3] amount of particles of ideal gas in m^3, 1 mbar and 20°C

# plasma frequency
def plasma(wavelength, eta):
    # SUSCEPTIBILITY OF PLASMA
    # value calculated as -(wp/w0)
    pFreq = (eta*numberDensity*electronCharge**2)/(electronMass*epsilonZero)
    lFreq = ((2*np.pi*speedLight)/wavelength)
    return -(pFreq/(lFreq**2))

# _____________________________________________________________________________________
# CALCULATE K VECTORS

def hhg_k(pressure, gas, wavelength, eta, parameters):
    """
    computes phase of harmonic wave including absorption

    Parameters
    ----------
    pressure : array-like, pressure modulation across z axis [bar]  
    gas: gas type: 'He', 'Ne', 'Ar', 'Kr', 'Xe'
    wavelength: [m]
    eta: ionization degree [%]
    parameters : dict (see documentation of the module)  

    Returns
    -------
    phase of harmonic wave [-]
    """

    add_med = np.zeros(len(pressure), dtype="complex_")
    # add_dip = np.zeros(len(z)) # vector for dipole moment

    gas_type = parameters['gas_type']
    XUV_table_type_dispersion = parameters['XUV_table_type_dispersion']
    XUV_table_type_absorption = parameters['XUV_table_type_absorption']
    gas_table_dis = gas_type+'_'+XUV_table_type_dispersion
    gas_table_abs = gas_type+'_'+XUV_table_type_absorption

    susNe = xuv.susc_ref(mn.ConvertPhoton(wavelength,'lambdaSI','omegaSI'), gas_table_dis)
    absor = xuv.beta_factor_ref(mn.ConvertPhoton(wavelength,'lambdaSI','omegaSI'), gas_table_abs)*1j*2

    for i, val in enumerate(pressure):
        # k additions by medium:
        # with (1-eta)
        # add_med[i] = (2*np.pi)/(wavelength)*np.sqrt((val/p_ref)*((1-eta)*(susNe+absor))+1)
        # without
        add_med[i] = (2*np.pi)/(wavelength)*np.sqrt((val/p_ref)*(susNe+absor)+1)


    return add_med

def ir_k(pressure, gas, wavelength, eta, geo):
    """
    computes phase of driver beam wave

    Parameters
    ----------
    pressure : array-like, pressure modulation across z axis [bar]  
    gas: gas type: 'He', 'Ne', 'Ar', 'Kr', 'Xe'
    wavelength: [m]
    eta: ionization degree [%]
    geo: number-like, geometrical phase [-]
    parameters : dict (see documentation of the module)  

    Returns
    -------
    phase of driver laser [-]
    """
    # define empty vectors
    add_med = np.zeros(len(pressure))
    add_gou = np.zeros(len(pressure))

    susNe = ir.getsusc(gas, wavelength)
    susPl = plasma(wavelength, eta)

    for i, val in enumerate(pressure):
        # add_med[i] = (2*np.pi)/(wavelength)*np.sqrt((val/p_ref)*(susPl+(1-eta)*(susNe))+1)
        add_med[i] = (2*np.pi)/(wavelength)*np.sqrt((val/p_ref)*(susPl+(susNe))+1)
        add_gou[i] = geo; 
    
    return add_med - add_gou


def dk_cumulation(zaxis, pressure, gas, wavelength, order, eta, geo, parameters):
    """
    computes accumulation of phase mismatch accross the z

    Parameters
    ----------
    zaxis: array-like, z axis [m]
    pressure : array-like, pressure modulation across z axis [bar]  
    gas: gas type: 'He', 'Ne', 'Ar', 'Kr', 'Xe'
    wavelength: laser wavelength [m]
    order: harmonic order [.]
    eta: ionization degree [%]
    geo: number-like, geometrical phase [-]
    parameters : dict (see documentation of the module)  

    Returns
    -------
    phase mismatch accumulation [-]
    """
    kFund = ir_k(pressure, gas, wavelength, eta, geo) 
    kHarm = hhg_k(pressure, gas, wavelength/order, eta, parameters) 

    kDel = kHarm - order*kFund
    return spi.cumulative_trapezoid(kDel, zaxis, initial=0)

# _____________________________________________________________________________________
# NUMERICAL MODEL

# def hh_signal(zaxis, pressure, gas, eta, wavelength, order, geo, parameters):
#     # SIGNAL OF THE FIELD AT EVERY Z POINT
#     # Calculate the phase (accumulation of k)
#     kFundCum = spi.cumulative_trapezoid(ir_k(pressure, gas, wavelength, eta, geo), zaxis, initial=0)
#     kHarmCum = spi.cumulative_trapezoid(hhg_k(pressure, gas, wavelength/order, eta), zaxis, initial=0)

#     phase = np.zeros(zaxis.shape[0], dtype = "complex_"); field = np.zeros(zaxis.shape[0], dtype = "complex_")
#     # Calculate the fields
#     for i, val in enumerate(kFundCum):
#         # phase of each harmonic wave
#         phase[i] = order*(val)+(kHarmCum[-1]-kHarmCum[i])
#         # field of each harmonic wave
#         field[i] = pressure[i]*np.exp(1j*phase[i])

#     # Calculate the field and convert it to intensity
#     return np.abs(spi.cumulative_trapezoid(field, zaxis, initial=0))**2


def hh_signal_dk(zaxis, pressure, gas, eta, wavelength, order, geo, parameters):
    """
    computes harmonic signal at every point z across z axis. Returns intensity

    Parameters
    ----------
    zaxis: array-like, z axis [m]
    pressure : array-like, pressure modulation across z axis [bar]  
    gas: gas type: 'He', 'Ne', 'Ar', 'Kr', 'Xe'
    wavelength: laser wavelength [m]
    order: harmonic order [.]
    eta: ionization degree [%]
    geo: number-like, geometrical phase [-]
    parameters : dict (see documentation of the module)  

    Returns
    -------
    Returns harmonic intensity [-]
    """
    kDel = dk_cumulation(zaxis, pressure, gas, wavelength, order, eta, geo, parameters)

    field = np.zeros(zaxis.shape[0], dtype = "complex_")
    # Calculate the fields
    for i, val in enumerate(kDel):
        # field of each harmonic wave
        field[i] = pressure[i]*np.exp(1j*val)

    # Calculate the field and convert it to intensity
    return np.abs(spi.cumulative_trapezoid(field, zaxis, initial=0))**2

# _____________________________________________________________________________________
# OPTIMAL GEOMETRICAL PHASE

def p_one(tbt, period, order):
    return tbt/(period*order)

def p_two(tbt, period, wavelength, order, pressure, gas, ratio):
    harm = (np.pi/wavelength)*((1/(2*ratio)*pressure)/p_ref*xuv.susc_ref(mn.ConvertPhoton(wavelength/order,'lambdaSI','omegaSI'), gas))
    fund = (np.pi/wavelength)*((1/(2*ratio)*pressure)/p_ref*ir.getsusc(gas, wavelength))
    return p_one(tbt, period, order) - harm + fund

def fullmodel(tbt, period, wavelength, order, pressure, gas, eta, parameters):
    """
    computes optimal geometrical phase in a single period where xi=1 (half medium, half empty space)

    Parameters
    ----------
    tvt: required phase mismatch accumulation achieved in a single period (2pi generaly)
    period: number-like [m] length of single period
    wavelength: laser wavelength [m]
    order: harmonic order [-]
    pressure : number-like, average pressure in a single period [bar]
    gas: gas type: 'He', 'Ne', 'Ar', 'Kr', 'Xe'
    eta: ionization degree [%]
    parameters : dict (see documentation of the module)  

    Returns
    -------
    Returns geometrical phase [-]
    """
    gas_type = parameters['gas_type']
    XUV_table_type_dispersion = parameters['XUV_table_type_dispersion']
    gas_table = gas_type+'_'+XUV_table_type_dispersion

    # with (1-eta)
    # res = p_one(tbt, period, order) + ((pressure)/(p_ref))*(np.pi/wavelength)*(plasma(wavelength, eta) - plasma(wavelength/order, eta) + (1-eta)*(ir.getsusc(gas, wavelength)-xuv.susc_ref(mn.ConvertPhoton(wavelength/order,'lambdaSI','omegaSI'), gas_table)))
    # without
    res = p_one(tbt, period, order) + ((pressure)/(p_ref))*(np.pi/wavelength)*(plasma(wavelength, eta) - plasma(wavelength/order, eta) + (ir.getsusc(gas, wavelength)-xuv.susc_ref(mn.ConvertPhoton(wavelength/order,'lambdaSI','omegaSI'), gas_table)))

    return res

# _____________________________________________________________________________________
# PRESSURE MODULATION

def modsin(centp, amp, period, z):
    """
    returns vector of pressure with sinusoidal modulation

    Parameters
    ----------
    centp: average pressure across the period [bar]
    amp: modulation amplitude, number between 0-1
    period: number-like [m] length of single period
    z: array-like, z-axis for the modulation

    Returns
    -------
    Returns sinusoidal pressure modulation [bar]
    """
    return amp*centp*np.sin((2*np.pi/period)*z) + centp

def modconst(centp, z):
    """
    returns vector with constant pressure

    Parameters
    ----------
    centp: average pressure across the period [bar]
    z: array-like, z-axis for the modulation

    Returns
    -------
    Returns vector with constant pressure [bar]
    """
    return centp*np.ones(z.shape[0])

def modstep(centp, period, z, ratio=0.5):
    """
    returns vector with step modulation, where average pressure is kept same when changing ratio of medium and empty space

    Parameters
    ----------
    centp: average pressure across the period [bar]
    period: number-like [m] length of single period
    z: array-like, z-axis for the modulation
    ratio: def=0.5, ratio of vacuum and medium number 0-1, higher equals to longer medium

    Returns
    -------
    Returns vector with step modulation while keeping average pressure same [bar]
    """
    return (0.5*centp*square((2*np.pi/period)*z, duty=ratio) + 0.5*centp)/ratio

def modstep2(pressure, period, z, ratio=0.5):
    """
    returns vector with step modulation, where medium has always same pressure

    Parameters
    ----------
    pressure: pressure in a medium [bar]
    period: number-like [m] length of single period
    z: array-like, z-axis for the modulation
    ratio: def=0.5, ratio of vacuum and medium number 0-1, higher equals to longer medium

    Returns
    -------
    Returns vector with step modulation, where medium has always same pressure [bar]
    """
    return 0.5*pressure*square((2*np.pi/period)*z, duty=ratio)+ 0.5*pressure

def modgauss(centp, period, z, sigma):
    """
    returns vector with gauss train modulation

    Parameters
    ----------
    pressure: average pressure in a medium [bar]
    period: number-like [m] length of single period
    z: array-like, z-axis for the modulation
    sigma: sigma of gaussian pulse

    Returns
    -------
    Returns vector with train of gaussian pulses, where medium has average pressure centp [bar]
    """
    n_pulses = int(np.ceil(z[-1] / period))  
    train = np.zeros_like(z)  
  
    for i in range(n_pulses):  
        pulse_centp = period * (i + 0.5)  
        train += np.exp(-(z - pulse_centp)**2 / (2 * sigma**2))  
  
    # scale the result to ensure average pressure
    scale_factor = centp/np.mean(train)
    train = train * scale_factor

    return train


## back compatibility, kept for instant
hh_signal = hh_signal_dk