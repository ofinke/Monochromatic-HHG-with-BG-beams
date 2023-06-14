import numpy as np
import sys
import units

# conversion of photons
def ConvertPhoton(x,inp,outp):
  """
  available I/O: 'omegaau', 'omegaSI', 'lambdaau', 'lambdaSI', 'eV', 'T0au',
                 'T0SI', 'Joule'     
  """
  # convert to omega in a.u.
  if (inp == 'omegaau'): omega = x
  elif (inp == 'lambdaSI'): omega = 2.0 * np.pi* units.hbar / (x * units.elmass * units.c_light * units.alpha_fine**2);
  elif (inp == 'lambdaau'): omega = 2.0 * np.pi/(units.alpha_fine*x);
  elif (inp == 'omegaSI'): omega = x * units.TIMEau;
  elif (inp == 'eV'): omega = x * units.elcharge/(units.elmass*units.alpha_fine**2*units.c_light**2);
  elif (inp == 'T0SI'): omega = units.TIMEau*2.0*np.pi/x;
  elif (inp == 'T0au'): omega = 2.0*np.pi/x;
  elif (inp == 'Joule'): omega = x / (units.elmass*units.alpha_fine**2 * units.c_light**2);
  else: sys.exit('Wrong input unit')

  # convert to output
  if (outp == 'omegaau'): return omega;
  elif (outp == 'lambdaSI'): return 2.0*np.pi*units.hbar/(omega*units.elmass*units.c_light*units.alpha_fine**2);
  elif (outp == 'lambdaau'): return 2.0*np.pi/(units.alpha_fine*omega);
  elif (outp == 'omegaSI'): return omega/units.TIMEau;
  elif (outp == 'eV'): return omega/(units.elcharge/(units.elmass*units.alpha_fine**2 * units.c_light**2));
  elif (outp == 'T0SI'): return units.TIMEau*2.0*np.pi/omega;
  elif (outp == 'T0au'): return 2.0*np.pi/omega;
  elif (outp == 'Joule'): return omega*(units.elmass*units.alpha_fine**2 * units.c_light**2);
  else: sys.exit('Wrong output unit')
  
