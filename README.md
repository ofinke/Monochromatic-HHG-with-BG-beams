# **Monochromatic HHG with BG beams**
This repository represent supplementary code for the article "Monochromatic high-harmonic generation by Bessel-Gauss beam in periodically modulated media"

### Jupyter notebooks
Presented notebooks consist of several example experimental cases:
- Optimised target for H41 in argon with 1030 nm driver for different cases of ionization ($\eta$) and medium defined as gas jets with gaussian pressure profile.
- Optimised target for H23 in argon with 800 nm driver
- Optimised target for H401 in helium with 1600 nm driver

We encurage readers of the article to download these notebooks and design targets for their own experimental conditions.

## Models
#### Analytical
Analytical model is described in the script ```XUV_signal_computation2.py``` with runtime examples accross all the provided jupyter notebooks.

#### Numerical
Numerical model uses script ```harmonic_signal.py``` which holds complete description of the code. Example how to run the numerical model can be found in the ```article_figure.ipynb``` notebook.
