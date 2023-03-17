# AGNvar

Python code for constructing time-dependent SEDs using an input driving X-ray light-curve.
If you use this code in your work, please reference Hagen & Done (2023)
https://ui.adsabs.harvard.edu/abs/2023MNRAS.521..251H/abstract


Requirements
----------------
* Python 3 (Tested using Python v.3.9.12)
* numpy (Tested using v.1.21.5)
* astropy (Tested using v.5.1)
* scipy (Tested using v.1.7.3)


Installation
-----------------
Currently the easiest way (read only way..) is to clone the repositroy, and import as you
would any of your other local python modules.
It may become a proper package (that you can pip install) at some point when I have time


Usage
-----------------
For an example on how to run the code see the jupyter notebook: agnvar_example.ipynb
Generally the code is intended to predict the shape of the re-processed continuum assuming
a variable X-ray source irradiating the accretion flow. To this extent it is designed to:

  1. Model the time-averaged SED according to input parameters defining geometry, mass, 
     mass accretion rate, etc. (generally these input parameters are based off typical
     XSPEC models)
     
  2. Evaluate the SED response to an input driving light-curve (always assumed to originate
     in the X-ray corona). This will create a set of time-dependent SEDs
     
  3. In order to give comparisons to real data, the .generate_lightcurve method is designed
     to take the results from the set of time-dependent SEDs and extract the light-curve in
     a given band pass
 
 For more details on the model, see Hagen & Done (2023)
 
 
 Available models
 ------------------
 The currently available SED models are:
 
  * AGNsed_var - This is based off Kubota & Done (2018), and considers a radially stratified flow,
    consisting of an outer standard Shakura-Sunyaev disc (with Novikov-Thorne emissivity), a warm 
    Comptonization region (where the disc fails to thermalize), and a hot Comptonizing corona 
    (powered by the accretion flow).
    
  * AGNbiconTH_var - This extends AGNsed_var to include the presence of a bi-conical outflow.
    This is a simplified model, where the outflow is smooth, and assumed to emitt like a 
    black-body with some characteristic temperature. Hence, for the case of the thermal outflow
    emission, the driving X-ray light-curve only affects the output luminosity; not spectral shape.
  
  * AGNdark_var - This gives a similar picture to e.g Kammoun et al (2021), where we have a standard
    outer accretion disc (Shakura-Sunyaev, with Novikov-Thoren emissivity), and then an inner 'dark'
    disc (extending to the inner modt stable orbit), where all the accretion power is transferred to 
    a lamppost corona giving the X-ray emisison. Hence, within the darkening radius the ONLY emission
    comes from re-processing of X-rays. Note, that although this is a similar picture to Kammoun et al
    (2021), it is not identical; so results will be slightly different (especially as we do not include GR)


Future intentions
------------------
* Write proper documentation (for now if you have any questions email me at scott.hagen@durham.ac.uk)

* Extend the bi-conical model to use radiative transfer results (i.e by reading a grid of, e.g, CLOUDY
  outputs)
  

Hope you find the code useful! 


Citing AGNvar
-------------
If you use AGNvar in you work / publications please cite Hagen & Done (2023).
https://ui.adsabs.harvard.edu/abs/2023MNRAS.521..251H/abstract

Journal reference: Hagen S., Done C., 2023, MNRAS, 521, 251

Bibtex:

```
@ARTICLE{2023MNRAS.521..251H,
       author = {{Hagen}, Scott and {Done}, Chris},
        title = "{Modelling continuum reverberation in active galactic nuclei: a spectral-timing analysis of the ultraviolet variability through X-ray reverberation in Fairall 9}",
      journal = {\mnras},
     keywords = {accretion, accretion discs, black hole physics, galaxies: active, galaxies: individual: Fairall 9, Astrophysics - High Energy Astrophysical Phenomena},
         year = 2023,
        month = may,
       volume = {521},
       number = {1},
        pages = {251-268},
          doi = {10.1093/mnras/stad504},
archivePrefix = {arXiv},
       eprint = {2210.04924},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023MNRAS.521..251H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

 
