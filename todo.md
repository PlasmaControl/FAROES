# Todos for FAROES science


* Remake the configuration / data modules
  * Separate "models" (particular scientific formulas), "configuration" (inputs to those formulas) and "data" (facts about the world).
  * The data should essentially never change.
  * The shape of the "configuration" may be dependent on the chosen models / model hierarchy

## Plasma
* Some sort of profiles. 'Simple' or 'Advanced' or 'Array'
  * Simple: `(1 - rho^2)^alpha`
  * Advanced: with pedestals, or off-axis peaks for the current density
  * Array: an arbitrary-sized 1D array
* Check existing Current Drive Efficiency calculations / have an option for /exactly/ what Jon Menard used
* Alternate simple bootstrap currents for checking the calculation
* Better bootstrap currents like Lin-Liu's formulation: may require some profiled or 1D array calculations.
* Alternately, precompute bootstrap currents / use a 4-,5-, or 6-D lookup table depending on what temperature, density, shape, and current profiles are used
* Helium dilution: calculate/estimate He confinement times, and calculate a self-consistent density
* Better fusion reaction rates with T **easy**
* L-H and H-L power thresholds
* Better ripple calculations:
  * Could use the PROCESS model to include magnet widths
  * Use the 2D magnet shape somehow
* RF heating & current drive calculations

### Radiation
* Brehmsstrahlung, Synchrotron, and impurity radiation

# Machine engineering
* Option for 'picture-frame' coils
* Better blanket shapes for positive and negative d coils, with options for thinner IB blankets (for STs)

# Pulsed operation
* Plasma resistivity and loop voltage calculations
* Integrate the CS model
* Estimate time between shots
* Availability factors based on shot time
* Need estimate of start-up flux: **hard**

# Costing
* Include costs for how many kAm of tape are needed: this provides a cost for higher current density, where the Generomak model would cost a higher current density magnets as lower because of the lower volume.
* Include planned & unplanned availability estimates
* Include modern interest, inflation, etc. Put applicable costs in 2020 dollars rather than 2010 dollars.
 
# Documentation
* Include documentation of all classes
* Include documentation of models
* Make docs display "Inputs" and "Outputs" in a fancy way like they do for Parameters and Returns
