# Modelling Solar Orbiter Dust Detection Rates in Inner Heliosphere as a Poisson Process

This repository contains the code used to perform analyses described in the article, freely accessible at https://doi.org/10.1051/0004-6361/202245165

### Article Abstract
*Context.* Solar Orbiter provides dust detection capability in inner heliosphere, but estimating physical properties of detected dust from
the collected data is far from straightforward.

*Aims.* First, a physical model for dust collection considering a Poisson process is formulated. Second, it is shown that dust on
hyperbolic orbits is responsible for the majority of dust detections with Solar Orbiter’s Radio and Plasma Waves (SolO/RPW). Third,
the model for dust counts is fitted to SolO/RPW data and parameters of the dust are inferred, namely: radial velocity, hyperbolic
meteoroids predominance, and solar radiation pressure to gravity ratio as well as uncertainties of these.

*Methods.* Non-parametric model fitting is used to get the difference between inbound and outbound detection rate and dust radial
velocity is thus estimated. A hierarchical Bayesian model is formulated and applied to available SolO/RPW data. The model uses the
methodology of Integrated Nested Laplace Approximation, estimating parameters of dust and their uncertainties.

*Results.* SolO/RPW dust observations can be modelled as a Poisson process in a Bayesian framework and observations up to this date
are consistent with the hyperbolic dust model with an additional background component. Analysis suggests a radial velocity of the
hyperbolic component around (63 ± 7) km/s with the predominance of hyperbolic dust about (78 ± 4) %. The results are consistent
with hyperbolic meteoroids originating between 0.02 AU and 0.1 AU and showing substantial deceleration, which implies effective
solar radiation pressure to gravity ratio & 0.5. The flux of hyperbolic component at 1 AU is found to be (1.1 ± 0.2) × 10−4 m−2s−1 and
the flux of background component at 1 AU is found to be (5.4 ± 1.5) × 10−5 m−2s−1.
