# Digital Image Processing (EE60062) Term Project

This repository contains code for paper implementation of LIME: Low-Light Image Enhancement via Illumination Map Estimation. The paper link can be found [here](https://ieeexplore.ieee.org/document/7782813). 


### Group members:
- Kolla Ananta Raj (19EE38009)
- Manav Nitin Kapadnis (19EE38010)
- Modi Omkar (19EE38015

# Installation

Clone the repository :

```
git clone https://github.com/rajanant49/LIME
```

Install dependencies :

```
pip3 install -r requirements.txt
```

# How to Run:

```
python3 enhance.py -f <path to image file>
    		   -it <number of iterations to converge>
    		   -alph <the alpha balancing parameter>
    		   -rho <scaling factor for miu>
		   -mu <positive penalty scalar>
		   -gm <the gamma correction factor>
    		   -st <Weighting Strategy 1, 2, 3>
    		   -eps <constant to avoid computation instability>
    		   -s <Spatial standard deviation for spatial affinity based Gaussian weights>
```

Example:

```
python3 enhance.py -st 3 -f './pics/2.jpg'
```
