[![Install & Train](https://github.com/boettiger-lab/gym_fishing/workflows/Install%20&%20Train/badge.svg)](https://github.com/boettiger-lab/gym_fishing/runs/1323937050?check_suite_focus=true)
[![PyPI Version](https://img.shields.io/pypi/v/gym_fishing)](https://pypi.org/project/gym-fishing/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/gym-fishing)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)



This repository provides [OpenAI-gym](https://github.com/openai/gym/) class definitions for the classic fisheries management problem.  See [Creating your own Gym Environments](https://github.com/openai/gym/blob/master/docs/creating-environments.md) for details on how a new gym environment is defined. 


## Installation

Install the latest release from pypi:


```
pip install gym_fishing
```


Or install the current dev version by cloning this repo and running:

```
python setup.py sdist bdist_wheel
pip install -e .
```

## Environments


So far, we have: 

A simple fishing model defined in a continuous state space of fish biomass, with:

- `fishing-v0`: A discrete action space with logistic growth, `n_actions` (default 100), and `quota = action / n_actions * K`.
- `fishing-v1` A continuous action space, `action` = quota.
- `fishing-v2`: A growth model that contains a tipping point. Uses continuous action space.  Set the tipping point parameter, `C`
- `fishing-v4`: Includes parameter uncertainty. 


Model parameters can be configured for each gym, including the choice of `r`, `K`, `sigma` (process noise).
Note that the reward will depend on these parameters, making comparison between models more difficult.
For simple models it is possible to rescale the reward by the known optimal strategy, given the parameters.
Because this approach does not generalize to more complex models where the solution is not known, it is not implemented here.


## Examples

### Known Solutions

MSY, Schaefer 1954: Asymptotic optimal solution

## Theory

The optimal dynamic management solution for the stochastic fisheries model is a "constant escapement" policy, as proven by [Reed 1979](https://doi.org/10.1016/0095-0696(79)90014-7).  For small noise, this corresponds to the same 'bang-bang' solution for the determinstic model, proven by [Clark 1973](https://doi.org/10.1086/260090).  Ignoring discounting, the long-term harvest under the constant escapement solution corresponds to the Maximum Sustainable Yield, or MSY, which is the optimal 'constant mortality' solution (i.e. under the constraint of having to harvest a fixed fraction _F_ of the stock each year), as demonstrated independently by [Schaefer 1954](https://doi.org/10.1007/BF02464432) and [Gordon 1954](https://doi.org/10.1086/257497). 

For a given $f(x)$, The biomass at MSY can trivially be solved for by maximizing the growth function $X_{t+1} = f(X_t)$.  Discretizing the state space, the dynamic optimal harvest can easily be found by stochastic dynamic programming.

Here, we seek to compare the performance of modern RL methods, which make no a-priori assumptions about the stock recruitment function, to this known optimal solution (given the underlying population dynamics).  



