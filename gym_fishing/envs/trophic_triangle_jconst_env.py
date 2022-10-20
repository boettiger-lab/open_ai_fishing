"""
Class inherited from trophicTriangleEnv made for an easy test:

Sets dJ/dt = 0. This considerably simplifies the dynamics: A will now converge towards its stable fixed point s*J/mA =: c.

The A solution is 

c - (c-A0)*exp(-mA*t), with A0 = A(t=0).

This gives a simple test of the env - not very realistic at all, but simple to test the environment: fish all of A after a time 1/mA. After each such harvest, the new A0 of the cycle is actually zero, so the evolution is c(1-exp(-mA*t)).

Each harvest (after the first round) will yield c(1-e) benefits, and divided by unit time (dividing by 1/mA -- i.e. multiplying by mA), we get an increase of total reward which grows as c * mA * (1-e).

Setting J=const, moreover, decouples A from F. That is, now the causal relation is just 

A -> F, and thus F can be safely ignored for the sake of harvesting.
"""

from trophic_triangle_env import trophicTriangleEnv


class trophicTriangleJConstEnv(trophicTriangleEnv):
    def Jdraw(self, fd) -> None:
        return self.fish_population[2]
