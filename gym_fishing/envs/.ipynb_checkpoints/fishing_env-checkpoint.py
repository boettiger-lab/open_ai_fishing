import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from math import floor

class FishingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.viewer = None
        self.boat_position = [0, 0]
        self.fish_population = np.array([1e5])
        self.K = 1e5
        #self.state = [self.boat_position, self.fish_population]
        self.years_passed = 0
        self.total_harvest = 0
        self.screen_width = 1200
        self.screen_height = 800
        self.boatwidth = 20
        self.boatheight = 10
        self.dockwidth = 20
        self.dockheight = 40
        self.harvest = 0
        self.reward = 0
        self.done = False
        self.r = 0.1
        self.eta = 1e-2
        self.action_space = list(map(lambda x: x / 1e4, list(range(0, 11000, 1000)) ))
    
    def harvest_draw(self, quota):
        """
        Select a value to harvest at each time step.
        """
        self.harvest = min(self.fish_population * (self.fish_population >= 0), quota)
    
    def population_draw(self):
        """
        Select a value for population to grow or decrease at each time step.
        """
        self.fish_population -= self.harvest
        self.fish_population = max(0, floor(self.fish_population + self.r * self.fish_population \
                                * (1 - self.fish_population / self.K) \
                                + self.fish_population * self.eta * np.random.normal(0,1)))
        self.fish_population = np.array([self.fish_population])
        
    
    def step(self, action):
        action = floor(1e4 * action)
        assert action % 1 == 0 and action >= 0, "%r (%s) invalid"%(action, type(action))
        
        if self.fish_population == 0:
            return self.fish_population, self.reward, True, {}
        harvest_draw(self, action)
        population_draw(self)
        self.reward += self.harvest / self.K
        if self.fish_population < 2e4:
            self.reward -= 10
        self.years_passed += 1
        if self.years_passed == 100:
            self.done = True
        
        return self.fish_population, self.reward, self.done, {}
            
        
    def random_reset(self):
        self.fish_population = int(np.random.randint(1, 10**5))
        self.reward = 0
        self.done = False
        self.years_passed = 0
        return np.array([self.fish_population]), self.reward, self.done, {}
    
    
    def reset(self, level = 1e5):
        self.fish_population = int(level)
        self.reward = 0
        self.done = False
        self.years_passed = 0
        return np.array([self.fish_population]), self.reward, self.done, {}
  
    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.screen_width, \
                                           self.screen_height)
            left, right, top, bottom = -self.boatwidth/2, \
                                        self.boatwidth/2, \
                                        self.boatheight/2, \
                                        -self.boatheight/2
            boat = rendering.FilledPolygon([(left, bottom), \
                                            (left, top), (right, top), \
                                            (right, bottom)])
            self.boattrans = rendering.Transform()
            boat.add_attr(self.boattrans)
            self.viewer.add_geom(boat)
            left, right, top, bottom = -self.dockwidth/2, self.dockwidth/2,\
                                        self.dockheight/2, -self.dockheight/2
            dock = rendering.FilledPolygon([(left, bottom), (left, top),\
                                            (right, top), (right, bottom)])
            dock.set_color(.8,.6,.4)
            self.docktrans = rendering.Transform()
            dock.add_attr(self.docktrans)
            self.viewer.add_geom(dock)
            self.track = rendering.Line((0, self.screen_height/2), \
                                        (self. screen_width,self.screen_height/2))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)
        
        
        position = self.state[0]
        boatx = position[0] + self.screen_width/2
        self.boattrans.set_translation(boatx, 50)
        self.docktrans.set_translation(self.screen_width/2, self.dockheight/2)
        
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
  
    def close(self):
        pass
