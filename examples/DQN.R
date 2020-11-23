## R dependencies
library(reticulate)
library(tidyverse)


## Python dependencies, via reticulate
gym         <- import ("gym")
gym_fishing <- import("gym_fishing")
sb3         <- import ("stable_baselines3")
os          <- import ("os")
torch       <- import ("torch")
np          <- import ("numpy")


## NB: Most integers must be explicitly typed as such (e.g. `1L` for `1`)

##  Turn CUDA off for reproducibility, if necessary.
Sys.setenv("CUDA_VISIBLE_DEVICES" = "")

## Set seeds
torch$manual_seed(12345L)
np$random$seed(12345L)

## Initialize our environment
ENV <- "fishing-v0"
env <- gym$make(ENV, n_actions = 100, sigma = 0.05) # with some process noise

## We will try training an ensemble of models

model <- sb3$DQN('MlpPolicy', env, verbose=0L)
model$learn(total_timesteps=200000L)



## Here we go.  Sit tight, this is gonna take a while!

## simulate models
df <- env$simulate(model, reps=50L)


## infer a policy function from simulations?

## Now what? Derive a policy function that is averaged over the models?


## Evaluate model over n replicates
reward <- sb3$common$evaluation$evaluate_policy(model, env, n_eval_episodes=50L)
reward <- data.frame(mean = reward[[1]], sd = reward[[2]])
reward

##

## repair NAs
as.na <- function(x){
  x[vapply(x, is.null, NA)] <- NA
  as.numeric(x)
}
sims <- df %>%
  as_tibble() %>%
  mutate(state = as.na(state),
         action = as.na(action),
         reward = as.na(reward),
         rep = as.integer(rep))

## Plot
 sims %>%
  pivot_longer(cols = c(state, action, reward)) %>%
  ggplot(aes(time, value, col = rep)) + 
  geom_point() +
  facet_wrap(~name,ncol=1, scales = "free_y")
ggsave("results/sac.png")
## Evaluate model
p1



