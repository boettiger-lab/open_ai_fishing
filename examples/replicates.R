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
Sys.setenv("CUDA_VISIBLE_DEVICES" = "0")

## Set seeds
torch$manual_seed(12345L)
np$random$seed(12345L)

## Initialize our environment
ENV <- "fishing-v0"
env <- gym$make(ENV, n_actions = 100L, sigma = 0.05) # with some process noise

train <- function(env, algo = "SAC"){
  init_model <- sb3[[algo]]
  model <- init_model('MlpPolicy', env, verbose=0L)
  model$learn(total_timesteps=200000L)
}


## Here we go.  Sit tight, this is gonna take a while!
models <- lapply(1:5, function(i) train(env, "SAC"))

## save models
dir.create("results")
lapply(seq_along(models), function(i){
  models[[i]]$save(paste0("examples/results/SAC", i))
})

## Load models
#models <- lapply(paste0("examples/results/SAC", 1:5), sb3$SAC$load)


## simulate models
df <- map_dfr(models, env$simulate, reps=50L, .id = "model")

policy <- map_dfr(models, env$policyfn, reps=500L, .id = "model")
policy$action <- unlist(policy$action)

policy %>% group_by(state, model) %>% summarize(action = mean(action)) %>%
  ggplot(aes(state, action, col=model)) + geom_line()


low <- function(x) quantile(x, probs = seq(0,1,by=0.05))[2]
high <- function(x) quantile(x, probs = seq(0,1,by=0.05))[20]

p2 <- policy %>% 
  group_by(state) %>% 
  summarise(mean_action = mean(action), low = low(action), high = high(action)) 

p2 %>%
  ggplot(aes(state, mean_action)) + 
  geom_line() +
  geom_ribbon(aes(ymin = low, ymax = high), alpha = 0.2)
ggsave("examples/results/SAC-policy.png")

## Evaluate model over n replicates
reward <- 
  map_dfr(models, function(model){
    reward <- sb3$common$evaluation$evaluate_policy(model, env, n_eval_episodes=50L)
    reward <- data.frame(mean = reward[[1]], sd = reward[[2]])
    reward
    },
    .id = "model"
  )
    ##

## repair NAs
as.na <- function(x){
  x[vapply(x, is.null, NA)] <- NA
  as.numeric(x)
}
sims <- df %>%
  as_tibble() %>%
  mutate(state = as.na(state),
         action = as.na(action)/100,
         reward = as.na(reward),
         rep = as.integer(rep))

sims %>% group_by(model, rep) %>% 
  mutate(reward = cumsum(reward)) %>% 
  ungroup() %>%
  pivot_longer(cols = c(state, action, reward)) %>%
  ggplot(aes(time, value, col = rep)) + 
  geom_line(alpha=.8) +
  facet_grid(name~model, scales = "free")


ggsave("examples/results/SAC-sims.png")



