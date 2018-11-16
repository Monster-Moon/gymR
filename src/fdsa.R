#########################
######### FDSA ##########
#########################

rm(list = ls())
gc(reset = T)

if(!require(gym)) install.packages('gym')
require(gym)

remote_base = "http://127.0.0.1:5000"
client = create_GymClient(remote_base)
env_id = "CartPole-v0"
instance_id = env_create(client, env_id) 

source('action_reward.R')

set.seed(1)
state_init = env_reset(client, instance_id) 
observation_dim = length(state_init)
beta_star = beta_init = rnorm(observation_dim)

alpha_val = 0.602
gamma_val = 0.101

episode_num = 100
episode_reward_vec = numeric(episode_num)
time_step = 200 ##
break_option = ifelse(time_step <=200, T, F)
for(k in 1:episode_num)
{
  a_k = 1/ (k+1)^alpha_val
  c_k = 1/ (k+1)^gamma_val
  
  episode_reward_val = reward_fun(client, instance_id, 
                                  beta_vec = beta_init, 
                                  time_step = time_step, 
                                  render_display = F,
                                  break_option = break_option) %>% sum()
  
  episode_reward_vec[k] = episode_reward_val
  if(min(episode_reward_vec) == episode_reward_val)
  {
    beta_star = beta_init
  }else{
    beta_init = beta_star
  }
  
  if(episode_reward_val == -time_step) break
  
  plus_minus_mat = c_k * diag(1, nrow = observation_dim, ncol = observation_dim)
  beta_plus_mat = matrix(beta_init, nrow = observation_dim, ncol = observation_dim) + plus_minus_mat
  beta_minus_mat = matrix(beta_init, nrow = observation_dim, ncol = observation_dim) - plus_minus_mat
  
  beta_plus_reward_list = lapply(1:observation_dim, 
                                 function(i) reward_fun(client, instance_id, 
                                                        time_step = time_step, 
                                                        beta_vec = beta_plus_mat[, i], 
                                                        break_option = break_option))
  beta_minus_reward_list = lapply(1:observation_dim, 
                                  function(i) reward_fun(client, instance_id, 
                                                         time_step = time_step, 
                                                         beta_vec = beta_minus_mat[, i], 
                                                         break_option = break_option))
  
  beta_plus_reward = lapply(beta_plus_reward_list, 'sum') %>% unlist()
  beta_minus_reward = lapply(beta_minus_reward_list, 'sum') %>% unlist()
  
  gradient_vec = (beta_plus_reward - beta_minus_reward)/(2 * c_k)
  beta_init = beta_init - a_k * gradient_vec
  
  cat(k, 'step : ', episode_reward_vec[k], '\n')  
}

reward_fun(client, instance_id, beta_vec = beta_star, time_step = 500, render_display = T, break_option = F)
env_close(client, instance_id)

