#########################
######### 2spsa #########
#########################

rm(list = ls())
gc(reset = T)

if(!require(gym)) install.packages('gym')
if(!require(dplyr)) install.packages('dplyr')
require(gym)
require(dplyr)

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

complete_stack = 0
complete_stack_lim = 5

episode_num = 100
episode_reward_vec = numeric(episode_num)
time_step = 500 
break_option = ifelse(time_step <=200, T, F)

hess_k = diag(1, 4)
for(k in 1:episode_num)
{
  ## Update
  a_k = 1/ (k+1)^alpha_val
  c_k = 1/ (k+1)^gamma_val
  tilde_c_k = 1/ (2*(k+1)^gamma_val)
  
  delta_k = sample(c(-1, 1), size = 4, replace = T)
  tilde_delta_k = sample(c(-1, 1), size = 4, replace = T)
  
  beta_plus = beta_init + c_k * delta_k
  beta_minus = beta_init - c_k * delta_k
  
  beta_plus_reward = reward_fun(client, instance_id, 
                                time_step = time_step, 
                                beta_vec = beta_plus, 
                                break_option = break_option) %>% sum()
  
  beta_minus_reward = reward_fun(client, instance_id, 
                                 time_step = time_step, 
                                 beta_vec = beta_minus, 
                                 break_option = break_option) %>% sum()
  
  beta_plus_tilde = beta_plus + tilde_c_k * tilde_delta_k
  beta_minus_tilde = beta_minus + tilde_c_k * tilde_delta_k
  
  beta_plus_tilde_reward = reward_fun(client, instance_id,
                                      time_step = time_step,
                                      beta_vec = beta_plus_tilde,
                                      break_option = break_option) %>% sum()
  
  
  beta_minus_tilde_reward = reward_fun(client, instance_id,
                                       time_step = time_step,
                                       beta_vec = beta_minus_tilde,
                                       break_option = break_option) %>% sum()
  
  beta_hess_plus = (beta_plus_tilde_reward - beta_plus_reward) / tilde_c_k * tilde_delta_k
  beta_hess_minus = (beta_minus_tilde_reward - beta_minus_reward) / tilde_c_k * tilde_delta_k
  
  delta_G = matrix(beta_hess_plus - beta_hess_minus, ncol = 1)
  tmp_hess_k = (delta_G %*% delta_k + t(delta_G %*% delta_k)) / (4 * c_k)
  hess_k = k / (k + 1) * hess_k + 1 / (k + 1) * tmp_hess_k
  
  hess_k_inv = try(solve(hess_k))
  if(class(hess_k_inv) == 'try-error')
  {
    hess_k_svd = svd(hess_k)
    hess_k_svd$d[hess_k_svd$d <= 0.1] = 0.1
    hess_k_bar = hess_k_svd$u %*% diag(hess_k_svd$d) %*% t(hess_k_svd$v)
    hess_k_inv = solve(hess_k_bar)
  }
  gradient_vec = (beta_plus_reward - beta_minus_reward)/(2 * c_k) * delta_k
  beta_init = beta_init - a_k * hess_k_inv %*% gradient_vec
  
  ## Stopping rule
  episode_reward_val = reward_fun(client, instance_id, 
                                  beta_vec = beta_init, 
                                  time_step = time_step, 
                                  render_display = F, 
                                  break_option = break_option) %>% sum()
  
  episode_reward_vec[k] = episode_reward_val
  cat(k, 'step : ', episode_reward_val, '\n')  
  if(min(episode_reward_vec) == episode_reward_val)
  {
    beta_star = beta_init
  }else{
    beta_init = beta_star
  }
  
  complete_stack = 0
  while(episode_reward_val == -time_step)
  {
    complete_stack = complete_stack + 1
    episode_reward_val = reward_fun(client, instance_id, 
                                    beta_vec = beta_init, 
                                    time_step = time_step, 
                                    render_display = F, 
                                    break_option = break_option) %>% sum()
    cat(complete_stack, 'stack completed\n')
    if(complete_stack == complete_stack_lim) break
  }
  if(complete_stack == complete_stack_lim) break
  
}

reward_fun(client, instance_id, beta_vec = beta_star, time_step = 500, render_display = T, break_option = F)
env_close(client, instance_id)

