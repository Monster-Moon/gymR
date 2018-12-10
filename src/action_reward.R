action_fun = function(state, beta_vec)
{
  return(as.numeric(ifelse(unlist(state) %*% beta_vec >= 0, 1, 0)))
}

reward_fun = function(client, instance_id, beta_vec, time_step, render_display = F, break_option = T)
{
  state_init = env_reset(client, instance_id) ## 게임을 초기화, 초기 State 값을 제공  
  action = action_fun(state_init, beta_vec)
  reward_vec = numeric(time_step)
  for(i in 1:time_step)  ## time step
  {
    stat_next = env_step(client, instance_id, action, render = render_display)
    action = action_fun(stat_next$observation, beta_vec)
    reward_vec[i] = -stat_next$reward
    if(stat_next$done & break_option) break
  }
  return(reward_vec)
}