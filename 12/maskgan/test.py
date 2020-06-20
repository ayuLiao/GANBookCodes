def calculate_reinforce_objective(hparams,
                                  log_probs,
                                  dis_predictions,
                                  present,
                                  estimated_values=None):
  # 生成器最终的目标函数
  final_gen_objective = 0.
  # 折扣因子 gamma
  gamma = hparams.rl_discount_rate
  eps = 1e-7

  # 生成器奖励的log对象
  eps = tf.constant(1e-7, tf.float32)
  dis_predictions = tf.nn.sigmoid(dis_predictions)
  rewards = tf.log(dis_predictions + eps)

  # 只作用在缺失的元素上，具体的做法依旧使用使用tf.where()方法来进行mask操作
  zeros = tf.zeros_like(present, dtype=tf.float32)
  log_probs = tf.where(present, zeros, log_probs)
  # 奖励
  rewards = tf.where(present, zeros, rewards)


  rewards_list = tf.unstack(rewards, axis=1)
  log_probs_list = tf.unstack(log_probs, axis=1)
  missing = 1. - tf.cast(present, tf.float32)
  missing_list = tf.unstack(missing, axis=1)

  # 将所有时间节点的奖励累积
  cumulative_rewards = []
  for t in xrange(FLAGS.sequence_length):
    cum_value = tf.zeros(shape=[FLAGS.batch_size])
    for s in xrange(t, FLAGS.sequence_length):
      cum_value += missing_list[s] * np.power(gamma, (s - t)) * rewards_list[s]
    cumulative_rewards.append(cum_value)
  cumulative_rewards = tf.stack(cumulative_rewards, axis=1)

  if FLAGS.baseline_method == 'critic':

    # critic loss，只在 missing tokens 上计算
    critic_loss = create_critic_loss(cumulative_rewards, estimated_values,
                                     present)

    # 通过 estimated_values(critic 产生的结果) 来得到 baselines
    baselines = tf.unstack(estimated_values, axis=1)

    # 计算
    advantages = []
    for t in xrange(FLAGS.sequence_length):
      log_probability = log_probs_list[t]
      cum_advantage = tf.zeros(shape=[FLAGS.batch_size])

      for s in xrange(t, FLAGS.sequence_length):
        cum_advantage += missing_list[s] * np.power(gamma,
                                                    (s - t)) * rewards_list[s]
      # (R_t - b_t)
      cum_advantage -= baselines[t]
      # 裁剪 advantages.
      cum_advantage = tf.clip_by_value(cum_advantage, -FLAGS.advantage_clipping,
                                       FLAGS.advantage_clipping)
      advantages.append(missing_list[t] * cum_advantage)
      final_gen_objective += tf.multiply(
          log_probability, missing_list[t] * tf.stop_gradient(cum_advantage))

    maintain_averages_op = None
    baselines = tf.stack(baselines, axis=1)
    advantages = tf.stack(advantages, axis=1)

  else:
    raise NotImplementedError

  return [
      final_gen_objective, log_probs, rewards, advantages, baselines,
      maintain_averages_op, critic_loss, cumulative_rewards
  ]