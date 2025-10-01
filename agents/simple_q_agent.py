"""
this is the variant with separate q agents for multiple actions
"""

import numpy as np
import random


class QLearningTable:
    """
    scalar q table model
    generates one action, but can take multiple observation values
    
    as all q_table models fully discrete: a discrete action (e.g. take a 
    step (=1) or don't (=0)) is chosen based on discrete observations (e.g. 
    is too far left (=-1) or right (=1) or at the proper position (=0))

    this model is not gymnasium compliant and not fully compliant with the other agents,
    it is supposed to be wrapped by the below vector version, nonetheless it can run on
    its own
    """
    def __init__(self, dim_states, num_actions, lr=0.1, 
                 gamma=0.9, epsilon=0.1, init_random=False, action_is_scalar=True, obs_is_scalar=True):
        """
        alpha: learning rate in Bellman eq
        gamma: discount factor in Bellman eq
        eps: epxploit/explore factor against random number
             if eps is a list, 0 is starting eps, 
                               1 is decrement multiplier
                               2 is min eps

        lr: learning rate alpha, either [alpha, start_episode, factor] or alpha

        14.5.25 lr introduced
        """
        self.dim_states = dim_states # list of obs
        self.num_actions = num_actions # number of action states
        self.lr = lr
        self.gamma = gamma
        self.eps = epsilon
        self.action_is_scalar = action_is_scalar
        self.obs_is_scalar = obs_is_scalar

        self.q_table = np.zeros(np.concatenate((dim_states, np.array([num_actions]))))
        if init_random:
            self.q_table[:] = np.random.rand(*self.q_table.shape)
        
    @property
    def epsilon(self):
        """
        14.5.25 introduced self.last_eps
        """
        if type(self.eps) == list:
            self.last_eps = self.eps[0]
            self.eps[0] = max(self.eps[2], self.eps[0] * self.eps[1])
        else:
            self.last_eps = self.eps
        return self.last_eps
    

    def adjust_lr(self, curr_episode):
        """
        14.5.25 introduced, similar to epsilon, but scheduling is taken care here
        self.lr: either [alpha, start_episode, factor] or alpha
        """
        if type(self.lr) == list and curr_episode >= self.lr[1]:
            self.lr[0] *= self.lr[2]


    @property
    def alpha(self):
        """
        14.5.25 introduced, similar to epsilon, but scheduling is taken care of outside
        """
        if type(self.lr) == list:
            return self.lr[0]
        else:
            return self.lr

        
    def choose_action(self, state):
        """
        returns action(state)
        either random or from qtable
        here action is a scalar wrapped in a tensor
        5.5.25 actually wrapped it now
        8.5.25 added scalar handling
        """
        if self.obs_is_scalar:
            state = [state]

        if random.uniform(0, 1) < self.epsilon:
            # Explore: select a random action
            action = np.random.randint(0, self.num_actions)
        else:
            # Exploit: select the action with max value (greedy)
            # extract action: state is always one dimension lower than q_table
            # due to the final dimension being action
            q = self.q_table[tuple(state)]
            action = np.argmax(q)
        if self.action_is_scalar:
            return action
        else:
            return np.array([action], dtype=np.int64)
    
    def learn(self, state, action, reward, next_state):
        """
        one step update of qtable
        implements the Bellman eq
        here internally action is a scalar, externally it is a tensor 
        5.5.25 made external action a tensor, introduced to distinguish between external and internal
        8.5.25 added scalar handling
        10.5.25 return abs() added
        """
        if not self.action_is_scalar: #?
            action = action[0]
        if self.obs_is_scalar:
            state = [state]
            next_state = [next_state]

        predict = self.q_table[tuple(state) + (action,)] # state_action_idx]
        target = reward + self.gamma * np.max(self.q_table[tuple(next_state)])
        # Update Q-table with Bellman equation
        # self.q_table[state_action_idx] += self.alpha * (target - predict)  
        delta_q = self.alpha * (target - predict)
        self.q_table[tuple(state) + (action,)] += delta_q
        return abs(target-predict), delta_q

    def train(self, env, num_episodes):
        #*****************************
        """
        full learning cycle over num_epsiodes or until done
        the scalar version is typically not called that way, but wrapped by
        the vector version, therefore it doesn't contain num_episodes and env
        8.5.25 added print for progress
        """
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            steps = 0
            total_td = 0
            delta_q = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _, _ = env.step(action)
                if not done:
                    td, dq = self.learn(state, action, reward, next_state)
                    total_td += td
                    delta_q += dq
                state = next_state
                steps += 1

            if episode % 50 == 0:
                print(f'Episode {episode}: avg TD = {total_td/steps:.4f}, avg delta Q = {delta_q/steps:.4f}, Steps = {steps}')

if __name__ == '__main__':
    from get_action_obs_structure import get_action_obs_structure_for_discrete
    from logger import Train_Logger
else:
    from agents.get_action_obs_structure import get_action_obs_structure_for_discrete
    from agents.logger import Train_Logger
import os 
import pickle

class Simple_Q_Agent:
    """ 
    implements a very simple q learning model, but
    with multiple action dimensions, each of which with its own
    scalar qtable model
    
    17.7.24 AR introduced set_eps()
    5.5.25  torch --> np, proper handling of tensors
    8.5.25 added scalar handling and better gym compatibility
    10.5.25 added save/load and train statistics
    """
    def __init__(self, env, params):
        """
        takes a gym environment and a number of num_episodes learning cycles 
        derives number and dimension of states and obs from the env by way of
        env.action_space and env.observation_space 
        """
        self.env = env
        self.num_episodes = params["num_episodes"]

        action, self.action_is_scalar, obs, self.obs_is_scalar = get_action_obs_structure_for_discrete(env)

        # Get number of actions from gym action space
        self.n_actions = len(action)
        # we assume every state to be of different dimension
        self.n_action_states = action

        # Get the number of observations from the gym env
        self.n_obs = len(obs)
        # as above for states
        self.n_obs_states = obs
       
        # get the q tables
        self.q = []
        for i , asi in enumerate(self.n_action_states):
            self.q.append(QLearningTable(self.n_obs_states, asi, 
                                         params["alpha"], params["gamma"], 
                                         params["epsilon"], params["init_random"],
                                         self.action_is_scalar, self.obs_is_scalar))
  
        self.path = params["path"]
        if self.path is not None:
          if not os.path.exists(self.path):
              print("creating directory ", self.path, " for best model state dicts")
              os.makedirs(self.path)

        # for lr scheduling
        self.count_episodes = 0


    def save(self):
        if self.path is not None:
            qlist = [qi.q_table for qi in self.q]
            print('... Saving Models ......')
            with open(self.path + "\\q_tables.pkl", 'wb') as f:
                pickle.dump(qlist, f, pickle.HIGHEST_PROTOCOL)


    def load(self):
        if self.path is not None:
            print('... Loading models ...')
            with open(self.path + "\\q_tables.pkl", "rb") as f:
                qlist = pickle.load(f)
            for i,qi in enumerate(qlist):
                self.q[i].q_table = qi

    def set_eps(self, new_eps):
        for qi in self.q:
            qi.eps = new_eps


    def choose_action(self, state):
        """
        wrapper around scalar version
        here action is a vector
        5.5.25 adjusted for embedded q to return tensor
        8.5.25 added scalar handling
        """
        action = np.zeros(self.n_actions, dtype=np.int32)
        for i, qi in enumerate(self.q):
            action[i] = qi.choose_action(state) if self.action_is_scalar else qi.choose_action(state)[0]
        if self.action_is_scalar:
            return action[0]
        else:
            return action


    def learn(self, state, action, reward, next_state, done):
        """
        one step
        wrapper around scalar version
        here action is a vector
        5.5.25 adjusted for torch --> np
        8.5.25 added scalar handling
        9.5.25 done included unused for compatibility reasons
        14.5.25 introduced return stats and moved code here
                introduced lr decay, but then learn() alway must be called, even if done
        18.5.25 switched to new stats, removed delta_q()
        """
        if self.action_is_scalar:
            action = [action]

        if done:
            self.count_episodes += 1

        abs_sum_td = 0
        abs_sum_delta_q = 0
        for ai, qi in zip(action, self.q):
            if done:
                qi.adjust_lr(self.count_episodes)
            else:
                td, delta_q = qi.learn(state, ai if self.action_is_scalar else np.array([ai]), reward, next_state)
                abs_sum_td += np.abs(td)
                abs_sum_delta_q += np.abs(delta_q)

        alpha = self.q[0].alpha

        return { "abs_sum_td": abs_sum_td, 
                 "epsilon": self.q[0].last_eps,
                 "abs_sum_delta_q": abs_sum_delta_q, 
                 "alpha":   alpha }
    

    def train(self, verbose = 2):
        """
        full learning cycle over num_epsiodes or until done
        improved for meaningful analysis data and saving best model

        11.5.25: moved delta_q code into its own routine
                 best now based on sum_reward, not sum_reward/steps
                 dto for reporting
        14.5.25  adjusted for stats
        """
        best_reward = -np.inf
        total_abs_sum_td = 0
        count = 0

        if verbose==1:
            print("epi\tsteps\tcurr eps\tepi rwrd\tavg delta td\tepi avg delta q\talpha")
        elif verbose==2:
            logger = Train_Logger(["steps", "curr eps", "epi reward", "cum delta td", "epi avg delta q", "alpha"])

        for i in range(self.num_episodes):
            state, _ = self.env.reset()
            done = False
            steps = 0
            sum_rewards = 0
            sum_abs_delta_q = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                stats = self.learn(state, action, reward, next_state, done)
                
                total_abs_sum_td += stats["abs_sum_td"]
                epsilon = stats["epsilon"]
                sum_abs_delta_q += stats["abs_sum_delta_q"] 
                alpha = stats["alpha"]
            
                state = next_state
                steps += 1
                count += 1
                sum_rewards += reward

            if verbose==1:
                print(f"{i:5}\t{steps:5}\t{epsilon:>11.4e}\t{float(sum_rewards):>11.4e}" \
                      f"\t{total_abs_sum_td/count:1.4e}\t{sum_abs_delta_q/steps:1.4e}\t{alpha}")
            elif verbose==2:
                logger.push_show([steps, self.q[0].epsilon, sum_rewards, total_abs_sum_td/count, \
                                  sum_abs_delta_q/steps, alpha])

            if sum_rewards > best_reward:
                self.save()
                best_reward = sum_rewards
            
        if verbose==2:
            logger.push_show(data=None, final_step=True)


if __name__ == "__main__":
    from global_agent_params import q_agent_params as agent_params
    agent_params['num_episodes'] = 300
    from simple_environment import SimpleEnvironment, NotSoSimpleEnvironment

    # inner system:
    env = SimpleEnvironment(5)
    # test the inner system:
    agent = QLearningTable([env.observation_space.n], env.action_space.n)
    agent.train(env=env, num_episodes=agent_params['num_episodes'])
    print(agent.q_table)
    for i,qi in enumerate(agent.q_table):
        print(i, np.argmax(qi))
 
    # outer system:
    agent = Simple_Q_Agent(env, agent_params)
    agent.train()
    print(agent.q[0].q_table)    
    for i,qi in enumerate(agent.q[0].q_table):
        print(i, np.argmax(qi))
    
    env = NotSoSimpleEnvironment([3**i for i in range(1,4)])
    agent = Simple_Q_Agent(env, agent_params)
    agent.train()
