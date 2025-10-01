"""
based on: 2 Produkte - EOQ - 1 Auftrag - Q learning, 5000 episodes instead of 500.py as of
16.7.25 

changelog:
24.9.25 use very simple model, comments see in the model file
"""

from env.very_simple_env import Environment
from agents.simple_q_agent import Simple_Q_Agent as Agent
import numpy as np
from agents.logger import Logger, Train_Logger
import os
import pickle
import matplotlib.pyplot as plt

env = Environment()

PATH = "q_learnings"
agent_params = {
   #
   "num_episodes":    1_000,                 # num sim episodes to conduct
   "alpha":           [1e-5, 100, 0.99],     # learning rate, either number or [current, start_episode_decay, factor]
   "gamma":           0.99,                  # transition rate
   "epsilon":         [1.0, 0.999995, 0.01], # explore/exploit threshold, either number or [start, multiplier, end]
   "init_random":     True,                  # set init internal values to random or to zero
   "path":            PATH,
}

agent = Agent(env, agent_params)
training = True

if training:
    logger = Logger(data_labels = ["inv1", "inv2", "service_level 1", "service_level 2", "action 1", # "action 2", 
                                   "reward"], name=PATH+"\\logger")
    train_logger = Train_Logger(["curr eps", "lr", "cum avg td", "epi avg delta q"], name=PATH+"\\train_logger")

    total_loss = 0
    count = 0

    best_mean_reward = -np.inf

    print("episode\tsteps\tavg reward")

    # debug
    inv1 = []
    inv2 = []
    service_level1 = []
    service_level2 = []
    action1 = []
    # action2 = []
    rewards = []
    epi_delta_q_ratio = 0
    steps = 0

    for e in range(agent.num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            #print(".", end="", flush=True)
            action = agent.choose_action(state)
            next_state, reward, done, _, info = env.step(action)
            stats = agent.learn(state, action, reward, next_state, done)
            state = next_state

            # debug
            inv1.append(env.d1.inv)
            inv2.append(env.d2.inv)
            service_level1.append(env.d1.service_level)
            service_level2.append(env.d2.service_level)
            action1.append(action[0].item())
            # action2.append(action[1].item())
            rewards.append(reward)

            total_loss += stats["abs_sum_td"] 
            epi_delta_q_ratio += stats["abs_sum_delta_q"] 

            steps += 1
            count += 1
        #print("")

        # ?warum mean?
        if np.mean(rewards) > best_mean_reward and len(rewards) == env.num_steps:
            best_mean_reward = np.mean(rewards)
            print(f"saving episode {e}")
            agent.save()
        
        if e % 5 == 0:
            print(f"Episode: {e:6} Reward: {len(rewards):5} Mean Reward: {np.mean(rewards):.3f}")

        #zum Plotten
        logger.push_show(data=[inv1, inv2, service_level1, service_level2, action1, # action2, 
                               rewards], final_step=e==agent.num_episodes-1, this_rewards=rewards)
        train_logger.push_show(data=[stats["epsilon"], stats["alpha"], \
                                    total_loss/count, epi_delta_q_ratio/steps], final_step=e==agent.num_episodes-1)

    os.rename(PATH+"\\q_tables.pkl", PATH+"\\best__q_tables.pkl")
    agent.save()

else:

    """
    plot steps over episodes if desired ...
    """
    if True:
        with open(PATH+"\\logger.pkl", "rb") as f:
            ld = pickle.load(f)
        durations = [len(di[0]) for di in ld]
        N = 20
        durations = np.convolve(durations, np.ones(N)/N, mode='valid')
        plt.plot(durations)
        plt.ylabel("simulation days")
        plt.xlabel("episode")
        plt.title("Learning process")
        plt.show()


    agent.load()
    agent.set_eps(0)

    """
    rerun inference until we get good results, then plot results
    """
    steps = 0
    avg_sl = 0
    count = 0

    env = Environment()
    while steps < env.num_steps or avg_sl < env.min_service_level:
        inv1 = []
        inv2 = []
        service_level1 = []
        service_level2 = []
        action1 = []
        rewards = []
            
        state, _ = env.reset()
        done = False
            
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, info = env.step(action)
            state = next_state

            inv1.append(env.d1.inv)
            inv2.append(env.d2.inv)
            service_level1.append(env.d1.service_level)
            service_level2.append(env.d2.service_level)
            action1.append(action[0].item())
            rewards.append(reward)

        steps = len(inv1)
        avg_sl = (np.array(service_level1).mean() + np.array(service_level2).mean()) / 2
        print("count: ", count, "steps reached: ", steps, ", avg service level: ", avg_sl)
        count += 1

    plt.plot(inv1, label="inv1")
    plt.plot(inv2, label="inv2")
    plt.xlabel("days")
    plt.xlabel("amount")
    plt.title(f"Inference: {count}. run survived sim at avg service level > {env.min_service_level}")
    plt.legend()
    plt.show()


    plt.plot(service_level1, label="service_level 1")
    plt.plot(service_level2, label="service_level 2")
    plt.xlabel("days")
    plt.xlabel("ratio")
    plt.title(f"Inference: {count}. run survived sim at avg service level > {env.min_service_level}")
    plt.legend()
    plt.show()

     # plot fancy final episode inv and action for final X percent
    X=0.2
    # plt.title(f"results for tail {X:.0%} of last episode")
    for j in range(1, 3):
        inv = inv1 if j==1 else inv2
        plt.plot(inv, label=f"inventory {j}")
        trop = [i for (i, a), v in zip(enumerate(action1), inv) if a==j]
        vrop = [v for (i, a), v in zip(enumerate(action1), inv) if a==j]
        plt.scatter(trop, vrop, color='b', label="order point")
        plt.xlim(len(inv)*(1-X), len(inv))
        plt.xlabel("day")
        plt.ylabel("amount")
        plt.title(f"Inference: {count}. run survived sim at avg service level > {env.min_service_level}")
        plt.legend() # loc = 'upper right')
        plt.show()

    """
    estimate the success rate
    """
    count = 0
    goods = 0

    while count < 1000:

        service_level1 = []
        service_level2 = []

        env = Environment()
    
        state, _ = env.reset()
        done = False
            
        while not done:

            action = agent.choose_action(state)
            next_state, reward, done, _, info = env.step(action)
            state = next_state

            service_level1.append(env.d1.service_level)
            service_level2.append(env.d2.service_level)
        
        steps = len(service_level1)
        avg_sl = (service_level1[-1] + service_level2[-1]) / 2 
        print(count, ": steps reached: ", steps, ", avg service level: ", avg_sl)
        count += 1
        goods += 1 if steps >= env.num_steps and avg_sl >= env.min_service_level else 0

    print("ratio: ", goods / count)

