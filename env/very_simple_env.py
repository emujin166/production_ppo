"""
changelog:
24.9.25
this implementation replaces the complex model with a very simple one, details see comments in the model file
- 2 products: one low demand, one high demand, intially fixed
- 1 machine with no job pipeline and fixed per piece production time
- initally "rotary job selector": 0=nothing, else product number
- sizing:
  - product 1: demand = 10 pieces every 3 days, replenishment amount = 100 pcs (approx 33 days of avg daily demand)
  - product 2: demand = 100 pieces every 2 days, replenishment amount = 1500 pcs (30 day of avg daily demand)
  - production time = 75 pcs/d (20 days for the product 2 order)
  - max inventory = 10 times the production order amount
  - terminate if inventory overflow, or mean_service_level lower than min_service_level, or num_steps

options for refinement:
- variation of demand amount and time
- variation of production time
- add job pipeline

Means: replenishment - Nachschub, Auffüllung
Inventar - physische Zählung und Erfassung aller Bestände und Materialien in einem Lager
"""
import gym
import numpy as np
from gym.spaces import MultiDiscrete

class Demand:
    """
    generates demand and tries to fulfill: first backorder, then current order
    maintains inventory
    """
    #Constructor - when an object is created, a memory is allocated for it and init helps organize that memory by assigning values to attributes
    def __init__(self, start_inv, max_inv, demand_amount, demand_cycle_length, replen_amount): 
        self.start_inv = start_inv
        self.max_inv = max_inv
        self.demand_amount = demand_amount
        self.demand_cycle_length = demand_cycle_length
        self.replen_amount = replen_amount
        self.reset()

    #Lager leeren
    def reset(self):
        self.inv = self.start_inv
        self.backlog = 0 # wartende Aufträge in der Warteschlange (Machine)
        self.replens = 0  # how often has been replenished
        self.demands = 0  # how often has demand been generate
        self.fulfills = 0 # how often has demand been fulfilled properly

    #Lager Auffüllen 
    def replenish(self, amount): 
        self.inv += amount
        self.replens += 1

    def gen_demand_and_fulfill(self, cycle):
        # 1. generate demand
        demand = self.demand_amount if cycle % self.demand_cycle_length == 0 else 0

        # 2. fulfill backlog first and then new demand
        # a) backlog - accumulation of tasks, requests that needs to be completed but hasnt been processed yet
        amount_used = min(self.backlog, self.inv)
        self.backlog -= amount_used
        self.inv -= amount_used

        # b) new demand
        if demand: 
            # wenn es eine Nachfrage(not null) gibt
            self.demands += 1
            if self.inv < demand:
                fulfilled = False # ?als done flag später?
                self.backlog += (demand - self.inv)
                #self.inventory = 0 # warum wird auf 0 gesetzt?
            else:
                self.fulfills += 1
                fulfilled = True
                self.inv -= demand # Die Nachfrage wird wird vom Lager abgezogen 
                return fulfilled
        else:
            # wenn es keine Nachfrage(null) gibt
            return None 
    
    #Decorater - a function that takes other functions as parameters. Syntax of property function is (fget=none, fset=none, fdel=none, doc=none)
    #Property function has 3 methods: getter(), setter(), deleter()
    @property
    def service_level(self):
        return self.fulfills / self.demands if self.demands else 1
    @property
    def inv_util(self):
        return self.inv / self.max_inv  #darf 1 nicht überschreiten

class Machine:
    """
    takes orders if possible and produce them
    """
    def __init__(self, daily_amount):
        self.daily_amount = daily_amount # production amount per day
        self.reset()

    def reset(self):
        self.order_amount = 0 # holds the current total prod amount requested and indicates busy
        self.pend_amount = 0 # counts the remaining amount
        self.order_product = None

    def take_order(self, amount, product):
        # returns success if machine was not busy and new order could be taken
        if self.pend_amount != 0:
            #warum?
            return False #machine is busy
        else:
            self.order_amount, self.order_product = amount, product
            self.pend_amount = amount
            return True #machine not busy
        
    def produce_one_day(self):
        # produces one day, and finishes order and returns amount if finished 
        if self.order_amount == 0:
            # no pending order
            return 0, None
        else:
            # a) produce - warum produce?
            self.pend_amount = max(0, self.pend_amount - self.daily_amount) #when the amount exceeds the daily amount, it will be stored in pending 

            # b) finish up order if possible
            if self.pend_amount == 0:
                finished_amount, finished_product = self.order_amount, self.order_product #wenn es keine pending Menge gibt, bedeutet, dass alle order amounts verarbeitet wurden und fertig sind
                self.order_amount, self.order_product = 0, None #aktualisieren
                return finished_amount, finished_product
            else:
                return 0, None
    @property
    def busy(self):
        return True if self.order_amount != 0 else False

def num2bin(num, max_num, max_bin): #wozu?
    # translates from num to bin
    assert num <= max_num, f"num={num} exceeds max_num={max_num}"
    assert num >=0, f"num={num} < 0"
    return int(num / max_num * max_bin)

class Environment(gym.Env):
    def __init__(self, min_service_level = 0.7, bin_size=25, num_steps=2_000):
        # rotary product selector
        self.action_space = MultiDiscrete([3]) # 3 Dimensionen

        # obs: inventories, service_levels, machine_busy
        self.observation_space = MultiDiscrete([bin_size, bin_size, bin_size, bin_size, 2])
        self.min_service_level = min_service_level
        self.bin_size = bin_size
        self.num_steps = num_steps
        self.reset()  

    def reset(self):
        self.t = 0 #?
        # start_inv, max_inv, demand_amount, demand_cycle_length, replen_amount)
        self.d1 = Demand(start_inv=100, max_inv=10*100, demand_amount=10,  demand_cycle_length=3, replen_amount=100)
        self.d2 = Demand(start_inv=1_500, max_inv=10*1_500, demand_amount=100, demand_cycle_length=2, replen_amount=1_500)
        self.m = Machine(daily_amount=75)
        return self.get_obs(), {}   
    
    #Qlearning
    def get_obs(self):
        obs = np.array([num2bin(self.d1.inv,           self.d1.max_inv, self.bin_size - 1),
                        num2bin(self.d2.inv,           self.d2.max_inv, self.bin_size - 1),
                        num2bin(self.d1.service_level, 1,               self.bin_size - 1),
                        num2bin(self.d2.service_level, 1,               self.bin_size - 1),
                        int(self.m.busy)], dtype=np.int64)
        return obs
    
    #decision making: fragen ob machine busy ist wenn ist, dann gar nicht 
    def step(self, action):
        # 1. fulfill
        self.d1.gen_demand_and_fulfill(self.t) #t = cycle(jeder 3 cycle bei P1, jeder 2 cycle bei P2 wird gecheckt?)
        self.d2.gen_demand_and_fulfill(self.t)

        # 2. plan ???
        if action[0].item() != 0:
            product = action[0].item()
            amount = self.d1.replen_amount if product == 1 else self.d2.replen_amount
            take_order_success = self.m.take_order(amount, product)
        else:
            take_order_success = True

        # 3. process one day - ???????
        amount, product = self.m.produce_one_day()
        if amount != 0:
            if product == 1:
                self.d1.replenish(amount)
            else:
                self.d2.replenish(amount)
        self.t += 1 

        # 4. ****reward
        reward = + (self.d1.service_level - self.d1.inv_util) \
                 + (self.d2.service_level - self.d2.inv_util) \
                 - (1 if not take_order_success else 0) 
                 # + (self.d1.service_level - self.d1.inv_util) * (self.d1.service_level - self.d1.inv_util) \
                 
        # 5. stop conditions
        terminate =    False \
                    or self.t >= self.num_steps \
                    or self.d1.inv_util > 1 \
                    or self.d2.inv_util > 1 \
                    or self.d1.service_level < self.min_service_level \
                    or self.d2.service_level < self.min_service_level
        
        # 6. finish up
        obs = self.get_obs() if not terminate else None
        return obs, reward, terminate, False, None