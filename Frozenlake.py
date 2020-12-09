################ Environment ################
import gym
import numpy as np
import contextlib
from gym.utils import seeding
from itertools import product
# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally: 
        np.set_printoptions(**original)

        
class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        
        self.random_state = np.random.RandomState(seed)
        
    def p(self, next_state, state, action):
        raise NotImplementedError()
    
    def r(self, next_state, state, action):
        raise NotImplementedError()
        
    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)
        
        return next_state, reward

        
class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        super().__init__(n_states, n_actions, seed)
        
        self.max_steps = max_steps
        
        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1./n_states)
        
    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)
        
        return self.state
        
    def step(self, action):
      
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')
        
        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)
        
        self.state, reward = self.draw(self.state, action)
        
        return self.state, reward, done
    
    def render(self, policy=None, value=None):
        raise NotImplementedError()

        
class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps,seed=None):
        lake = np.array(lake)
        n_states = lake.size
        pi = np.zeros(n_states, dtype=float)
        pi[np.where(lake.reshape(-1) == '&')[0]] = 1.0
        super().__init__(n_states, 4, max_steps, pi, seed)
        #slipprob = 0.1
        #lake: A matrix that represents the lake. For example:   
        # start (&), frozen (.), hole (#), goal ($)
        self.lake =  lake
        gridSize = lake.shape

        self.slip = slip
        
        cells = list(np.ndindex(gridSize)) #returns a tuple of indices based on dimension
        states = list(range(len(cells)))
    
    
        
        
        #HELPER FUNCTION: convert between one-dimensional and two-dimensional representations
        to_2d = lambda x :np.unravel_index(x, gridSize)
        to_1d = lambda x: np.ravel_multi_index(x, gridSize)  
        
        self.absorbing_state = self.n_states - 1
        
        baseline_reward = 0
        absorbing_cells = {(3, 3): 1, (1, 3): -1,(1,1): -1, (2,3): -1, (3,0): -1}
        probs = [0.925, 0.025, 0.025, 0]
        absorbing_states = {to_1d(s):r for s, r in absorbing_cells.items()} #
                
        #Store state reward
        
        state_rewards = np.full(self.n_states, baseline_reward) #fillstates with reward
        for state, reward in absorbing_states.items():
            state_rewards[state] = reward
        
        #get all the probability for every action
        action_outcomes = {} 
        actions = ['w','a','s','d']
        for i, action in enumerate(actions):
            p_ =  dict(zip([actions[j % 4] for j in range(i, self.n_actions + i)], probs))
            action_outcomes[actions[(i) % 4]] = p_
        
    
    #Create transition and reward matrix as a placeholder
        
        
        #P = {s: {a: [] for a in range(n_actions)} for s in range(self.n_states)}
      

        def get_new_cell(state, move):
            to_2d = lambda x :np.unravel_index(x, gridSize) #put in index, get the 2d index
            cell = to_2d(state) 
            if actions[move] == 'w':
                return cell[0] - 1, cell[1]
            elif actions[move] == 's':
                return cell[0] + 1, cell[1]
            elif actions[move] == 'd':
                return cell[0], cell[1] + 1
            elif actions[move] == 'a':
                return cell[0], cell[1] - 1   
        
        def update_trans_rewards(state, action, outcome):
            if state in absorbing_states.keys():
                self.transitions[action, state, state] = 1
            else:
                new_cell = get_new_cell(state, outcome)
                p = action_outcomes[actions[action]][actions[outcome]]
            
                if new_cell not in cells or new_cell:
                    self.transitions[action, state, state] += p
                    self.rewards[action, state, state] = baseline_reward
                else:
                    new_state = to_1d(new_cell)
                    self.transitions[action, state, new_state] = p
                    self.rewards[action, state, new_state] = state_rewards[new_state]
        
        
        self.rewards = np.zeros(shape=(self.n_actions, self.n_states, self.n_states))
        self.transitions = np.zeros((self.n_actions, self.n_states, self.n_states))
        actions_ = list(range(self.n_actions))
        
        for action, outcome, state in product(actions_, actions_, states):
            update_trans_rewards(state, action, outcome)
        # TODO:
    """    

    def categorical_sample(prob_n, np_random):

        prob_n = np.asarray(prob_n)
        csprob_n = np.cumsum(prob_n)
        return (csprob_n > np_random.rand()).argmax()
    

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]    
    """
    def step(self, action):
        
     
        state, reward, done = Environment.step(self, action)
        
        done = (state == self.absorbing_state) or done
        
        return state, reward, done
    
    #Get new cell after a action is taken


    #Updating probability matrix and return probability
    def p(self, next_state, state, action):
        
        p = self.transitions[action][state][next_state]
        
        """if 
            """
        return p
      
    #Update reward matrix and return reward
    def r(self, next_state, state, action):
        # TODO:
        if action == 'w':
            action = 0
        if action == 'a':
            action = 1
        if action =='s':
            action = 2
        if action =='d':
            action = 3
        r = self.rewards[action][state][next_state]
        
"""if new_state and new_state.isGoal():
            return 0
        else:
            return -1
"""
 
    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)
            
            if self.state < self.absorbing_state:
                lake[self.state] = '@'
                
            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']
            
            print('Lake:')
            print(self.lake)
        
            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))
            
            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))
            
    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)
        
        return self.state      
                
   
  
lake =  np.chararray((4,4))
lake[:] = '.'
lake[0][0]= '&'
lake[1][1]='#'
lake[1][3]='#'
lake[2][3]='#'
lake[3][0]='#'
lake[3][3]='$'
lake = np.array([i.decode() for i in lake])


n_actions = 4
n_states = 16 
slip = 0.1
max_steps = 16

fz = FrozenLake(lake,slip, max_steps, seed = 42)
              
def play(env):
    actions = ['w', 'a', 's', 'd']
    
    state = env.reset()
    env.render()
    
    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')
            
        state, r, done = env.step(actions.index(c))
        
        env.render()
        print('Reward: {0}.'.format(r))

################ Model-based algorithms ################
         

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=np.float)
    delta = 0
    value = np.zeros(env.n_states)
    for i in range(max_iterations):
        for s in range(env.n_nstates):
            v = 0
            # for all the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # Go through possible next states for each action
                for  prob, next_state in env.p[s][a]:
                    # Calculate the expected value
                    v += action_prob * prob * (reward[s][a] + gamma * V[next_state]) #action prob is 0.25 since there are 4 actions
            # How much the value function changed
            delta = max(delta, np.abs(v - V[s]))
            value[s] = v
        # Stop the evaluation oce threshold is reached
        if delta < theta:
            break

    return value

def q_extraction(env, V, s, gamma=1):

    q = np.zeros(env.n_actions)
         
    # for all possible actions in state s, which is actually all actions in env.n_actions as we saw 
    # earlier
    for a range(n_actions):
        # for all possible outcomes if action a_intended is taken
        for probs, _ in env.P[s][a]:
            # the new value is calculated
             q[a] += probs*(r[s][a] + gamma * V[next_stste])
    
    return q

def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)
    """in every state the improved policy choose the action with the highest state action function
    value
    """  
    for s in range(env.n_states):
       
        # get the state action values
        q_s = q_Extraction(env, V, s, gamma=1)
        # deterministic policy: chose the action with the highest state action value
        policy[s][np.argmax(q_s)] = 1
    # TODO:

    return policy
    
def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    if policy is None:
        policy = np.ones([env.nS, env.nA]) / env.nA
    else:
        policy = np.array(policy, dtype=int)
    
    for i in range (max_iterations_:
         value = policy_evaluation(env, policy, gamma, theta)
        # use the state function to improve the policy
         next_policy = policy_improvement(env, value, gamma)
         #calculated policy assigned as the next policy for the iteration
         delta = policy - next_policy
         policy = next_policy.copy()
         
         if delta.all()<theta:
             break
          
    # TODO:
        
    return policy, value
    
def value_iteration(env, gamma, theta, max_iterations, value=None):
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)
    newValue = value.copy()
    policy = np.zeros(env.n_states)
    
    for i in range(max_iterations):
        for state in range(self.n_states):
            action_values = []
            for i in range(n_actions): # i = 4 
                state_value = 0 #for every action of the current state
                for i in range(len(env.P[state][action])): #The 3 possible future scenario after 1 action is decided.
                    state_action_value = prob *(reward[s][a] + gamma* self.value[next_state]
                    state_value += state_action_value
            action_values.append(state_value) #value for each action
            max_action = np.argmax(np.asarray(action_values)) #choose best action
            policy[state] = best_action #policy extraction
            newValue[state] = action_value[max_action] #update statevalue    
        if abs(value-newvalue ) < theta:
            break
            print(i)
        else: value = newValue

    return policy, value
"""
################ Tabular model-free algorithms ################

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        s = env.reset()
        # TODO:
    
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value
    
def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    q = np.zeros((env.n_states, env.n_actions))
    
    for i in range(max_episodes):
        s = env.reset()
        # TODO:
        
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
        
    return policy, value

################ Non-tabular model-free algorithms ################

class LinearWrapper:
    def __init__(self, env):
        self.env = env
        
        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states
        
    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0
          
        return features
    
    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)
        
        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)
            
            policy[s] = np.argmax(q)
            value[s] = np.max(q)
        
        return policy, value
        
    def reset(self):
        return self.encode_state(self.env.reset())
    
    def step(self, action):
        state, reward, done = self.env.step(action)
        
        return self.encode_state(state), reward, done
    
    def render(self, policy=None, value=None):
        self.env.render(policy, value)
        
def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        q = features.dot(theta)

        # TODO:
    
    return theta
    
def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    theta = np.zeros(env.n_features)
    
    for i in range(max_episodes):
        features = env.reset()
        
        # TODO:

    return theta    

################ Main function ################

def main():
    seed = 0
    
    # Small lake
    lake =   [['&', '.', '.', '.'],
              ['.', '#', '.', '#'],
              ['.', '.', '.', '#'],
              ['#', '.', '.', '$']]

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    
    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 100
    
    print('')
    
    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)
    
    print('')
    
    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)
    
    print('')
    
    print('# Model-free algorithms')
    max_episodes = 2000
    eta = 0.5
    epsilon = 0.5
    
    print('')
    
    print('## Sarsa')
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)
    
    print('')
    
    print('## Q-learning')
    policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)
    
    print('')
    
    linear_env = LinearWrapper(env)
    
    print('## Linear Sarsa')
    
    parameters = linear_sarsa(linear_env, max_episodes, eta,
                              gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)
    
    print('')
    
    print('## Linear Q-learning')
    
    parameters = linear_q_learning(linear_env, max_episodes, eta,
                                   gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)
"""