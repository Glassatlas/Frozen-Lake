import numpy as np

from itertools import product
# Configures numpy print options
#@contextlib.contextmanager
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
        self.n_steps = 0
        self.state = 0
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
        self.lake_flat = lake.flatten()
        n_states = lake.size
        pi = np.zeros(n_states, dtype=float)
        pi[np.where(lake.reshape(-1) == '&')[0]] = 1.0
        super().__init__(n_states, 4, max_steps, pi, seed)
        # start (&), frozen (.), hole (#), goal ($)
        self.lake =  lake
        self.gridSize = lake.shape

        self.slip = slip

        cells = list(np.ndindex(self.gridSize)) #returns a tuple of indices based on dimension
        states = list(range(len(cells)))

        #HELPER FUNCTION: convert between one-dimensional and two-dimensional representations
        to_2d = lambda x :np.unravel_index(x, self.gridSize)
        to_1d = lambda x: np.ravel_multi_index(x, self.gridSize)

        self.absorbing_state = self.n_states

        baseline_reward = 0


        #Define hole and goal states via symbols
        '''Hole is = -1 and goal state = 1
        We start with a list that turns are later converted to dictionary'''
        self.absorbing_states = np.zeros(n_states, dtype=float)
        self.absorbing_states[np.where(lake.reshape(-1) == '#')[0]] = -1.0
        self.absorbing_states[np.where(lake.reshape(-1) == '$')[0]] = +1.0 #returns a list.
        absorbing_cells = {to_2d(s):i for s, i in enumerate(self.absorbing_states)} #turn states into cells coordinate(n,m)
        absorbing_cells = {x:y for x,y in absorbing_cells.items() if y!=0} #get rid of all 0
        self.absorbing_states = {to_1d(s):r for s, r in absorbing_cells.items()} #turn cells into state dict
        self.goal_state = {x:y for (x,y) in self.absorbing_states.items() if y == 1.0} #get goal state

        probs = [0.925, 0.025, 0.025, 0.025]


        #A dict that contains all the probability for every action
        action_outcomes = {}
        self.actions = ['w','a','s','d']
        for i, action in enumerate(self.actions):
            p_ =  dict(zip([self.actions[j % 4] for j in range(i, self.n_actions + i)], probs))
            action_outcomes[self.actions[(i) % 4]] = p_


        def get_new_cell(state, move):
            to_2d = lambda x :np.unravel_index(x, self.gridSize) #put in index, get the 2d index
            actions = ['w','a','s','d']
            cell = to_2d(state)
            v = None
            if actions[move] == 'w':
                v= cell[0] - 1, cell[1]
            elif actions[move] == 's':
                v = cell[0] + 1, cell[1]
            elif actions[move] == 'd':
                v = cell[0], cell[1] + 1
            elif actions[move] == 'a':
                v = cell[0], cell[1] - 1
            return v

        #Create transition and reward matrix as a placeholder
        #p = {s: {a: [] for a in range(n_actions)} for s in range(self.n_states)}
        self.rewards = np.zeros(shape=(self.n_actions, self.n_states, self.n_states)) #4x16x16 matrix for 4 actions
        self.transitions = np.zeros((self.n_actions, self.n_states, self.n_states)) #4x16x16 matrix for 4 actions

        #update of transition matrix and reward matrix.
        #here, p => {s: {a: [] for a in range(n_actions)} for s in range(self.n_states)}

        def update_trans_rewards(state, action, outcome):
            if state in self.absorbing_states.keys():
                self.transitions[action, state, state] = 1 #probability that it will be in the same state
                if state in self.goal_state.keys():
                    self.rewards[action,state,state] = 1
            else:
                new_cell = get_new_cell(state, outcome)
                p = action_outcomes[self.actions[action]][self.actions[outcome]]
                if new_cell not in cells:# or new_cell:
                    self.transitions[action, state, state] += p
                else:
                    new_state = to_1d(new_cell)
                    self.transitions[action, state, new_state] = p

        actions_ = list(range(self.n_actions))
        for action, outcome, state in product(actions_, actions_, states):
            update_trans_rewards(state, action, outcome)


    def get_new_cell(self,state, move):
        to_2d = lambda x :np.unravel_index(x, self.gridSize) #put in index, get the 2d index
        to_1d = lambda x: np.ravel_multi_index(x, self.gridSize)
        actions = ['w','a','s','d']
        cell = to_2d(state)
        v = None
        if actions[move] == 'w':
            v= cell[0] - 1, cell[1]
        elif actions[move] == 's':
            v = cell[0] + 1, cell[1]
        elif actions[move] == 'd':
            v = cell[0], cell[1] + 1
        elif actions[move] == 'a':
            v = cell[0], cell[1] - 1
        return v

    def step(self, action):
        state, reward, done = Environment.step(self, action)
        done = (state == self.absorbing_state) or done
        return state, reward, done

    class Memoize:

        def __init__(self, fn):
            self.fn = fn
            self.memo = {}

        def __call__(self, *args):
            if args not in self.memo:
                self.memo[args] = self.fn(*args)
            return self.memo[args]


    #return transitional probability
    def p(self, next_state, state, action):
      '''p value is taken from the transition matrix'''
      p = self.transitions[action][state][next_state]
      #print(p)  #print all the  transitional probability for every possible state.
      return p

    #Return tuple of tuple of probability of an action with associated future state
    def p_v2(self, state, action):
        future_states = []
        pList= env.transitions[action][state]
        pList = pList[pList != 0]
        tup_p = tuple(e for e in pList)
        y = np.where(env.transitions[state][action] != 0)
        for item in y:
            future_states.extend(item)
        future_states = tuple(future_states)
        probs = tuple(zip(tup_p,future_states))

        return probs

    #Update reward matrix and return reward
    def r(self, next_state, state, action):

        r = self.rewards[action][state][next_state]
        return r

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
            # policy = np.array([actions[a] for a in policy[:-1]])
            policy = np.array([actions[a] for a in policy])
            #print(policy)
            print(policy.reshape(self.lake.shape))

            print('Value:')
            # with _printoptions(precision=3, suppress=True):
                # print(value[:-1].reshape(self.lake.shape))
            print(value.reshape(self.lake.shape))


        def reset(self):
            self.n_steps = 0
            self.state = self.random_state.choice(self.n_states, p=self.pi)


            return self.state

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
    for a in range(n_actions):
        # for all possible outcomes if action a_intended is taken
        for probs, _ in env.P[s][a]:
            # the new value is calculated
             q[a] += probs*(r[s][a] + gamma * V[next_stste])

    return q

def policy_improvement(env, value, gamma):
    policy = np.zeros(env.n_states, dtype=int)
  # in every state the improved policy choose the action with the highest state action function
  #   value

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

    for i in range (max_iterations_):
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
        for state in range(env.n_states):
            action_values = []
            for action in range(env.n_actions): # i = 4
                state_value = 0 #for every action of the current state
                for i in range(len(env.p_v2(state, action))): #The 3 possible future scenario after 1 action is decided.
                    prob, future_state = env.p_v2(state, action)[i]
                    next_state, reward, done = env.step(action)
                    state_action_value = prob * (reward + gamma* value[next_state])
                    state_value += state_action_value
            action_values.append(state_value) #value for each action
            max_action = np.argmax(np.asarray(action_values)) #choose best action
            policy[state] = max_action #policy extraction
            newValue[state] = action_values[max_action] #update statevalue
        if abs(value-newvalue ) < theta:
            break
            print(i)
        else: value = newValue

    return policy, value


################ Tabular model-free algorithms ################

def epsilon_greedy(env, q, epsilon,s, seed=None):
    # Check presence of seed
    if seed == None:
        rand_p = np.random.rand()
    else:
        np.random.seed(seed)
        rand_p = np.random.rand()
        # Outputs random action with p=epsilon, and argmax(Q) with p=1-epsilon
    if rand_p < epsilon:
        action = np.random.randint(0, env.n_actions)
    else:
        # if q.ndim==1:
        #     action = np.argmax(q[s])
        # else:
        #     action = np.argmax(q[s, :])
        action = np.argmax(q[s, :])

    return action


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    #S env, N max_episodes, alpha eta, gamma, epsilon, seed=None
    #print(type(env), type(max_episodes), type(eta), type(gamma), type(epsilon))
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        #Done:
        #Epsilon greedy policy
        action = epsilon_greedy(env, q, epsilon[i],s, seed)
        #Start counter
        done = False
        while not done:
            # Get next state
            next_state, reward, done = env.step(action)
            # Get next action
            next_action = epsilon_greedy(env, q, epsilon[i],next_state, seed)
            # Estimate q
            q[s,action] += eta[i]*(reward+(gamma*q[next_state, next_action])-q[s,action])
            # Update new state and new action
            s,action = next_state, next_action
    # Return deterministic policy
    policy = q.argmax(axis=1) #Get index
    value = q.max(axis=1)
    print('sarsa results:',policy, value)

    return policy, value #pi policy

def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    # for each(s,a)∈S×A do: Q(s,a) ← 0 end for
    q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        s = env.reset()
        action = epsilon_greedy(env, q, epsilon[i],s, seed)
        done = False
        # while state s is not terminal do:
        while not done:
            # r ← observed reward for action a at state s；s′ ← observed next state for action a at state s
            newstate, r, done = env.step(action)
            # Select action a for state s according to an ε-greedy policy based on Q.
            # Epsilon greedy policy
            # Outputs random action with p=epsilon, and argmax(Q) with p=1-epsilon
            newaction = epsilon_greedy(env, q, epsilon[i],newstate, seed)
            # Q(s,a)←Q(s,a)+α[r+γmaxa′ Q(s′,a′)−Q(s,a)]
            q[s][action] += eta[i]*(r + gamma *  np.max(q[newstate][newaction]) - q[s][action])
            # s ← s′
            s = newstate
    # for each state s ∈ S do # π(s) ← arg maxa Q(s, a)
    policy = q.argmax(axis=1)
    print("hellllo")
    print(policy)
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
    #phi is the feature vector
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros([env.n_features])
    q = np.zeros([env.n_actions])
    #q = np.zeros((env.n_states, env.n_actions))

    for i in range(max_episodes):
        features = env.reset()
        for a in range(env.n_actions):
            #print(q.shape, theta.shape, features.shape)
            q[a] = features[a].dot(theta)
            #q[a] = theta.dot(features[a])
        # Done:
        # Epsilon greedy policy
        s = env.env.reset()
        action = epsilon_greedy(env, q, epsilon[i],s, seed)
        # Start time counter
        done = False
        while not done:
            # Get next state
            next_state, reward, done = env.step(action)
            # Get next action
            next_action = epsilon_greedy(env, q, epsilon[i],next_state, seed)
            # Create delta, the temporal difference (different between q-learning one)
            delta = reward + (gamma * q[next_action]) - q[action]
            # Update theta
            theta += eta[i]*delta*features[s,action]
            # Update new state and new action
            s, action = next_state, next_action
    # Return deterministic policy
    policy = q.argmax(axis=1)  # Get index
    value = q.max(axis=1)
    print('theta', theta)
    return theta

def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    for i in range(max_episodes):
        features = env.reset()
        # s ← initial state for episode i
        s = env.reset()
        # for each action a: Q(a) ← sum of θi φ(s, a)i
        q = features.dot(theta)
        done = False
        action = epsilon_greedy(env, q, epsilon[i],s, seed)
        # while state s is not terminal do
        while not done:
            # r ← observed reward, s′ ← observed next state for action a at state s
            newstate, r, done = env.step(action)
            newaction = epsilon_greedy(env, q, epsilon[i],newstate, seed)
            # δ ← r − Q(a)
            sigma = r - q
            # for each action a′: do
                # Q(a′) ← 􏰀i θi φ(s′, a′)i
            newq = features[newstate][newaction].dot(theta) # not sure
            # δ ← δ + γ maxa′ Q(a′) {Note: δ is the temporal difference}
            sigma += gamma*np.max(newaction)*newq
            # θ ← θ + αδφ(s, a)
            theta += eta*sigma*features[s][action] # not sure
            # s ← s′
            s = newstate

    return theta

################ Main function ################
def main():

    seed = 0

    # Small lake what the lake will look like
    ''' lake =   [['&', '.', '.', '.'],
              ['.', '#', '.', '#'],
              ['.', '.', '.', '#'],
              ['#', '.', '.', '$']]
    '''


    lake =  np.chararray((4,4))
    lake[:] = '.'
    lake[0][0]= '&'
    lake[1][1]='#'
    lake[1][3]='#'
    lake[2][3]='#'
    lake[3][0]='#'
    lake[3][3]='$'
    lake = np.array([i.decode() for i in lake])

    #big lake
    '''[['&', '.', '.', '.', '.', '.', '.', '.'],
       ['.', '.', '.', '.', '.', '.', '.', '.'],
       ['.', '.', '.', '#', '.', '.', '.', '.'],
       ['.', '.', '.', '.', '.', '#', '.', '.'],
       ['.', '.', '.', '#', '.', '.', '.', '.'],
       ['.', '#', '#', '.', '.', '.', '.', '.'],
       ['.', '#', '.', '.', '.', '.', '.', '.'],
       ['.', '.', '.', '.', '.', '.', '.', '$']]

    '''

    # lake =  np.chararray((8,8))
    # lake[:] = '.'
    # lake[0][0]= '&'
    # lake[2][3]='#'
    # lake[3][5]='#'
    # lake[4][3]='#'
    # lake[5][1]='#'
    # lake[5][2]='#'
    # lake[5][6]='#'
    # lake[6][6]='#'
    # lake[6][4]='#'
    # lake[6][1]='#'
    # lake[7][3]='#'
    # lake[7][7]='$'
    # lake = np.array([i.decode() for i in lake])

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    play(env)
    # env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)

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

if __name__ == "__main__":
   main()
