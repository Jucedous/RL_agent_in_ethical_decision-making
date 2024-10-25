import pomdp_py
import random
import pickle
import os
import matplotlib.pyplot as plt

class BurningRoomState(pomdp_py.State):
    def __init__(self, fire, object_safe, agent_safe, human_preference):
        self.fire = fire
        self.object_safe = object_safe
        self.agent_safe = agent_safe
        self.human_preference = human_preference

    def __hash__(self):
        return hash((self.fire, self.object_safe, self.agent_safe, self.human_preference))

    def __eq__(self, other):
        return (self.fire == other.fire and
                self.object_safe == other.object_safe and
                self.agent_safe == other.agent_safe and
                self.human_preference == other.human_preference)

    def __str__(self):
        return f"Fire: {self.fire}, Object Safe: {self.object_safe}, Agent Safe: {self.agent_safe}, Preference: {self.human_preference}"

class BurningRoomAction(pomdp_py.Action):
    def __init__(self, action_type):
        self.action_type = action_type

    def __str__(self):
        return self.action_type

class BurningRoomObservation(pomdp_py.Observation):
    def __init__(self, fire_status=None, preference=None):
        self.fire_status = fire_status
        self.preference = preference

    def __str__(self):
        return f"Fire Status: {self.fire_status}, Preference: {self.preference}"

class BurningRoomRewardModel(pomdp_py.RewardModel):
    def sample(self, state, action, next_state):
        if action.action_type == 'short_grab':
            if state.fire and not state.agent_safe:
                return -10  # Penalty if the agent is destroyed by the fire.
            elif not state.object_safe:
                return -5  # Penalty if the object is not safe.
            else:
                return 10  # Reward if the object is safe and the agent is not destroyed.
        elif action.action_type == 'long_grab':
            if state.object_safe:
                return 6
        elif action.action_type == 'ask':
            return -0.5
        return -5

class BurningRoomPolicyModel(pomdp_py.PolicyModel):
    def __init__(self, q_table, epsilon=0.1):
        self.q_table = q_table
        self.epsilon = epsilon

    def sample(self, state):
        if random.random() < self.epsilon:
            return random.choice([BurningRoomAction('short_grab'), BurningRoomAction('long_grab'), BurningRoomAction('ask')])
        else:
            state_key = str(state)
            best_action_str = max(self.q_table[state_key], key=self.q_table[state_key].get)
            return BurningRoomAction(best_action_str)

class BurningRoomTransitionModel(pomdp_py.TransitionModel):
    def sample(self, state, action):
        if action.action_type == 'short_grab':
            if state.fire and random.random() < 0.7:
                return BurningRoomState(state.fire, state.object_safe, False, state.human_preference)
            else:
                return BurningRoomState(state.fire, True, state.agent_safe, state.human_preference)
        elif action.action_type == 'long_grab':
            return BurningRoomState(state.fire, True, state.agent_safe, state.human_preference)
        elif action.action_type == 'ask':
            return state
        return state

class BurningRoomObservationModel(pomdp_py.ObservationModel):
    def sample(self, next_state, action):
        if action.action_type == 'ask':
            return BurningRoomObservation(fire_status=next_state.fire, preference=next_state.human_preference)
        return BurningRoomObservation()

    def probability(self, observation, next_state, action):
        if action.action_type == 'ask':
            return 1.0 if observation.fire_status == next_state.fire and observation.preference == next_state.human_preference else 0.0
        return 0.5

class BurningRoomAgent(pomdp_py.Agent):
    def __init__(self, init_belief, q_table, epsilon=0.1):
        policy_model = BurningRoomPolicyModel(q_table, epsilon)
        transition_model = BurningRoomTransitionModel()
        reward_model = BurningRoomRewardModel()
        observation_model = BurningRoomObservationModel()
        super().__init__(init_belief, policy_model, transition_model, observation_model, reward_model)

if __name__ == "__main__":
    reset_learning = True

    if reset_learning and os.path.exists('q_table.pkl'):
        os.remove('q_table.pkl')

    try:
        with open('q_table_burning_room.pkl', 'rb') as f:
            q_table = pickle.load(f)
    except FileNotFoundError:
        q_table = {}
        for fire in [True, False]:
            for object_safe in [True, False]:
                for agent_safe in [True, False]:
                    for preference in ['object', 'agent']:
                        state = BurningRoomState(fire, object_safe, agent_safe, preference)
                        state_key = str(state)
                        q_table[state_key] = {
                            'short_grab': 0,
                            'long_grab': 0,
                            'ask': 0
                        }

    init_belief = pomdp_py.Histogram({
        BurningRoomState(True, True, True, 'object'): 0.2,
        BurningRoomState(False, True, True, 'object'): 0.2,
        BurningRoomState(True, False, True, 'object'): 0.2,
        BurningRoomState(True, True, False, 'object'): 0.2,
        BurningRoomState(False, False, True, 'object'): 0.1,
        BurningRoomState(False, True, False, 'object'): 0.1
    })
    agent = BurningRoomAgent(init_belief, q_table)

    print("Initial Belief:")
    for state, prob in agent.belief.histogram.items():
        print(f"State: {state}, Probability: {prob}")

    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor

    num_episodes = 100
    episode_rewards = []

    for episode in range(num_episodes):
        cumulative_reward = 0
        for i in range(5):
            current_state = agent.belief.random()
            action = agent.policy_model.sample(current_state)
            next_state = agent.transition_model.sample(current_state, action)
            reward = agent.reward_model.sample(current_state, action, next_state)
            observation = agent.observation_model.sample(next_state, action)
            cumulative_reward += reward

            current_state_key = str(current_state)
            next_state_key = str(next_state)
            best_next_action_str = max(q_table[next_state_key], key=q_table[next_state_key].get)
            best_next_action = BurningRoomAction(best_next_action_str)
            q_table[current_state_key][action.action_type] += alpha * (reward + gamma * q_table[next_state_key][best_next_action.action_type] - q_table[current_state_key][action.action_type])

        episode_rewards.append(cumulative_reward)

    with open('q_table.pkl', 'wb') as f:
        pickle.dump(q_table, f)

    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward over Episodes')
    plt.show()