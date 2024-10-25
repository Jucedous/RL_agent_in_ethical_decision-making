import pomdp_py
import random

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
                return -10
            elif state.object_safe:
                return 10
        elif action.action_type == 'long_grab':
            if state.object_safe:
                return 6
        elif action.action_type == 'ask':
            return -0.5
        return -5

class BurningRoomPolicyModel(pomdp_py.PolicyModel):
    def sample(self, state):
        return random.choice([BurningRoomAction('short_grab'), BurningRoomAction('long_grab'), BurningRoomAction('ask')])

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
    def __init__(self, init_belief):
        policy_model = BurningRoomPolicyModel()
        transition_model = BurningRoomTransitionModel()
        reward_model = BurningRoomRewardModel()
        observation_model = BurningRoomObservationModel()
        super().__init__(init_belief, policy_model, transition_model, observation_model, reward_model)

if __name__ == "__main__":
    init_belief = pomdp_py.Histogram({
        BurningRoomState(True, True, True, 'object'): 0.2,   # Fire present, object and agent safe
        BurningRoomState(False, True, True, 'object'): 0.2,  # No fire, object and agent safe
        BurningRoomState(True, False, True, 'object'): 0.2,  # Fire present, object not safe, agent safe
        BurningRoomState(True, True, False, 'object'): 0.2,  # Fire present, object safe, agent not safe
        BurningRoomState(False, False, True, 'object'): 0.1, # No fire, object not safe, agent safe
        BurningRoomState(False, True, False, 'object'): 0.1  # No fire, object safe, agent not safe
    })
    agent = BurningRoomAgent(init_belief)

    print("Initial Belief:")
    for state, prob in agent.belief.histogram.items():
        print(f"State: {state}, Probability: {prob}")

    for i in range(5):
        action = agent.policy_model.sample(agent.belief.random())
        print(f"Step {i + 1} - Action: {action}")
        next_state = agent.transition_model.sample(agent.belief.random(), action)
        reward = agent.reward_model.sample(agent.belief.random(), action, next_state)
        observation = agent.observation_model.sample(next_state, action)
        print(f"Next State: {next_state}")
        print(f"Reward: {reward}")
        print(f"Observation: {observation}")
        print("-" * 30)