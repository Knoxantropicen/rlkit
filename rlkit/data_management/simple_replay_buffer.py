import numpy as np

from rlkit.data_management.replay_buffer import ReplayBuffer


class SimpleReplayBuffer(ReplayBuffer):
    def __init__(
            self, max_replay_buffer_size, observation_dim, action_dim,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation
        self._advance()

    def add_sample_batch(self, observations, actions, rewards, terminals,
                    next_observations, **kwargs):
        assert len(observations) == len(actions) == len(rewards) == len(terminals) == len(next_observations)
        batch_size = len(observations)
        buffer_remaining_size = self._max_replay_buffer_size - self._top
        sample_top = 0

        while batch_size > buffer_remaining_size:
            self._observations[self._top:] = observations[sample_top:buffer_remaining_size]
            self._actions[self._top:] = actions[sample_top:buffer_remaining_size]
            self._rewards[self._top] = rewards[sample_top:buffer_remaining_size]
            self._terminals[self._top:] = terminals[sample_top:buffer_remaining_size]
            self._next_obs[self._top:] = next_observations[sample_top:buffer_remaining_size]
            self._advance_batch(buffer_remaining_size)
            batch_size -= buffer_remaining_size
            sample_top += buffer_remaining_size
            buffer_remaining_size = self._max_replay_buffer_size
        
        self._observations[self._top:self._top+batch_size] = observations[sample_top:]
        self._actions[self._top:self._top+batch_size] = actions[sample_top:]
        self._rewards[self._top:self._top+batch_size] = rewards[sample_top:]
        self._terminals[self._top:self._top+batch_size] = terminals[sample_top:]
        self._next_obs[self._top:self._top+batch_size] = next_observations[sample_top:]
        self._advance_batch(batch_size)

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1
        
    def _advance_batch(self, n):
        self._top = (self._top + n) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += n
            self._size = min(self._size, self._max_replay_buffer_size)

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )

    def num_steps_can_sample(self):
        return self._size
