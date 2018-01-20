import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from IPython import embed


parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='CartPole-v0')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

if args.mode == 'train':
	# Okay, now it's time to learn something! We visualize the training here for show, but this
	# slows down training quite a lot. You can always safely abort the training prematurely using
	# Ctrl + C.
	dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

	# After training is done, we save the final weights.
	dqn.save_weights('dqn_{}_weights.h5f'.format(args.env_name), overwrite=True)
	dqn.test(env, nb_episodes=5, visualize=True)

elif args.mode == 'test':
	# Finally, evaluate our algorithm for 5 episodes.
	weights_filename = args.weights
	dqn.load_weights(weights_filename)