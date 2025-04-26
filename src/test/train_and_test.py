from connect_four_gymnasium import ConnectFourEnv
from connect_four_gymnasium.players import BabyPlayer
from stable_baselines3 import PPO

env = ConnectFourEnv(opponent=BabyPlayer())
model = PPO("MlpPolicy", env, verbose=1)


model.learn(total_timesteps=10000)

from connect_four_gymnasium.tools import EloLeaderboard
from connect_four_gymnasium.players import (ModelPlayer)

myModelPlayer = ModelPlayer(model,name="Your trained Model")

print('Your elo: ',EloLeaderboard().get_elo(myModelPlayer, num_matches=200))