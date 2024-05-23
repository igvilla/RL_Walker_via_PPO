from dm_control import suite,viewer
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# import walker

from walker import MLPGaussianActor



"""
Setup walker environment
"""
r0 = False
e = suite.load('walker', 'walk',
                 task_kwargs={'random': r0})
U=e.action_spec();udim=U.shape[0]
X=e.observation_spec();xdim=14+1+9

hdim=32

controller = MLPGaussianActor(xdim, udim, 
                 hidden_sizes=[hdim]*3, activation=nn.Tanh)
controller.load_state_dict(th.load('actor2.pth'))

def policy_action(t):
    x = t.observation
    xp = np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())
    u,_ = controller(th.from_numpy(xp).float().unsqueeze(0))
    return u.sample()


"""
#Visualize a random controller
def u(dt):
    return np.random.uniform(low=U.minimum,
                             high=U.maximum,
                             size=U.shape)
viewer.launch(e,policy=u)
"""

# Example rollout using a network
viewer.launch(e,policy=policy_action)
