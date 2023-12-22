
import string
from typing import Any, Dict, Optional, Tuple, Type
import random

import numpy as np
from cmath import inf
import gym
import torch
from torch import long, nn,Tensor,tensor, bool
from torch.nn import BatchNorm1d, BatchNorm2d,AvgPool1d, Conv1d
from stable_baselines3.common.type_aliases import Schedule
from torch_geometric.nn import GCNConv, global_mean_pool,BatchNorm
from stable_baselines3.common.policies import MultiInputActorCriticPolicy, BasePolicy,BaseFeaturesExtractor
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)


class CustomGLPolicy(BasePolicy):
    def __init__(self, 
                observation_space: gym.spaces.Dict,
                action_space: gym.spaces.Discrete,
                lr_schedule: Schedule = Schedule,
                log_std_init: float = 0.0,
                use_sde: bool = False,
                squash_output: bool = False,
                ortho_init: bool = True,
                features_dim = 136,
                features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
                optimizer_kwargs: Optional[Dict[str, Any]] = None
                ):
        super(CustomGLPolicy,self).__init__(observation_space,
                                            action_space,
                                            features_extractor_kwargs,
                                            # features_dim,
                                            optimizer_class = optimizer_class,
                                            optimizer_kwargs = optimizer_kwargs,   
                                            squash_output = squash_output                                         
                                            )

        self.features_extractor = GNNFeatureExtractor()
        value_net_net = [nn.Linear(features_dim, features_dim, bias=True),nn.LeakyReLU(),nn.Linear(features_dim, features_dim, bias=True), nn.LeakyReLU(), nn.Linear(features_dim, features_dim, bias=True),nn.Linear(features_dim, 1, bias=True)]
        self.value_net = nn.Sequential(*value_net_net)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.action_dist = make_proba_distribution(action_space,use_sde=use_sde)

    def _predict(self, observation: Tensor, deterministic: bool = True) -> Tensor:
            actions, _, _ = self.forward(observation, deterministic=deterministic)
            return tensor([actions])

    def _build(self):
        pass

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
            distribution, values = self.get_distribution(obs)
            log_prob = distribution.log_prob(actions)

            return values, log_prob, distribution.entropy()

    def forward(self, obs, deterministic = False):

        distribution,values = self.get_distribution(obs)
        # print('distribution',distribution)
        # print('values',values)
        # entropy = random.random()
        # if entropy < 0.05:
        #     deterministic = False
        # else:
        #     deterministic = True  # Leaving entropy alone for now
        deterministic = True
        actions = distribution.get_actions(deterministic=deterministic)
        # print('actions',actions)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def predict_values(self,obs):
        _, values = self.get_distribution(obs)
        return values

    def get_distribution(self, obs):

        feature_embedding,mean_actions = self.extract_features(obs)

        values = self.value_net(feature_embedding)
        # print('value',values)

        latent_sde = feature_embedding

        if isinstance(self.action_dist, DiagGaussianDistribution):
            distribution =  self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            distribution =  self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
        else:
            raise ValueError("Invalid action distribution")

        return distribution, values
    






class GNNFeatureExtractor(nn.Module):
    def __init__(self): #This custom GNN receives the obs dict for the action log-probabilities
        super(GNNFeatureExtractor,self).__init__()
        verti_input_channels = 4
        ev_input_channels = 5
        hidden_channels = 150
        output_channels = 64 #length of each graph embedding
        input_dim = 136 #length of feature vector for MLP
        output_dim = 11 #output action dimensions
        self.vertiport = GCN(verti_input_channels,hidden_channels,output_channels) #input channels, hidden channels, output channels
        self.evtols = GCN(ev_input_channels,hidden_channels,output_channels) #Input channels, hidden channels, output channels
        self.output_space = GRLMLP(input_dim,output_dim) #Input dimension, output dimension

    def forward(self,data):
        # print('data from PPO',data)
        verti_features = data['vertiport_features'].float()
        verti_edge = data['vertiport_edge'][0].long() #edge connectivity matrix doesn't change, so only need the first instance
        ev_features = data['evtol_features'].float()
        ev_edge = data['evtol_edge'][0].long()
        next_drone = data['next_drone_embedding']
        mask = data['mask'].bool()

        verti_embed = self.vertiport(verti_features,verti_edge)
        ev_embed = self.evtols(ev_features,ev_edge)
        final_features = torch.cat((verti_embed,ev_embed,next_drone),dim=1)
        output = self.output_space(final_features,mask) #Testing how the custom feature extractor works

        # print('verti_features',verti_features.shape,'\n','verti_edge',verti_edge.shape)
        # print('\n','ev_features',ev_features.shape,'\n','ev_edge',ev_edge.shape)
        # print('\n','next_drone',next_drone.shape)
        # print('verti embed',verti_embed.shape)
        # print('ev embed',ev_embed.shape)
        # print('length of final features',final_features.shape)
        # print('shape of output is',output.shape)
        # output = output.reshape(output.shape[0],-1)
        # log_prob = torch.argmax(output,dim=1)

        return final_features, output


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, out_channels)
        self.Leaky_ReLU = nn.LeakyReLU(negative_slope=0.1)
        self.BatchNorm = BatchNorm(in_channels,track_running_stats=True) #Supposedly works the same as BatchNorm1d

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edge]


        # x = self.BatchNorm(x) #Acting weird with batches, ironically
        x = self.conv1(x, edge_index)
        x = self.Leaky_ReLU(x)
        x = self.conv2(x, edge_index)
        x = self.Leaky_ReLU(x)
        x = self.conv3(x, edge_index)
        x = self.Leaky_ReLU(x)
        x = self.conv4(x, edge_index)
        # print('x without pooling',x.shape)

        #Pooling for the graph embedding, just a mean pool
        return x.mean(dim=1)

class GRLMLP(nn.Module):
    def __init__(self,input_dim, output_dim) -> None:
        super().__init__()

        hidden1 = 128
        hidden2 = 64
        self.input = nn.Linear(input_dim,hidden1)
        self.hidden_layer = nn.Linear(hidden1, hidden1)
        self.hidden_layer2 = nn.Linear(hidden1,hidden2)
        self.output = nn.Linear(hidden2,output_dim)
        self.Leaky_ReLU = nn.LeakyReLU(negative_slope=0.1)
        self.tanh = nn.Tanh()

    def forward(self, x, mask=None):

        #x should be (input_dim,-1)
        # batch_size = x.shape[0]
        # x = x.view(batch_size,-1)
 
        #Forward propogation
        h1 = self.input(x)
        l1 = self.tanh(h1)

        h2 = self.hidden_layer(l1)
        # l2 = self.Leaky_ReLU(h2)
        l2 = self.tanh(h2)

        h3 = self.hidden_layer2(l2)
        l3 = self.tanh(h3)

        # print('mask',mask)
        ypred = self.output(l3)
        ypred[mask] = -inf
        return torch.log_softmax(ypred,dim = -1)
       




       
class CustomBaselinePolicy(BasePolicy):
    def __init__(self, 
                observation_space: gym.spaces.Dict,
                action_space: gym.spaces.Discrete,
                lr_schedule: Schedule = Schedule,
                log_std_init: float = 0.0,
                use_sde: bool = False,
                squash_output: bool = False,
                ortho_init: bool = True,
                features_dim = 50,
                features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
                optimizer_kwargs: Optional[Dict[str, Any]] = None
                ):
        super(CustomBaselinePolicy,self).__init__(observation_space,
                                            action_space,
                                            features_extractor_kwargs,
                                            # features_dim,
                                            optimizer_class = optimizer_class,
                                            optimizer_kwargs = optimizer_kwargs,   
                                            squash_output = squash_output                                         
                                            )

        self.features_extractor = BaseFeatureExtractor()
        value_net_net = [nn.Linear(features_dim, features_dim, bias=True),nn.LeakyReLU(),nn.Linear(features_dim, features_dim, bias=True), nn.LeakyReLU(), nn.Linear(features_dim, features_dim, bias=True),nn.Linear(features_dim, 1, bias=True)]
        self.value_net = nn.Sequential(*value_net_net)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.action_dist = make_proba_distribution(action_space,use_sde=use_sde)

    def _predict(self, observation: Tensor, deterministic: bool = True) -> Tensor:
            actions, values, log_prob = self.forward(observation, deterministic=deterministic)
            return tensor([actions])

    def _build(self):
        pass

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
            distribution, values = self.get_distribution(obs)
            log_prob = distribution.log_prob(actions)

            return values, log_prob, distribution.entropy()

    def forward(self, obs, deterministic = False):

        distribution,values = self.get_distribution(obs)
        # print('distribution',distribution)
        # print('values',values)
        # entropy = random.random()
        # if entropy < 0.05:
        #     deterministic = False
        # else:
        #     deterministic = True
        deterministic = True
        actions = distribution.get_actions(deterministic=deterministic)
        # print('actions',actions)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def predict_values(self,obs):
        _,values = self.get_distribution(obs)
        return values

    def get_distribution(self, obs):

        feature_embedding,mean_actions = self.extract_features(obs)

        values = self.value_net(feature_embedding)
        # print('value',values)

        latent_sde = feature_embedding

        if isinstance(self.action_dist, DiagGaussianDistribution):
            distribution =  self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            distribution =  self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
        else:
            raise ValueError("Invalid action distribution")

        return distribution, values

class BaseFeatureExtractor(nn.Module):
    def __init__(self): #This custom GNN receives the obs dict for the action log-probabilities
        super(BaseFeatureExtractor,self).__init__()
        input_channels = 64
        hidden_channels = 100
        output_channels = 50 #length of each graph embedding
        input_dim = 50 #length of feature vector for MLP
        output_dim = 11 #output action dimensions
        # self.vertiport = GCN(verti_input_channels,hidden_channels,output_channels) #input channels, hidden channels, output channels
        # self.evtols = GCN(ev_input_channels,hidden_channels,output_channels) #Input channels, hidden channels, output channels
        self.network = nn.Sequential(
                nn.Linear(input_channels,hidden_channels),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(hidden_channels,hidden_channels),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(hidden_channels, hidden_channels),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(hidden_channels,output_channels)
        )
        self.output_space = BaseMLP(input_dim,output_dim) #Input dimension, output dimension

    def forward(self,data):
        # print('data from PPO',data)
        next_drone = data['next_drone_embedding'].float()
        mask = data['mask'].bool()

        next_drone_embed = self.network(next_drone)
        # print(next_drone_embed)
        output = self.output_space(next_drone_embed,mask) #Testing how the custom feature extractor works

        return next_drone_embed,output

class BaseMLP(nn.Module):
    def __init__(self,input_dim, output_dim) -> None:
        super().__init__()

        hidden1 = 64
        hidden2 = 64
        self.input = nn.Linear(input_dim,hidden1)
        self.hidden = nn.Linear(hidden1,hidden2)
        self.hidden2 = nn.Linear(hidden1, hidden2)
        self.output = nn.Linear(hidden2,output_dim)
        self.Leaky_ReLU = nn.LeakyReLU(negative_slope=0.1)
        self.tanh = nn.Tanh()

    def forward(self, x, mask=None):

        #x should be (input_dim,-1)
        # batch_size = x.shape[0]
        # x = x.view(batch_size,-1)
 
        #Forward propogation
        h1 = self.input(x)
        # l1 = self.Leaky_ReLU(h1)
        l1 = self.tanh(h1)

        h2 = self.hidden(l1)
        l2 = self.tanh(h2)

        h3 = self.hidden2(l2)
        l3 = self.tanh(h3)
        # print('mask',mask)
        ypred = self.output(l3)
        ypred[mask] = -inf
        return torch.log_softmax(ypred,dim = -1)

        



