import torch
import torch.nn as nn
from torchsummary import summary
import  numpy as np
"""
classes for training encoder
"""
class Encoder(nn.Module):
    def __init__(self, resolution=64, ch_img=3, ch_conv=4, ch_linear=64):
        super(Encoder, self).__init__()
        self.resolution = resolution
        conv_layers = [nn.Conv2d(ch_img, ch_conv, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(ch_conv)]
        resolution /= 2

        while (resolution != 1):
            conv_layers.append(nn.Conv2d(ch_conv, ch_conv * 2, kernel_size=3, stride=2, padding=1))
            ch_conv *= 2
            conv_layers.append(nn.BatchNorm2d(ch_conv))
            resolution /= 2

        self.conv_layers = nn.ModuleList(conv_layers)
        self.linear_layer = nn.Linear(in_features=ch_conv, out_features=ch_linear)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, imgs):
        # print("encoder conv-------------------------")
        temp = imgs
        i = 0
        for conv_layer in self.conv_layers:
            if i % 2 == 0:
                temp = self.tanh(conv_layer(temp))
            else:
                temp = conv_layer(temp)
            i += 1
            # print(temp[0:1, 0:1, 0:1, 0:1 ])
            # print(temp[2:3, 0:1, 0:1, 0:1])
        out_ = self.linear_layer(temp.squeeze())
        # print("encoder out-------------------------")
        # print(out_[0:1, 0:1])
        # print(out_[2:3, 0:1])
        return out_


class ThreeLayer_MLP(nn.Module):
    def __init__(self, in_ch=64, out_ch=64, hidden_units=32):
        super(ThreeLayer_MLP, self).__init__()
        self.linear_layer1 = nn.Linear(in_ch, hidden_units)
        self.linear_layer2 = nn.Linear(hidden_units, hidden_units)
        self.linear_layer3 = nn.Linear(hidden_units, out_ch)
        self.tanh = nn.Tanh()

    def forward(self, img):
        # print("mlp-------------------------")
        temp = self.tanh(self.linear_layer1(img))
        # print(temp[0:1, 0:1])
        # print(temp[2:3, 0:1])
        temp = self.tanh(self.linear_layer2(temp))
        # print(temp[0:1, 0:1])
        # print(temp[2:3, 0:1])
        out_ = self.linear_layer3(temp)
        # print(out_[0:1, 0:1])
        # print(out_[2:3, 0:1])
        return out_

class Reward_Predictor(nn.Module):
    def __init__(self, params):
        super(Reward_Predictor, self).__init__()
        self.device = params['device']
        self.t = params['T']
        self.K = params['K']
        self.out_ch_mlp = params['out_ch_mlp']
        self.in_size_lstm = params['in_size_lstm']

        self.encoder = Encoder(ch_img=1).to(self.device)
        self.mlp = ThreeLayer_MLP(params['in_ch_mlp'], params['out_ch_mlp']).to(self.device)

        reward_head_mlp = []
        for i in range(self.K):
            reward_head_mlp.append(ThreeLayer_MLP(in_ch=params['out_ch_mlp'], out_ch=1))
        self.reward_head_mlp = nn.ModuleList(reward_head_mlp).to(self.device)

        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(params['out_ch_mlp']).to(self.device)
        self.loss_ = nn.MSELoss()
        self.lstm = nn.LSTM(input_size=params['in_size_lstm'], hidden_size=params['out_ch_mlp'], num_layers=1, batch_first=True).to(self.device)

    def forward(self, img, bs=2):  # img (bs, t, 3, 64, 64)
        # summary(self.encoder, (3, 64, 64)) #show encoder structure
        img = img.to(self.device)
        encoded_img = []
        for i in range(bs):
            mm = torch.squeeze(img[i:i + 1, :, :, :, :], 0)
            encoded_img_ = self.encoder(mm)  # (t,64)
            encoded_img.append(encoded_img_)

        encoded_img = torch.stack(encoded_img, dim=0)  # (bs, t, 64)
        encoded_img = encoded_img.reshape(bs, -1)  # (bs, in_ch_mlp)

        h0 = self.mlp(encoded_img)  # (bs, out_ch_mlp)
        h0 = self.bn(h0)
        h0 = torch.unsqueeze(h0, dim=0)  # (1, bs, out_ch_mlp)

        input = torch.zeros(bs, self.t, self.in_size_lstm).to(self.device)  # (bs, t, 64)
        c0 = torch.zeros(1, bs, self.out_ch_mlp).to(self.device)  # (1, bs, out_ch_mlp)
        out_, _ = self.lstm(input, (h0, c0))  # (bs, t, 128)
        #print("lstm------------------------")
        #print(out_[0:1, 0:1, 0:1])
        #print(out_[2:3, 0:1, 0:1])

        #out_ = self.tanh(out_)
        # print("lstm activation------------------------")
        # print(out_[0:1, 0:1, 0:1])
        # print(out_[2:3, 0:1, 0:1])
        return out_

    def loss(self, h, labels, epoch, step, train=True):  # h (bs,t,128) label (K,bs,t)
        K_task_rewards = []
        for k in range(self.K):
            mlp3 = self.reward_head_mlp[k]
            t_step_rewards = []
            for t in range(self.t):
                h_ = h[:, t:t + 1, :].squeeze()  # (bs, 128)
                r_t = mlp3(h_)  # (bs, 1)
                t_step_rewards.append(r_t)
            t_step_rewards = torch.stack(t_step_rewards, dim=1).squeeze(dim=2)  # (bs, t)
            K_task_rewards.append(t_step_rewards)

        K_task_rewards = torch.stack(K_task_rewards, dim=0)  # (K, bs, t)
        K_task_rewards = self.sigmoid(K_task_rewards)
        loss_ = self.loss_(K_task_rewards, labels)
        if train == True:
            if epoch % 7 == 0 and (step % 499) == 0:
                print("epoch:", epoch, ", step:", step, ", pred:", K_task_rewards, ", reward:", labels)
        return loss_, K_task_rewards
"""
classes for training Decoder
"""
class Decoder(nn.Module):
    """image reconstruction"""

    def __init__(self, resolution=1, ch_img=1, ch_conv=128, ch_linear=128):
        super(Decoder, self).__init__()
        self.linear_layer = nn.Linear(in_features=ch_linear, out_features=ch_conv)
        self.tanh = nn.Tanh()

        conv_layers = [nn.ConvTranspose2d(ch_conv, ch_conv // 2, kernel_size=2, stride=2), nn.BatchNorm2d(ch_conv//2)]
        resolution *= 2
        ch_conv = ch_conv // 2

        while (resolution != 32):
            conv_layers.append(nn.ConvTranspose2d(ch_conv, ch_conv // 2, kernel_size=2, stride=2))
            ch_conv = ch_conv // 2
            conv_layers.append(nn.BatchNorm2d(ch_conv))
            resolution *= 2

        conv_layers.append(nn.ConvTranspose2d(ch_conv, ch_img, kernel_size=2, stride=2))
        conv_layers.append(nn.BatchNorm2d(ch_img))
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, imgs): # (bs, 128)
        temp = self.linear_layer(imgs).view(-1, 128, 1, 1)
        # i = 0
        for conv_layer in self.conv_layers:
            temp = conv_layer(temp)
            # i += 1
            # print("Conv"+str(i)+"-----------")
            # de_img_ = temp.cpu().detach().numpy()
            # img_flat = de_img_.flatten()
            # print(np.min(img_flat), np.max(img_flat))

        temp = self.tanh(temp)
        # print("after tanh-----------")
        # de_img_ = temp.cpu().detach().numpy()
        # img_flat = de_img_.flatten()
        # print(np.min(img_flat), np.max(img_flat))
        return temp

class Train_Decoder(nn.Module):
    """image reconstruction"""

    def __init__(self, T=25, ch_linear=128):
        super(Train_Decoder, self).__init__()
        self.t = T
        self.decoder_ = Decoder(ch_linear=ch_linear)

    def forward(self, imgs, bs=32): # (bs, t, 128)
        #summary(self.decoder_, (128, )) #show decoder structure
        decoded_img = []
        for i in range(bs):
            mm = torch.squeeze(imgs[i:i + 1, :, :], 0) #(t, 128)
            decoded_img_ = self.decoder_(mm)  # (t, 1, 64, 64)
            decoded_img.append(decoded_img_)

        decoded_img = torch.stack(decoded_img, dim=0)  # (bs, t, 1, 64, 64)
        return decoded_img


"""
classes for baselines
Copy from the Github repository of OpenAI Spinning Up
https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo
"""
import scipy.signal
from gym.spaces import Box, Discrete
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    """
    nn.Identity module will just return the input without any manipulation and can be used to e.g. replace other layers.
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(32, 32), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]











