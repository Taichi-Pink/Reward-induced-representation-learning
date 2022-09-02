"""
Copy from the Github repository of OpenAI Spinning Up
https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo

For RL training, we use PPO. Regarding observation/state, we have three modes: 'oracle' or 'image-scratch' or 'reward-prediction'.

'oracle': (x,y)- coordinates of the agent, target, and distractors
'image-scratch': observation encoded by an encoder with randomly initialized parameters.
'reward-prediction': observation encoded by a pre-trained encoder
"""
from torch.optim import Adam
from models import *
from torch.utils.tensorboard import SummaryWriter
from sprites_env.envs.sprites import *
from general_utils import *
import imageio as io
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        #adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        #self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std() + 1e-10)  # Ziyi

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def ppo(env_fn, env_mode="state-based", rl_mode="oracle", actor_critic=MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=1000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, target_kl=0.01, save_freq=10):
    """
    Proximal Policy Optimization (by clipping),

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing
                                           | the log probability, according to
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical:
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`.

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Random seed
    seed += 1024  # Ziyi - seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    ###Ziyi
    data_spec = AttrDict(
        resolution=64,
        max_ep_len=steps_per_epoch,
        max_speed=0.05,  # total image range [0, 1]
        obj_size=0.2,  # size of objects, full images is 1.0
        follow=True,
    )
    env.set_config(data_spec)
    distractor = "d0"  # change1: d0 or d1
    ###
    obs_dim = env.observation_space.shape[0] # Ziyi
    act_dim = env.action_space.shape

    ####image-based
    if env_mode == 'image-based':
        T, N = 25, 25
        params = {'T': T, 'N': N, 'K': 4, 'in_ch_mlp': N * 64, 'out_ch_mlp': 128, 'in_size_lstm': 64, 'device': device}
        predictor = Reward_Predictor(params).to(device)
        # for name, param in predictor.named_parameters():
        #  print(name)
        predictor.load_state_dict(torch.load(
            "./model_weights/encoder/predictor8.pt"))  # change2: comment out or mot
        for param in predictor.parameters():
            param.requires_grad = False
        en_ = predictor.encoder
    ####

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ####reload model check parameters
    #ac.load_state_dict(torch.load("./model_weights/rl/" + rl_mode + "/"+distractor+"/ac0.pt"))
    #for name, param in ac.named_parameters():
    #   print(name)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
    print('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Set up experience buffer
    #local_steps_per_epoch = steps_per_epoch #Ziyi - local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    ###Ziyi
    writer = SummaryWriter(log_dir="./logs/logs_rl/"+rl_mode+"/"+distractor)
    step = 0
    image_trace = []
    ######

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        #print(ratio.mean().item()) its value of first iteration is 1.0
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv  # Clamps all elements in input into the range [ 0.8, 1.2 ]
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()  # one tensor

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()  # one tensor value
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl']  # Ziyi: kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                print('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            #mpi_avg_grads(ac.pi)  # average grads across MPI processes
            pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            #mpi_avg_grads(ac.v)  # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        #print("LossPi=",pi_l_old, "LossV=",v_l_old, "KL=",kl, "Entropy=",ent, "ClipFrac=",cf, "DeltaLossPi=",(loss_pi.item() - pi_l_old), "DeltaLossV=",(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        print("epoch", epoch)
        for t in range(steps_per_epoch):
            ####image-based
            if env_mode == 'image-based':
                o = o[None, None, :, :].repeat(2, axis=0)
                o = en_(torch.as_tensor(o, dtype=torch.float32).to(device))
                o = o[0:1, :].squeeze(0).cpu().detach().numpy()
            ####
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))  # a value: (0,1)


            ### Ziyi
            if (epoch == epochs-1) and (t>(steps_per_epoch-31)):
                print("step: ", t, "pos:", a)

            next_o, r, d, info = env.step(a)  # d means if done

            writer.add_scalar('reward/'+distractor, r, step)
            step += 1
            if ((epoch % save_freq == 0) or (epoch == epochs - 1)):
                if env_mode=="state-based":
                    image_trace.append(info)
                else:
                    image_trace.append(next_o)
            ###
            ep_ret += r
            ep_len += 1
            # save and log
            buf.store(o, a, r, v, logp)
            # Update obs (critical!)
            o = next_o

            if d:
                v = 0 #_, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                buf.finish_path(v)
                print("EpRet=", ep_ret, "EpLen=", ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        # Perform PPO update!
        update()
        # Save model and image trace
        torch.save(ac.state_dict(), "./model_weights/rl/"+rl_mode+"/"+distractor+"/ac"+str(epoch)+".pt")
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            image_stack = np.stack(image_trace, axis=0)  # (steps_per_epoch, 64, 64)
            img = make_image_seq_strip([image_stack[None, steps_per_epoch-30:steps_per_epoch, None, :, :].repeat(3, axis=2).astype(np.float32)],
                                       sep_val=1.0).astype(np.float32)
            img = img * 255.0
            img = img.astype(np.uint8)
            io.imwrite("./logs/logs_rl/" + rl_mode + "/"+distractor+"/trace" + str(epoch) + ".png", img[0].transpose(1, 2, 0))
        image_trace = []

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Sprites-v0') #change1: 'SpritesState-v0' or 'Sprites-v0'
    parser.add_argument('--env_mode', type=str, default='image-based') #change2: 'state-based' or 'image-based'
    parser.add_argument('--rl_mode', type=str, default='reward-prediction')  # change3:'oracle' or 'image-scratch' or 'reward-prediction'
    parser.add_argument('--hid', type=int, default=32)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=10000)#change4: 1000
    parser.add_argument('--epochs', type=int, default=500)#change5
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    ppo(lambda: gym.make(args.env), env_mode=args.env_mode, rl_mode=args.rl_mode, actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs)
