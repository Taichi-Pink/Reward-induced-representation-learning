"""
Train encoder using 4 rewards for task 3,4,5
"""
from models import *
from load_data import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
############ change parameters
T, N = 25, 25
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params = {'T': T, 'N':N, 'K':4,'in_ch_mlp': N*64, 'out_ch_mlp': 128, 'in_size_lstm': 64, 'device': device}
learning_rate = 0.0001
spec = AttrDict(
        resolution=64,
        max_seq_len=N+T,
        max_speed=0.05,  # total image range [0, 1]
        obj_size=0.2,  # size of objects, full images is 1.0
        shapes_per_traj=2,
        rewards=[AgentXReward, AgentYReward, TargetXReward, TargetYReward],
    )
epoch, bs, seq_len = 5, 32, 160000
############

data_load = data_loader(spec, batch_size=bs, seq_len=seq_len) #seq_len: how many data per epoch
writer    = SummaryWriter(log_dir='./logs/logs_encoder')
predictor = Reward_Predictor(params).to(device)
#summary(predictor, (N, 1, 64, 64))
predictor.load_state_dict(torch.load("./model_weights/encoder/predictor8.pt"))

optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
loss_history = []
for ep in range(epoch):
    l = 0.0
    step = 0
    for img_, r_ in data_load:  # r_shape 4x(bs, T+N),  img (bs, T+N, 1, 64, 64)
        img__ = img_[:, :N, :, :, :]  # img__ (bs, N, 1, 64, 64)

        #### test loaded imgs and rewards
        # for i in range(bs):
        #     print("batch: ", i)
        #     for seq in range(N, N+T):
        #         r_agent_x = r_['agent_x'][i, seq].numpy()
        #         r_agent_y = r_['agent_y'][i, seq].numpy()
        #         r_target_x = r_['target_x'][i, seq].numpy()
        #         r_target_y = r_['target_y'][i, seq].numpy()
        #         print('agent: ',r_agent_x, r_agent_y, ' target: ', r_target_x, r_target_y)
        #
        #         img = img__[i, seq-N, :, :, :].squeeze().numpy()
        #         plt.imshow(img)
        #         plt.show()

        # Compute prediction error
        pred = predictor(img__, bs)  # (bs, t, 64)

        r_a_x, r_a_y, r_t_x, r_t_y = torch.unsqueeze(r_['agent_x'][:,N:], dim=0), torch.unsqueeze(r_['agent_y'][:,N:], dim=0), torch.unsqueeze(r_['target_x'][:,N:], dim=0), torch.unsqueeze(r_['target_y'][:,N:], dim=0)  # (1, bs, t)
        reward = torch.cat((r_a_x, r_a_y, r_t_x, r_t_y), 0).to(device)  # (4, bs, t)
        loss__, _ = predictor.loss(pred, reward, ep, step)
        l += loss__
        step += 1

        # Backpropagation
        optimizer.zero_grad()
        loss__.backward()
        optimizer.step()
     
    writer.add_scalar('loss/encoder', l/(seq_len/bs), ep)
    torch.save(predictor.state_dict(), "./model_weights/encoder/predictor"+str(ep)+".pt")
    loss_history.append(l/(seq_len/bs))

# ###check if correctly save weights
# predictor_test = Reward_Predictor(params)
# predictor_test.load_state_dict(torch.load("./model_weights/encoder/predictor4.pt"))
# train_param_list = []
# test_param_list = []
# ######check if all parameters are saved
# for name, param in predictor_test.named_parameters():
#    print(name)
#
# #######check if saved and loaded weights are the same
# for name, param in predictor.parameters():
#    print(type(param), param.size())
#    train_param_list.append(param)
#
# for param in predictor_test.parameters():
#    print(type(param), param.size())
#    test_param_list.append(param)
# i=0
# for param in test_param_list:
#     train_param = train_param_list[i]
#     if not torch.equal(param, train_param):
#         print("not equal!!", i)
#     i += 1

writer.flush()
print(loss_history)
