"""
Task 2:
Train encoder using 1 reward for later training a detached decoder
"""
from models import *
from load_data import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import gc

############ change parameters
T, N = 25, 25
# T, N = 5, 5
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params = {'T': T, 'N': N, 'K': 1, 'in_ch_mlp': N * 64, 'out_ch_mlp': 128, 'in_size_lstm': 64, 'device': device}
learning_rate = 0.0001
spec = AttrDict(
    resolution=64,
    max_seq_len=N + T,
    max_speed=0.05,  # total image range [0, 1]
    obj_size=0.2,  # size of objects, full images is 1.0
    shapes_per_traj=1,
    rewards=[VertPosReward], #HorPosReward #change1
)
epoch, bs, seq_len = 5, 32, 160000
# epoch, bs, seq_len = 3, 32, 64
task = "v"#change2
############

data_load = data_loader_task2(spec, batch_size=bs, seq_len=seq_len)
writer    = SummaryWriter(log_dir="./logs/logs_encoder_task2/"+task+"/")#change2
predictor = Reward_Predictor(params).to(device)
predictor.load_state_dict(torch.load("./model_weights/encoder/predictor3_task2_"+task+".pt"))#change3-epoch

optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
loss_history = []
for ep in range(epoch):
    l = 0.0
    step = 0
    for img_, r_ in data_load:  # r_shape 1x(bs, T+N) #img (bs, T+N, 1, 64, 64)
        img__ = img_[:, :N, :, :, :]  # img__ (bs, N, 1, 64, 64)
        # test loaded imgs and rewards
        for i in range(bs):
            print("batch: ", i)
            for seq in range(0, N):
                r = r_['vertical_position'][i, seq].numpy()
                print('agent: ',r)

                img = img__[i, seq, :, :, :].squeeze().numpy()
                plt.imshow(img)
                plt.show()

        # Compute prediction error
        pred = predictor(img__, bs)  # (bs, t, 64)

        reward = torch.unsqueeze(r_['vertical_position'][:, N:], dim=0).to(device)  # (1,bs,t) #change4
        loss__, _ = predictor.loss(pred, reward, ep, step)
        l += loss__
        step += 1

        # Backpropagation
        optimizer.zero_grad()
        loss__.backward()
        optimizer.step()

    writer.add_scalar('loss/encoder_task2_'+task, l / seq_len, ep)
    torch.save(predictor.state_dict(), "./model_weights/encoder/predictor" + str(ep) + "_task2_"+task+".pt")
    loss_history.append(l / seq_len)

    del img_
    del r_
    del img__
    del pred
    del reward
    del loss__
    del l
    del step
    gc.collect()

writer.flush()
print(loss_history)
