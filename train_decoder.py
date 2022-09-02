"""
Task 2:
Train a detached decoder using a pre-trained encoder (1 reward)
"""
from models import *
from load_data import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import imageio as io
############ change parameters
T, N = 25, 25
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
epoch, bs, seq_len = 30, 32, 160000
# epoch, bs, seq_len = 5, 32, 64
task = "v"

#### train decoder
decoder_ = Train_Decoder(T=T, ch_linear=params['out_ch_mlp']).to(device)

#### load pretrained encoder
predictor = Reward_Predictor(params).to(device)
predictor.load_state_dict(torch.load("./model_weights/encoder/predictor9_task2_v.pt"))
for param in predictor.parameters():
    param.requires_grad = False
# for name, param in predictor.named_parameters():
#    print(name)

data_load = data_loader_task2(spec, batch_size=bs, seq_len=seq_len)
writer    = SummaryWriter(log_dir="./logs/logs_decoder/"+task+"/")
optimizer = torch.optim.Adam(decoder_.parameters(), lr=learning_rate)
loss_ = nn.MSELoss()
loss_history = []

for ep in range(epoch):
    l = 0.0
    for img_, _ in data_load:
        img_  = img_.to(device)
        img__ = img_[:, :N, :, :, :]  # img__ (bs, N, 1, 64, 64)
        encoded_img = predictor(img__, bs=bs)  # (bs, t, 128)

        de_img = decoder_(encoded_img)  # (bs, t, 1, 64, 64)
        loss__ = loss_(de_img, img_[:, N:, :, :, :])
        l += loss__

        # Backpropagation
        optimizer.zero_grad()
        loss__.backward()
        optimizer.step()

    de_img_ = de_img.cpu().detach().numpy()  # value range: (-1, 1) shape: (bs, t, 1, 64, 64)
    # img_flat = de_img_.flatten()
    # max_value = np.max(img_flat)
    # min_value = np.min(img_flat)
    # print(min_value, max_value)

    # ######check if save correctly
    # temp_ = de_img_[0:1, :, :, :, :].squeeze(0)#(t,1,64,64)
    # temp_ = (temp_ + 1.0)/2.
    # for index in range(T):
    #     tt = temp_[index:index+1, :, :, :].squeeze() #(64,64)
    #     too_small = (tt > 0.1) & (tt < 0.9)
    #     tt[too_small] = 1.0
    #     tt = tt[None, :].repeat(3, axis=0).transpose(1, 2, 0)
    #     plt.imshow(tt)
    #     plt.show()
    #################

    img = make_image_seq_strip([de_img_[0:1, :, :, :, :].repeat(3, axis=2).astype(np.float32)], sep_val=1.0)
    img = img * 255.0
    img = img.astype(np.uint8)
    io.imwrite("./logs/logs_decoder/" + task + "/pred" + str(ep) + ".png", img[0].transpose(1, 2, 0))

    de_img_ = img_[:, N:, :, :, :].cpu().detach().numpy()  # range: (-1,1) shape: (bs, t, 1, 64, 64)
    # test label
    # for seq in range(T):
    #     img = de_img_[0:1, seq:seq+1, :, :, :].repeat(3, axis=2).squeeze().astype(np.float32)
    #     plt.imshow(img.transpose(1, 2, 0))
    #     plt.show()
    img = make_image_seq_strip([de_img_[0:1, :, :, :, :].repeat(3, axis=2).astype(np.float32)], sep_val=1.0)
    img = img*255.0
    img = img.astype(np.uint8)
    io.imwrite("./logs/logs_decoder/" + task + "/label" + str(ep) + ".png",  img[0].transpose(1, 2, 0))

    writer.add_scalar('loss/decoder_'+task, l / (seq_len/bs), ep)
    torch.save(decoder_.state_dict(), "./model_weights/decoder/decoder" + str(ep) + "_"+task+".pt")
    loss_history.append(l / (seq_len/bs))

writer.flush()
print(loss_history)

