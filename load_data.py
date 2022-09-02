from torch.utils.data import DataLoader
from sprites_datagen.moving_sprites import *
from sprites_datagen.rewards import *

class ImageDataset(Dataset):
    def __init__(self, sprite_dataset, seq_len):
        super(ImageDataset, self).__init__()
        self.sprite_dataset = sprite_dataset
        self.seq_len = seq_len

    def __len__(self):
        return self.seq_len

    def __getitem__(self, idx):
        self.data     = self.sprite_dataset[0]
        training_data = self.data.images
        label_reward  = self.data.rewards
        return training_data, label_reward

def data_loader(spec, batch_size, seq_len):
    dataset_  = MovingSpriteDataset(spec)
    dataset   = ImageDataset(dataset_, seq_len)
    loader_   = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader_

def data_loader_task2(spec, batch_size, seq_len):
    dataset_  = MovingSpriteDataset_OneShape(spec)
    dataset   = ImageDataset(dataset_, seq_len)
    loader_   = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader_

if __name__ == '__main__':
    spec = AttrDict(
        resolution=64,
        max_seq_len=30,
        max_speed=0.05,  # total image range [0, 1]
        obj_size=0.2,  # size of objects, full images is 1.0
        shapes_per_traj=4,
        rewards=[AgentXReward, AgentYReward, TargetXReward, TargetYReward],
    )  # Ziyi tested spec

    data_load = data_loader(spec, batch_size=20, seq_len=30)
    step = 0
    import matplotlib.pyplot as plt
    for img_, r_ in data_load: #image_shape (20,30,3,64,64) r_shape 4x(20,30)
        step += 1
        print(img_.shape)
        # for bs in range(20):
        #     for seq in range(30):
        #         r_agent_x = r_['agent_x'][bs, seq].numpy()
        #         r_agent_y = r_['agent_y'][bs, seq].numpy()
        #         r_target_x = r_['target_x'][bs, seq].numpy()
        #         r_target_y = r_['target_y'][bs, seq].numpy()
        #         print('agent: ',r_agent_x, r_agent_y, ' target: ', r_target_x, r_target_y)
        #
        #         img__ = img_[bs, seq, :, :, :].numpy().transpose(1, 2, 0)
        #         plt.imshow(img__)
        #         plt.show()

    print(step)
