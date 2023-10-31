""" Training for the Clothes Manipulator in Neural Turing Machines. """

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from optimized_data_supplier import OptimizedDataSupplier
from sequence_dataloader import SequenceDataset, pad_collate
from data_supplier import DataSupplier
from clothesManipulatorTask import ClothesManipulatorModelTraining, \
    ClothesManipulatroTaskParams
from torch.utils.data import DataLoader
from tqdm import tqdm


TASKS = {
    'clothesManipulatorTask': (ClothesManipulatorModelTraining, ClothesManipulatroTaskParams),
}

# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("DEVICE:", device)

SAVING_PATH = 'nets/last.pth'


def init_model(args):
    model_cls, params_cls = TASKS[args]
    params = params_cls()

    model = model_cls(params=params)  # Encapsuleted
    return model


def clip_grads(net):
    """ Gradient clipping to the range [10, 10]. """
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)


def train_batch(net, criterion, optimizer, data_supplier):
    """ Trains a single batch. """
    optimizer.zero_grad()
    sample_distance = np.random.randint(1, 8)
    #q_id, t_id = data_supplier.get_next_pair_sample(sample_distance)
    q_id, t_id = data_supplier.find_couple_fast(sample_distance)
    q_dis_feat, t_dis_feat = data_supplier.get_disentangled_features(q_id, t_id)
    control, manip_vectors = data_supplier.get_manipulation_vectors(q_id, t_id, sample_distance)  # input 151

    target = torch.reshape(torch.from_numpy(t_dis_feat), (12, 1, 340))
    query = torch.reshape(torch.from_numpy(q_dis_feat), (1, 12, 340))
    net_inputs = torch.reshape(torch.from_numpy(manip_vectors), (len(manip_vectors), 1, 151)).float()

    inp_seq_len = net_inputs.size(0)
    outp_seq_len, batch_size, _ = target.size()

    # New sequence
    net.init_sequence_query(batch_size, query)

    # Feed the sequence
    for i in range(inp_seq_len):
        net(net_inputs[i])  # Quando chiamo net() -> chiamo il metodo forward

    net_memory = net.get_memory()
    memory = torch.reshape(net_memory.memory, (12, 1, 340))

    loss = criterion(memory, target)
    loss.backward()
    clip_grads(net)
    optimizer.step()
    return loss.item() / batch_size


def train_model(model, data_supplier, optimized_data_supplier):
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter("tensorboard/runs")

    epochs = 4001
    num_batches = 200

    print("Start Training!")

    for i in range(epochs):
        print('***** Running Epoch: ', i, ' *****')
        losses = []
        for j in range(num_batches):
            if j % 2 == 0:
                loss = train_batch(model.net, model.criterion, model.optimizer, data_supplier, 3)
                losses += [loss]
            else:
                loss = train_batch(model.net, model.criterion, model.optimizer, optimized_data_supplier, 1)
                losses += [loss]
        writer.add_scalars('Loss', {'trainset': np.array(losses).mean()}, i + 1)

        if i % 500 == 0:
            torch.save(model.net.state_dict(), f'nets/model_{i}.pth')
    writer.close()  # tensorboard --logdir=../tensorboard/runs

    print("Done Training!")

def train_model_with_dataloader(model, dataloader):
    writer = SummaryWriter("tensorboard/runs")
    epochs = 1000
    for e in range(epochs):
        print(f'Epoch {e}')
        for batch_query, batch_target, batch_manipulations, manip_lengths in tqdm(dataloader):
            bs = batch_query.shape[0]
            inp_seq_len = batch_manipulations.size(1)

            # New sequence
            model.net.init_sequence_query(bs, batch_query)

            # Feed the sequence
            for i in range(inp_seq_len):
                model.net(batch_manipulations[:,i,...])  # Quando chiamo net() -> chiamo il metodo forward

            net_memory = model.net.get_memory()
            memory = torch.reshape(net_memory.memory, (bs, 12, 340))

            loss = model.criterion(memory, batch_target)
            loss.backward()
            clip_grads(model.net)
            model.optimizer.step()
            loss_per_sample = loss.item() / bs
        writer.add_scalars('Loss', {'trainset': np.array(loss_per_sample).mean()}, e)
        if e % 20 == 0:
            torch.save(model.net.state_dict(), f'nets/fixed_dataloader_model_{i}.pth')
    writer.close()  # tensorboard --logdir=../tensorboard/runs




def main():
    # Initialize arguments
    file_root = 'splits/Shopping100k'
    img_root_path = 'Shopping100k/Images'
    dis_feat_root = 'disentangledFeaturesExtractor'
    mode = 'train'
    task = "clothesManipulatorTask"

    # Initialize the model
    model = init_model(task)  # MODEL
    #data_supplier = DataSupplier(file_root, img_root_path, dis_feat_root, mode, False)
    #optimized_data_supplier = OptimizedDataSupplier(file_root, img_root_path, dis_feat_root, mode, False)

    sequence_dataset = SequenceDataset(file_root, img_root_path, dis_feat_root, mode, False, recompute_distances=True)
    dataloader = DataLoader(sequence_dataset, batch_size=128, shuffle=False, collate_fn=pad_collate)

    # path = '../nets/last.pth'
    # model.net.load_state_dict(torch.load(path))

    #train_model(model, data_supplier, optimized_data_supplier)  # TRAIN

    train_model_with_dataloader(model, dataloader)


if __name__ == '__main__':
    main()
