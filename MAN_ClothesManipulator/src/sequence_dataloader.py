"""
 The "get_next_pair_sample" method return a couple of samples at distance less or equal to "max_distance_between_pair"
"""

import numpy as np
import torchvision.transforms as transforms
import constants as C
from dataloader import Data
from utils import cut_index
from utils import listify_manip
from numpy.linalg import norm
from tqdm import tqdm
import time
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence



def normalize(dis_feat, normalization):
    if normalization:
        dis_feat_normalized = (dis_feat - np.min(dis_feat)) / (np.max(dis_feat) - np.min(dis_feat))
        return dis_feat_normalized
    return dis_feat

def pad_collate(batch):
    (xx, yy, zz) = zip(*batch)
    # x_lens = [len(x) for x in xx]
    # y_lens = [len(y) for y in yy]
    z_lens = [len(z) for z in zz]

    # xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    # yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
    xx_batch = torch.stack(xx)
    yy_batch = torch.stack(yy)
    zz_pad = pad_sequence(zz, batch_first=True, padding_value=0)

    return xx_batch, yy_batch, zz_pad, z_lens

class SequenceDataset(Dataset):

    def __init__(self, file_root, img_root_path, dis_feat_root, mode='train', normalized=False,
                 recompute_distances=False, fix_seq_len=None, max_seq_len=8):
        self.mode = mode  # 'train' or 'test'
        self.fix_seq_len = fix_seq_len
        self.max_seq_len = max_seq_len
        if self.mode == 'train':
            self.data = Data(file_root, img_root_path,
                             transforms.Compose([
                                 transforms.Resize((C.TRAIN_INIT_IMAGE_SIZE, C.TRAIN_INIT_IMAGE_SIZE)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.CenterCrop(C.TARGET_IMAGE_SIZE),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                             ]), self.mode)
            if normalized:
                dis_feat_root = format(dis_feat_root + "/feat_train_Norm.npy")
            else:
                dis_feat_root = format(dis_feat_root + "/feat_train_senzaNorm.npy")
        elif self.mode == 'test':
            self.data = Data(file_root, img_root_path,
                             transforms.Compose([
                                 transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                             ]), self.mode)
            if normalized:
                dis_feat_root = format(dis_feat_root + "/feat_test_Norm.npy")
            else:
                dis_feat_root = format(dis_feat_root + "/feat_test_senzaNorm.npy")

        self.dis_feat_file = np.load(dis_feat_root)

        if recompute_distances:
            self.distance_dict = self.build_distance_dict(self.data.label_data)
            np.save("distance_dict_new.npy", self.distance_dict)
        else:
            self.distance_dict = np.load('distance_dict_new.npy', allow_pickle=True)[None][0]
        print('Dataset ready')

    def __len__(self):
        return 3200 # a random length since our dataset is virtually infinite
    
    def __getitem__(self, idx):
        if self.fix_seq_len:
            sample_distance = np.random.randint(1, self.fix_seq_len)
        else:
            sample_distance = np.random.randint(1, self.max_seq_len+1)
        q_id, t_id = self.find_couple_fast(sample_distance)
        q_dis_feat, t_dis_feat = self.get_disentangled_features(q_id, t_id)
        control, manip_vectors = self.get_manipulation_vectors(q_id, t_id, sample_distance)  # input 151
        # print(manip_vectors)
        assert control==True, "manipulation exceeds distance"

        target = torch.reshape(torch.from_numpy(t_dis_feat), (12, 340))
        query = torch.reshape(torch.from_numpy(q_dis_feat), (12, 340))
        net_inputs = torch.reshape(torch.from_numpy(manip_vectors), (len(manip_vectors), 151)).float()
        return target, query, net_inputs

    def get_next_pair_sample(self, max_distance_between_pair):
        q, t = self.find_couple_fast(max_distance_between_pair)
        return q, t
    
    def compare_labels(self, id1, id2):
        cut_index_np = np.array(cut_index)
        l1 = np.split(self.data.label_data[id1], cut_index_np[:-1,1])
        l2 = np.split(self.data.label_data[id2], cut_index_np[:-1,1])

        for q,w in zip(l1,l2):
            print(q)
            print(w)

    
    def build_distance_dict(self, labels):
        n_labels = labels.shape[0]
        cut_index_np = np.array(cut_index)
        #distance_matrix = np.ones((n_labels, n_labels), dtype=np.uint8)*-1
        distance_dict = {}
        split_labels = np.split(labels, cut_index_np[:-1,1], axis=1)
        attribute_signatures = np.stack([np.all(x==0, axis=1) for x in split_labels], axis=1)
        
        for query_id in tqdm(range(len(labels))):
            query_labels = np.split(labels[query_id], cut_index_np[:-1, 1])
            valid_attributes = np.where([sum(x) for x in query_labels])[0]
            invalid_attributes = np.setdiff1d(range(len(cut_index_np)), valid_attributes)

            #valid_ids_mask = np.prod([np.sum(split_labels[x],1) == 0 for x in invalid_attributes], axis=0)# ids of garments that share the same types of attributes with the query
            valid_ids_mask = np.all(np.logical_not(attribute_signatures ^ attribute_signatures[query_id]),axis=1)
            valid_ids = np.where(valid_ids_mask)[0]
            valid_ids = valid_ids[valid_ids > query_id] # avoid repetitions

            valid_changes = [np.sum(np.abs(query_labels[x] - split_labels[x]), axis=1)==2 for x in valid_attributes]
            num_changes = np.sum(valid_changes,0)

            unique_changes = np.unique(num_changes[valid_ids])
            for uc in unique_changes:
                if uc == 0:
                    continue # skipping 0 changes
                if uc not in distance_dict:
                    distance_dict[uc] = {}
                target_ids = valid_ids[np.where(num_changes[valid_ids] == uc)[0]]
                # cur_pairs = [(query_id, t) for t in target_ids if t>query_id]
                # distance_dict[uc].extend(cur_pairs)
                distance_dict[uc][query_id] = target_ids
        return distance_dict

    
    def find_couple_fast(self, distance_between_pair):
        query = np.random.choice(list(self.distance_dict[distance_between_pair].keys()))
        target = np.random.choice(self.distance_dict[distance_between_pair][query])
        if np.random.random() > 0.5:
            query, target = target, query # swap query and target randomly
        return query, target

    def get_disentangled_features(self, q_id, t_id):
        q_dis_feat = self.dis_feat_file[q_id]
        t_dis_feat = self.dis_feat_file[t_id]
        return q_dis_feat, t_dis_feat

    def get_one_hot_labels(self, q_id, t_id):
        labels = self.data.label_data
        return labels[q_id], labels[t_id]

    def get_on_hot_label(self, id):
        labels = self.data.label_data
        return labels[id]

    def get_images(self, q_id, t_id):
        return self.get_image(q_id), self.get_image(t_id)

    def get_image(self, id):
        return self.data.__getitem__(id)

    def get_manipulation_vectors(self, q_id, t_id, max_distance_between_pair):
        q_label, t_label = self.get_one_hot_labels(q_id, t_id)
        multi_manip = np.subtract(t_label, q_label)
        multi_manip = listify_manip(multi_manip)
        if len(multi_manip) <= max_distance_between_pair:
            return True, multi_manip
        return False, multi_manip

    def cosine_similarity(self, dis_feat, target_feat):
        return np.dot(target_feat, dis_feat) / (norm(target_feat) * norm(dis_feat))

    def find_x_ids_images_more_similiar(self, dis_feat, x):
        x_ids = []
        for j in range(x):
            max = -2
            for i in range(self.data.__len__()):
                if i not in x_ids:
                    curr_cs = self.cosine_similarity(dis_feat, self.dis_feat_file[i])
                    if curr_cs > max:
                        max = curr_cs
                        best_id = i
            x_ids.append(best_id)
        return x_ids
