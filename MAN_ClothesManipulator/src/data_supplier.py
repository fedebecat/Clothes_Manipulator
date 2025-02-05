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


def normalize(dis_feat, normalization):
    if normalization:
        dis_feat_normalized = (dis_feat - np.min(dis_feat)) / (np.max(dis_feat) - np.min(dis_feat))
        return dis_feat_normalized
    return dis_feat


class DataSupplier:

    def __init__(self, file_root, img_root_path, dis_feat_root, mode='train', normalized=False):
        self.mode = mode  # 'train' or 'test'

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

        if False:
            self.distance_dict = self.build_distance_dict(labels)
            np.save("distance_dict.npy", self.distance_dict)
        else:
            self.distance_dict = np.load('distance_dict.npy', allow_pickle=True)[None][0]
        print(len(self.distance_dict))

    def get_next_pair_sample(self, max_distance_between_pair):
        #st = time.time()
        #q, t = self.find_couple(labels, max_distance_between_pair)  # query = start image, target = wanted image
        #et = time.time()
        #st_fast = time.time()
        q, t = self.find_couple_fast(max_distance_between_pair)
        #et_fast = time.time()
        #print(f'original time: {et - st}, new time: {et_fast - st_fast}, speedup: {(et - st)/(et_fast - st_fast)}')
        return q, t
    
    def build_distance_dict(self, labels):
        n_labels = labels.shape[0]
        cut_index_np = np.array(cut_index)
        #distance_matrix = np.ones((n_labels, n_labels), dtype=np.uint8)*-1
        distance_dict = {}
        split_labels = np.split(labels, cut_index_np[:-1,1], axis=1)
        for query_id in tqdm(range(len(labels))):
            query_labels = np.split(labels[query_id], cut_index_np[:-1, 1])
            #query_labels = [x[query_id] for x in split_labels]
            valid_attributes = np.where([sum(x) for x in query_labels])[0]
            invalid_attributes = np.setdiff1d(range(len(cut_index_np)), valid_attributes)

            valid_ids_mask = np.prod([np.sum(split_labels[x],1) == 0 for x in invalid_attributes], axis=0)
            #valid_ids = np.where(valid_ids_mask)[0] # ids of garments that share the same types of attributes with the query
            valid_ids = np.where(np.where(valid_ids_mask)[0] > query_id)[0] # avoid repetitions

            valid_changes = [np.sum(np.abs(query_labels[x] - split_labels[x]), axis=1)==2 for x in valid_attributes]
            num_changes = np.sum(valid_changes,0)
            #distance_matrix[query_id,valid_ids] = num_changes[valid_ids]

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
        #num_queries = len(self.distance_dict[distance_between_pair])
        query = np.random.choice(list(self.distance_dict[distance_between_pair].keys()))
        #num_targets = len(self.distance_dict[distance_between_pair][query])
        target = np.random.choice(self.distance_dict[distance_between_pair][query])

        #query = self.distance_dict[distance_between_pair][query]
        #target = self.distance_dict[distance_between_pair][query][target_id]
        if np.random.random() > 0.5:
            query, target = target, query # swap query and target randomly
        return query, target


    def find_couple(self, labels, max_distance_between_pair):
        n_labels = labels.shape[0]
        cut_index_np = np.array(cut_index)

        found_q = -1
        found_t = -1

        q_indexes = np.arange(n_labels)
        np.random.shuffle(q_indexes)

        for q_id in q_indexes:
            t_indexes = np.arange(n_labels)  # [0, 1, ..., n_labels]
            np.random.shuffle(t_indexes)
            for t_id in t_indexes:
                if not np.array_equal(labels[q_id], labels[t_id]):
                    if not self.too_much_distance(labels[q_id], labels[t_id], cut_index_np, max_distance_between_pair):
                        found_q = q_id
                        found_t = t_id
                        return found_q, found_t  # Contain id of two images with <=N distance

        return found_q, found_t  # return -1, -1

    def too_much_distance(self, q_lbl, t_lbl, cut_index_np, max_distance_between_pair):
        multi_manip = np.subtract(q_lbl, t_lbl)
        distance = 0
        for ci in cut_index_np:
            if np.any(multi_manip[ci[0]:ci[1]]):  # return false if [0, 0, ..., 0], true if [-1, 0, 1, 0 ..., 0]
                distance += 1
        if distance > max_distance_between_pair:
            return True
        else:
            return False

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
