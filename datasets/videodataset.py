import json
from pathlib import Path

import torch
import torch.utils.data as data

from .loader import VideoLoader


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_database(data, subset, root_path, video_path_formatter,ratio):
    video_ids = []
    video_paths = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        #TODO traning data or all data
        if this_subset == subset:
            video_ids.append(key)
            # TODO  set ratio of traing data
            value_tmp = value['annotations']
            segment_tmp = value_tmp['segment']
            segment_end = segment_tmp[1]
            for m in range(len(segment_tmp)):
                if m == 1:
                    segment_tmp[m] = int(segment_tmp[m]*ratio)#前面部分帧
                    # segment_tmp[m+1] = int((segment_end-segment_tmp[1])*0.5)
                    segment_middle = (int((segment_end-segment_tmp[1])*0.5)+segment_tmp[m])#后面中间帧
            segment_start = segment_tmp[1]
            segment_tmp_2 = [segment_start, segment_middle]
            segment_tmp_3 = [segment_middle, segment_end]
            value_tmp['segment'] = segment_tmp
            value_tmp['segment2'] = segment_tmp_2
            value_tmp['segment3'] = segment_tmp_3
            annotations.append(value_tmp)
            # annotations.append(value['annotations'])
            if 'video_path' in value:
                video_paths.append(Path(value['video_path']))
            else:
                label = value['annotations']['label']
                video_paths.append(video_path_formatter(root_path, label, key))

    return video_ids, video_paths, annotations


class VideoDataset(data.Dataset):

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 video_loader=None,
                 video_path_formatter=(lambda root_path, label, video_id:
                                       root_path / label / video_id),
                 image_name_formatter=lambda x: f'image_{x:05d}.jpg',
                 target_type='label',
                 ratio = 1):
        self.data, self.class_names = self.__make_dataset(
            root_path, annotation_path, subset, video_path_formatter,ratio)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform

        if video_loader is None:
            self.loader = VideoLoader(image_name_formatter)
        else:
            self.loader = video_loader

        self.target_type = target_type
    def __make_dataset(self, root_path, annotation_path, subset,
                       video_path_formatter,ratio):
        with annotation_path.open('r') as f:
            data = json.load(f)
        video_ids, video_paths, annotations = get_database(
            data, subset, root_path, video_path_formatter,ratio)
        class_to_idx = get_class_labels(data)
        idx_to_class = {}
        for name, label in class_to_idx.items():
            idx_to_class[label] = name

        n_videos = len(video_ids)
        dataset = []
        for i in range(n_videos):
            if i % (n_videos // 5) == 0:
                print('dataset loading [{}/{}]'.format(i, len(video_ids)))

            if 'label' in annotations[i]:
                label = annotations[i]['label']
                label_id = class_to_idx[label]
            else:
                label = 'test'
                label_id = -1

            video_path = video_paths[i]
            if not video_path.exists():
                continue

            segment = annotations[i]['segment']
            segment_2 = annotations[i]['segment2']
            segment_3 = annotations[i]['segment3']
            if segment[1] == 1:
                continue

            frame_indices = list(range(segment[0], segment[1]))
            frame_indices_2 = list(range(segment_2[0],segment_2[1]))
            frame_indices_3 = list(range(segment_3[0],segment_3[1]))
            sample = {
                'video': video_path,
                'segment': segment,
                'frame_indices': frame_indices,
                'frame_indices_2': frame_indices_2,
                'frame_indices_3': frame_indices_3,
                'video_id': video_ids[i],
                'label': label_id
            }
            dataset.append(sample)

        return dataset, idx_to_class

    def __loading(self, path, frame_indices):
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip

    def __getitem__(self, index):
        path = self.data[index]['video']
        if isinstance(self.target_type, list):
            target = [self.data[index][t] for t in self.target_type]
        else:
            target = self.data[index][self.target_type]

        frame_indices = self.data[index]['frame_indices']
        frame_indices_2 = self.data[index]['frame_indices_2']
        frame_indices_3 = self.data[index]['frame_indices_3']
        # print("before aver sample:{}\n{}".format(frame_indices,frame_indices_2))
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
            frame_indices_2 = self.temporal_transform(frame_indices_2)
            frame_indices_3 = self.temporal_transform(frame_indices_3)
        # print("after aver sample:{}\n{}".format(frame_indices, frame_indices_2))
        clip = self.__loading(path, frame_indices)
        clip2 = self.__loading(path,frame_indices_2)
        clip3 = self.__loading(path, frame_indices_3)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, clip2, clip3, target

    def __len__(self):
        return len(self.data)