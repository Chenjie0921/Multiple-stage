import time
import json
from collections import defaultdict

import torch
import torch.nn.functional as F

from utils import AverageMeter


def get_video_results(outputs, class_names, output_topk):
    sorted_scores, locs = torch.topk(outputs,
                                     k=min(output_topk, len(class_names)))

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i].item()],
            'score': sorted_scores[i].item()
        })

    return video_results


def inference(data_loader, model, MLP, querycoding, classifer, result_path, class_names, no_average,
              output_topk):
    print('inference')

    model.eval()
    MLP.eval()
    classifer.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    results = {'results': defaultdict(list)}

    end_time = time.time()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            video_ids, segments = zip(*targets)
            outputs_some, feature = model(inputs) #feature.shape B * 1 * N
            #TODO FOR Conv
            # for n in range(2, 4):
            #     feature = torch.squeeze(feature, 2)
            # features = MLP(feature)
            # features = torch.squeeze(features,2)
            # feature = torch.squeeze(feature,2)
            #TODO FOR Transformer
            for n in range(2,5):
                feature = torch.squeeze(feature,2)
            feature = torch.unsqueeze(feature,1)
            features = MLP(feature)
            part_1 = features[:, 0:1, :]  # query 1 shengchengdetezhen
            part_2 = features[:, 1:2, :]
            # feature_middle = feature + features
            # feature_end = MLP(feature_middle)
            part_1 = torch.squeeze(part_1,1)
            feature = torch.squeeze(feature,1)
            part_2 = torch.squeeze(part_2, 1)

            features_res = feature + part_1 + part_2
            # features_res = queryen(features_res)
            # features_res = torch.unsqueeze(features_res, 1)
            # features_res = torch.cat((features,feature),1)
            outputs = classifer(features_res)
            # outputs = torch.squeeze(outputs, 1)
            outputs = F.softmax(outputs, dim=1).cpu()

            for j in range(outputs.size(0)):
                results['results'][video_ids[j]].append({
                    'segment': segments[j],
                    'output': outputs[j]
                })

            batch_time.update(time.time() - end_time)
            end_time = time.time()
            # print('{}\t'.format(video_ids))
            print('[{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time))

    inference_results = {'results': {}}
    if not no_average:
        for video_id, video_results in results['results'].items():
            video_outputs = [
                segment_result['output'] for segment_result in video_results
            ]
            video_outputs = torch.stack(video_outputs)
            average_scores = torch.mean(video_outputs, dim=0)
            inference_results['results'][video_id] = get_video_results(
                average_scores, class_names, output_topk)
    else:
        for video_id, video_results in results['results'].items():
            inference_results['results'][video_id] = []
            for segment_result in video_results:
                segment = segment_result['segment']
                result = get_video_results(segment_result['output'],
                                           class_names, output_topk)
                inference_results['results'][video_id].append({
                    'segment': segment,
                    'result': result
                })

    with result_path.open('w') as f:
        json.dump(inference_results, f)
