import json
import argparse
from pathlib import Path


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def load_ground_truth(ground_truth_path, subset):
    with ground_truth_path.open('r') as f:
        data = json.load(f)

    class_labels_map = get_class_labels(data)

    ground_truth = []
    for video_id, v in data['database'].items():
        if subset != v['subset']:
            continue
        this_label = v['annotations']['label']
        ground_truth.append((video_id, class_labels_map[this_label]))

    return ground_truth, class_labels_map


def load_result(result_path, top_k, class_labels_map):
    with result_path.open('r') as f:
        data = json.load(f)

    result = {}
    for video_id, v in data['results'].items():
        labels_and_scores = []
        for this_result in v:
            label = class_labels_map[this_result['label']]
            score = this_result['score']
            labels_and_scores.append((label, score))
        labels_and_scores.sort(key=lambda x: x[1], reverse=True)
        result[video_id] = list(zip(*labels_and_scores[:top_k]))[0]
    return result


def remove_nonexistent_ground_truth(ground_truth, result):
    exist_ground_truth = [line for line in ground_truth if line[0] in result]

    return exist_ground_truth


def evaluate(ground_truth_path, result_path, subset, top_k, ignore):
    print('load ground truth')
    ground_truth, class_labels_map = load_ground_truth(ground_truth_path,
                                                       subset)
    print('number of ground truth: {}'.format(len(ground_truth)))

    print('load result')
    result = load_result(result_path, top_k, class_labels_map)
    print('number of result: {}'.format(len(result)))

    n_ground_truth = len(ground_truth)
    ground_truth = remove_nonexistent_ground_truth(ground_truth, result)
    if ignore:
        n_ground_truth = len(ground_truth)

    print('calculate top-{} accuracy'.format(top_k))
    # dict_class = {}
    # for j in range(0,51):
    #     count = 0
    #     for i in ground_truth:
    #         if i[1]==j:
    #             count += 1
    #             dict_class.update([(j,count)])

    # cal=1
    # if cal == 1:
    #     for line in ground_truth:
    #         if line[1] in result[line[0]]:
    # correct_cate = {}
    # for j in range(0,51):
    #     count=0
    #     for line in ground_truth:
    #         if line[1]==j and line[1] in result[line[0]]:
    #             count += 1
    #     correct_cate.update([(j,count)])
    #     # correct_cate = [1 if line[1] in result[line[0]] else 0 for line in ground_truth]
    correct = [1 if line[1] in result[line[0]] else 0 for line in ground_truth]
    accuracy = sum(correct) / n_ground_truth

    print('top-{} accuracy: {}'.format(top_k, accuracy))
    # accuracy=0
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truth_path', default='/home/lz/3D-ResNets-PyTorch/data/hmdb51_1.json',type=Path)
    parser.add_argument('--result_path',default='/media/lz/lz2/Action_usual_result/results/50%57.58/val_50.json', type=Path)
    parser.add_argument('-k', default=1,type=int)
    parser.add_argument('--subset', type=str, default='validation')
    parser.add_argument('--save', action='store_true')
    parser.add_argument(
        '--ignore',
        default=True,
        action='store_true',
        help='ignore nonexistent videos in result')

    args = parser.parse_args()

    accuracy = evaluate(args.ground_truth_path, args.result_path, args.subset,
                        args.k, args.ignore)

    if args.save:
        with (args.result_path.parent / 'top{}.txt'.format(
                args.k)).open('w') as f:
            f.write(str(accuracy))
