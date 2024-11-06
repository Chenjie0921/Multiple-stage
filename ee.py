# import torch
#
# device = torch.device('cuda:0')
# MLP = torch.nn.Sequential(
#         torch.nn.Linear(in_features=2048, out_features=2048, bias=True),
#         torch.nn.ReLU(inplace=True),
#         torch.nn.Linear(in_features=2048, out_features=2048, bias=True)
#     )
# MLP.to(device)
# tensor1 = torch.ones((64,2048,1,1,1))
# for i in range(2,5):
#     tensor1 = torch.squeeze(tensor1, 2)
#
# output = MLP(tensor1)
# print(output)

# root_path = '/media/lz/lz2/ucf_101'
import math
import random
frame_indices = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
if len(frame_indices) > 16:
    interval = math.floor(len(frame_indices)-1 / 15)
    start_range = frame_indices[-1] - (frame_indices[0] + 15 * interval)
    print(start_range)
    if start_range == 0:
        start_frame = frame_indices[0]
    else:
        start_frame = random.randint(frame_indices[0], frame_indices[0] + start_range)
    frame_indices = [start_frame + x * interval for x in range(16)]
else:
    frame_indices = t(frame_indices)