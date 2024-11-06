import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# data = {
#     'x': ['BoxingPunchingBag', 'IceDancing', 'PlayingSitar', 'BrushingTeeth', 'JumpingJack', 'Lunges', 'Kayaking', 'JugglingBalls', 'HeadMassage', 'Skijet', 'Knitting', 'Billiards', 'Biking', 'Nunchucks', 'Rafting', 'Hammering', 'FloorGymnastics', 'PlayingDaf', 'FrontCrawl', 'PushUps', 'BlowDryHair', 'Mixing', 'HammerThrow', 'RopeClimbing', 'ApplyLipstick', 'HighJump', 'ThrowDiscus', 'PlayingFlute', 'BalanceBeam', 'HandstandWalking'],
#     'y1': [42, 46, 42, 19, 32, 19, 31, 33, 39, 27, 34, 40, 36, 25, 24, 21, 24, 41, 22, 29, 29, 45, 38, 27, 30, 27, 32, 44, 29, 16],
#     'y2': [37, 44, 38, 15, 28, 21, 32, 38, 39, 18, 32, 40, 33, 28, 22, 16, 32, 39, 12, 27, 30, 40, 31, 25, 25, 25, 37, 41, 24, 14]
# }

x =  ['BoxingPunchingBag', 'IceDancing', 'PlayingSitar', 'BrushingTeeth', 'JumpingJack', 'Lunges', 'Kayaking', 'JugglingBalls', 'HeadMassage', 'Skijet', 'Knitting', 'Billiards', 'Biking', 'Nunchucks', 'Rafting', 'Hammering', 'FloorGymnastics', 'PlayingDaf', 'FrontCrawl', 'PushUps', 'BlowDryHair', 'Mixing', 'HammerThrow', 'RopeClimbing', 'ApplyLipstick', 'HighJump', 'ThrowDiscus', 'PlayingFlute', 'BalanceBeam', 'HandstandWalking'],
y1 =  [42, 46, 42, 19, 32, 19, 31, 33, 39, 27, 34, 40, 36, 25, 24, 21, 24, 41, 22, 29, 29, 45, 38, 27, 30, 27, 32, 44, 29, 16],
y2 = [37, 44, 38, 15, 28, 21, 32, 38, 39, 18, 32, 40, 33, 28, 22, 16, 32, 39, 12, 27, 30, 40, 31, 25, 25, 25, 37, 41, 24, 14]

df = pd.DataFrame(data)

sns.barplot(data=df, x='x', y='y1', color="skyblue", label="Group 1")
sns.barplot(data=df, x='x', y='y2', color="orange", label="Group 2")

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.show()
print(1)