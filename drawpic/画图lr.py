#画图
import matplotlib.pyplot as plt
import numpy as np

draw_loss1 = np.load("draw_loss_lr=0.01.npy")
draw_loss2 = np.load("draw_loss_lr=0.03.npy")
draw_loss3 = np.load("draw_loss_lr=0.003.npy")
draw_loss4 = np.load("draw_loss_lr=0.1.npy")
draw_loss5 = np.load("draw_loss_lr=0.001.npy")

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(draw_loss1,  label='lr=0.01')
ax.plot(draw_loss2,  label='lr=0.03')
ax.plot(draw_loss3,  label='lr=0.003')
ax.plot(draw_loss4,  label='lr=0.1')
ax.plot(draw_loss5,  label='lr=0.001')

ax.legend()
ax.set_ylabel('Training Loss')
ax.set_xlabel('epoch')
plt.xticks([]) # 隐藏x轴刻度
plt.show()

