#画图
import matplotlib.pyplot as plt
import numpy as np

draw_loss1 = np.load("draw_loss_batchsize=64.npy")
draw_loss2 = np.load("draw_loss_batchsize=128.npy")
draw_loss3 = np.load("draw_loss_batchsize=256.npy")
draw_loss4 = np.load("draw_loss_batchsize=512.npy")
draw_loss5 = np.load("draw_loss_batchsize=32.npy")

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(draw_loss1, label='batchsize=64')
ax.plot(draw_loss2, label='batchsize=128')
ax.plot(draw_loss3, label='batchsize=256')
ax.plot(draw_loss4, label='batchsize=512')
ax.plot(draw_loss5, label='batchsize=32')
ax.legend()
ax.set_ylabel('Training Loss')
ax.set_xlabel('epoch')
plt.xticks([]) # 隐藏x轴刻度
plt.show()
