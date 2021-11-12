#画图
import matplotlib.pyplot as plt
import numpy as np

draw_loss1 = np.load("draw_loss_momentum=0.5.npy")
draw_loss2 = np.load("draw_loss_momentum=0.9.npy")
draw_loss3 = np.load("draw_loss_momentum=0.99.npy")

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(draw_loss1,  label='momentum=0.5')
ax.plot(draw_loss2,  label='momentum=0.9')
ax.plot(draw_loss3,  label='momentum=0.99')

ax.legend()
ax.set_ylabel('Training Loss')
ax.set_xlabel('epoch')
plt.xticks([]) # 隐藏x轴刻度
plt.show()
