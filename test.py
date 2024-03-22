import mitsuba as mi
import matplotlib.pyplot as plt
import matplotlib
import cv2
import numpy as np
import os


def visual_cmap_(array, map_info, save_path, save_fg=True):
    cmap = matplotlib.colormaps['hsv']
    vmin, vmax = map_info['vmin'], map_info['vmax']
    array = (array.clip(vmin, vmax) - vmin) / (vmax - vmin)
    rgba_map = cmap(array)
    rgb_map = rgba_map[..., :3]
    if save_fg:
        bgra_map = (np.concatenate([rgb_map[...,::-1]], -1) * 255).astype(np.uint8)
        # cv2.imwrite(save_path, bgra_map)
    return rgb_map

def cal_avr(image):
    gray=(mi.TensorXf(image)[:,:,0]+mi.TensorXf(image)[:,:,1]+mi.TensorXf(image)[:,:,2])/3
    return gray


# 设置渲染的 variant
mi.set_variant('cuda_spectral_polarized')

# 加载场景数据
scene = mi.load_file('scenes/cbox_pol.xml')
# scene = mi.load_file('real_env_ball.xml')

image = mi.render(scene, spp=512)
bitmap = mi.Bitmap(image, channel_names=['R', 'G', 'B'] + scene.integrator().aov_names())

channels = dict(bitmap.split())
print(channels.keys())

save_path = './data/cbox_1024'

# 取出 3 个 Stokes 分量
S0 = channels['S0'].convert(srgb_gamma=True) 
S1 = channels['S1'].convert(srgb_gamma=True)
S2 = channels['S2'].convert(srgb_gamma=True)

# 绘制图表显示
fig, ax = plt.subplots(ncols=5, figsize=(30, 5))
ax[0].imshow(S0)
ax[0].set_xlabel("S0: Intensity", size=14, weight='bold')
ax[1].imshow(S1)
ax[1].set_xlabel("S1: Horizontal vs. vertical", size=14, weight='bold')
ax[2].imshow(S2)
ax[2].set_xlabel("S2: Diagonal", size=14, weight='bold')

# 转换成 numpy array 并保存
S0 = np.array(S0)
cv2.imwrite(os.path.join(save_path, 'S0.png'), cv2.cvtColor(S0, cv2.COLOR_BGR2RGB) * 255)
S1 = np.array(S1)
cv2.imwrite(os.path.join(save_path, 'S1.png'), cv2.cvtColor(S1, cv2.COLOR_BGR2RGB) * 255)
S2 = np.array(S2)
cv2.imwrite(os.path.join(save_path, 'S2.png'), cv2.cvtColor(S2, cv2.COLOR_BGR2RGB) * 255)

# 计算 DoP:p 和 AoP:theta
# 先把 Stokes 参数转换成灰度图
S0_gray=cal_avr(channels['S0'])
S1_gray=cal_avr(channels['S1'])
S2_gray=cal_avr(channels['S2'])

p = np.clip(np.sqrt(S1_gray ** 2 + S2_gray ** 2) / (S0_gray + 1e-7), a_min=0, a_max=1)  # in [0, 1]
theta = np.arctan2(S2_gray, S1_gray) / 2  # in [-pi/2, pi/2]
# cmap 格式保存 AoP
theta = visual_cmap_(theta.astype(np.float32), map_info={'vmin': np.min(theta), 'vmax': np.max(theta)}, save_path='')

# 绘制图像
ax[3].imshow(p, cmap='viridis')
ax[3].set_xlabel("DoP", size=14, weight='bold')
ax[4].imshow(theta)
ax[4].set_xlabel("AoP", size=14, weight='bold')

# 保存
plt.savefig('./data/ball_512/ball.png')
cv2.imwrite('./data/ball_512/DoP.png', p * 255)
cv2.imwrite('./data/ball_512/AoP.png', theta * 255)

# 转换成不同角度的 pol 分量
I0 = (S0 + S1) / 2
cv2.imwrite(os.path.join(save_path, 'I0.png'), cv2.cvtColor(I0, cv2.COLOR_BGR2RGB) * 255)
I45 = (S0  + S2) / 2
cv2.imwrite(os.path.join(save_path, 'I45.png'), cv2.cvtColor(I45, cv2.COLOR_BGR2RGB) * 255)
I90 = (S0 - S1) / 2
cv2.imwrite(os.path.join(save_path, 'I90.png'), cv2.cvtColor(I90, cv2.COLOR_BGR2RGB) * 255)
I135 = (S0 + np.sqrt(2)/2 * S1 - np.sqrt(2)/2 * S2 ) / 2
cv2.imwrite(os.path.join(save_path, 'I135.png'), cv2.cvtColor(I135, cv2.COLOR_BGR2RGB) * 255)

# 再合并成一张图
# 再把各个极化分量的排列还原
polar_img = np.zeros((1024, 1024, 3))
polar_img[::2, ::2, :] = I90[::2, ::2, :]
polar_img[1::2, ::2, :] = I135[1::2, ::2, :]
polar_img[::2, 1::2, :] = I45[::2, 1::2, :]
polar_img[1::2, 1::2, :] = I0[1::2, 1::2, :]
cv2.imwrite(os.path.join(save_path, 'raw.png'), polar_img * 255)
