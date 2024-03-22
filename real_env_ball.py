import mitsuba as mi

import cv2
import numpy as np

import xml.etree.ElementTree as ET

def lookat_from_rot(theta, phi, r):
    # phi wrt xz plane
    # theta wrt positive z axis
    origin = np.array([r*np.cos(phi)*np.sin(theta) ,
                       r*np.sin(phi),
                       r*np.cos(phi)*np.cos(theta)])
    print(origin)
    target = np.array([0.,0.,0.])
    up = np.array([0.,1.,0.])
    lookat_mtx =[origin,up]
    return lookat_mtx

def edit_xml(xml_path,origin,up):
    tree = ET.ElementTree(file=xml_path)
    root = tree.getroot()
    print(origin)
    root[4][4][0].set('origin','{},{},{}'.format(origin[0],origin[1],origin[2]))
    root[4][4][0].set('up','{},{},{}'.format(up[0],up[1],up[2]))

    tree.write(xml_path)

def cal_avr(image):
    gray=(mi.TensorXf(image)[:,:,0]+mi.TensorXf(image)[:,:,1]+mi.TensorXf(image)[:,:,2])/3
    return gray

phi_range = np.linspace(0.,45.,3)+1.
theta_range = np.linspace(-180.,170,12)+1.
phi_grid, theta_grid = np.meshgrid(phi_range,theta_range)
cam_depth = 3.0

# origin_all,up_all=cpm.cpm()
xml_path='real_env_ball.xml'
savepath ='./'
mi.set_variant('cuda_spectral_polarized')

for num in range(36):
    phi = phi_grid.reshape(-1)[num]
    theta = theta_grid.reshape(-1)[num]
    cam_mtx = lookat_from_rot(theta*np.pi/180., phi*np.pi/180., cam_depth)

    edit_xml(xml_path,cam_mtx[0],cam_mtx[1])
    scene = mi.load_file(xml_path)
    image = mi.render(scene, spp=512)
    # print(scene.integrator().aov_names())
    bitmap = mi.Bitmap(image, channel_names=['R', 'G', 'B'] + scene.integrator().aov_names())
    # bitmap = mi.Bitmap(image, channel_names=['R', 'G', 'B'] + scene.integrator().aov_names())

    #bitmap.write('real_env_sphere.exr')


    channels = dict(bitmap.split())
    print(channels.keys())
    ########## save normal
    # normal = channels['normal']
    # np.save('./data/bunny0423/bunny_n/{:04d}.npy'.format(num),np.array(normal))

    ########  save rgb
    I=np.array(channels['S0'].convert(srgb_gamma=True))
    cv2.imwrite('./data/{:04d}.png'.format(num),cv2.cvtColor(I,cv2.COLOR_BGR2RGB)*255)

    ######## calculate dop aop
    # S0_gray=cal_avr(channels['stokes.S0'])
    # S1_gray=cal_avr(channels['stokes.S1'])
    # S2_gray=cal_avr(channels['stokes.S2'])
    # p = np.clip(np.sqrt(S1_gray ** 2 + S2_gray ** 2) / (S0_gray + 1e-7), a_min=0, a_max=1)  # in [0, 1]
    # theta = np.arctan2(S2_gray, S1_gray) / 2  # in [-pi/2, pi/2]
    # theta = (theta < 0) * np.pi + theta  # convert to [0, pi] by adding pi to negative values
    # theta = theta.astype(np.float32)
    # np.save('./data/bunny1205/bunny/bunny_p_{}.npy'.format(num),np.array(p))
    # np.save('./data/bunny1205/bunny/bunny_t_{}.npy'.format(num),np.array(theta))


    ####### save S0 S1 S2
    # S0 = channels['stokes.S0']
    # np.save("./data/bunny0423_d/bunny_s0/{:04d}.npy".format(num),np.array(S0))
    # S1 = channels['stokes.S1']
    # np.save("./data/bunny0423_d/bunny_s1/{:04d}.npy".format(num),np.array(S1))
    # S2 = channels['stokes.S2']
    # np.save("./data/bunny0423_d/bunny_s2/{:04d}.npy".format(num),np.array(S2))

#############
    # fig, ax = plt.subplots(ncols=3, figsize=(18, 5))

    # img0=ax[0].imshow(channels['stokes.S0'].convert(srgb_gamma=True))
    # ax[0].set_xticks([]); ax[0].set_yticks([])
    # plt.colorbar(img0, ax=ax[0])

    # img1=ax[1].imshow(p, cmap='GnBu',vmax=1,vmin=0)
    # ax[1].set_xticks([]); ax[1].set_yticks([])
    # plt.colorbar(img1, ax=ax[1])

    # img2=ax[2].imshow(theta, cmap='hsv')
    # ax[2].set_xticks([]); ax[2].set_yticks([])
    # plt.colorbar(img2, ax=ax[2])

    # plt.show()
##############