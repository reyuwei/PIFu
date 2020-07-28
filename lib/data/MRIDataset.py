from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import logging

log = logging.getLogger('trimesh')
log.setLevel(40)
import json
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt


def load_trimesh(subject_list, root_dir):
    meshs = {}
    for i, f in enumerate(subject_list):
        sub_name = f
        meshs[sub_name] = trimesh.load(os.path.join(root_dir, '%s_mano.obj' % sub_name))

    return meshs


""" General util functions. """


def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg


def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d


def load_db_annotation(base_path, set_name=None):
    if set_name is None:
        # only training set annotations are released so this is a valid default choice
        set_name = 'training'

    print('Loading FreiHAND dataset index ...')
    t = time.time()

    # assumed paths to data containers
    k_path = os.path.join(base_path, '%s_K.json' % set_name)
    mano_path = os.path.join(base_path, '%s_mano.json' % set_name)
    xyz_path = os.path.join(base_path, '%s_xyz.json' % set_name)

    # load if exist
    K_list = json_load(k_path)
    mano_list = json_load(mano_path)
    xyz_list = json_load(xyz_path)

    # should have all the same length
    assert len(K_list) == len(mano_list), 'Size mismatch.'
    assert len(K_list) == len(xyz_list), 'Size mismatch.'

    print('Loading of %d samples done in %.2f seconds' % (len(K_list), time.time() - t))
    return zip(K_list, mano_list, xyz_list)


class MRIDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train', projection_mode='perspective'):
        self.opt = opt
        self.projection_mode = projection_mode

        self.root_joint_id = 9
        # Path setup
        self.root = self.opt.dataroot
        self.MRI = os.path.join(self.root, 'mri')
        self.MASK = os.path.join(self.root, 'mask')
        self.RENDER = os.path.join(self.root, 'rgb')

        self.is_train = (phase == 'train')
        self.load_size = self.opt.loadSize

        self.num_sample_inout = self.opt.num_sample_inout
        self.num_sample_color = self.opt.num_sample_color

        self.subjects = self.get_subjects()

        self.num_views = self.opt.num_views
        self.yaw_list = list(range(0, self.num_views, 1))

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.db_data_anno = list(load_db_annotation(Path(self.root).parent, "training"))

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])

        self.mesh_dic = load_trimesh(self.subjects, self.MRI)
        self.shift_mesh_center()

    def shift_mesh_center(self):
        for i, subject in enumerate(self.subjects):
            mesh = self.mesh_dic[subject]
            subject_id = int(subject)
            K, mano, xyz = self.db_data_anno[subject_id]
            K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
            mesh.vertices -= np.array(xyz[self.root_joint_id])

    def get_subjects(self):
        all_subjects = np.sort(os.listdir(self.RENDER))
        all_subjects = [x[:-4] for x in all_subjects]
        if self.opt.data_samples != -1:
            all_subjects = all_subjects[:self.opt.data_samples]
        np.random.shuffle(all_subjects)
        var_num = int(0.3 * len(all_subjects))
        var_subjects = all_subjects[:var_num]
        # var_subjects = np.loadtxt(os.path.join(self.root, 'val.txt'), dtype=str)
        if len(var_subjects) == 0:
            return all_subjects

        if self.is_train:
            return sorted(list(set(all_subjects) - set(var_subjects)))
        else:
            return sorted(list(var_subjects))

    def get_render(self, subject, num_views, random_sample=False):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img': [num_views, C, W, H] images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask': [num_views, 1, W, H] masks
        '''

        calib_list = []
        render_list = []
        mask_list = []
        extrinsic_list = []

        render_path = os.path.join(self.RENDER, subject + '.jpg')
        mask_path = os.path.join(self.MASK, subject + '.jpg')

        mask = Image.open(mask_path).convert('L')
        render = Image.open(render_path).convert('RGB')

        # loading calibration data
        subject_id = int(subject)
        K, mano, xyz = self.db_data_anno[subject_id]
        K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]

        K_4 = np.eye(4)
        K_4[:3, :3] = K
        K = K_4

        extrinsic = np.eye(4)
        extrinsic[:3, -1] = xyz[self.root_joint_id]
        # if self.is_train:
        # # Pad images
        # pad_size = int(0.1 * self.load_size)
        # render = ImageOps.expand(render, pad_size, fill=0)
        # mask = ImageOps.expand(mask, pad_size, fill=0)
        #
        # w, h = render.size
        # th, tw = self.load_size, self.load_size
        #
        # # random flip
        # if self.opt.random_flip and np.random.rand() > 0.5:
        #     scale_intrinsic[0, 0] *= -1
        #     render = transforms.RandomHorizontalFlip(p=1.0)(render)
        #     mask = transforms.RandomHorizontalFlip(p=1.0)(mask)
        #
        # # random scale
        # if self.opt.random_scale:
        #     rand_scale = random.uniform(0.9, 1.1)
        #     w = int(rand_scale * w)
        #     h = int(rand_scale * h)
        #     render = render.resize((w, h), Image.BILINEAR)
        #     mask = mask.resize((w, h), Image.NEAREST)
        #     scale_intrinsic *= rand_scale
        #     scale_intrinsic[3, 3] = 1
        #
        # # random translate in the pixel space
        # if self.opt.random_trans:
        #     dx = random.randint(-int(round((w - tw) / 10.)),
        #                         int(round((w - tw) / 10.)))
        #     dy = random.randint(-int(round((h - th) / 10.)),
        #                         int(round((h - th) / 10.)))
        # else:
        #     dx = 0
        #     dy = 0
        #
        # trans_intrinsic[0, 3] = -dx / float(self.opt.loadSize // 2)
        # trans_intrinsic[1, 3] = -dy / float(self.opt.loadSize // 2)
        #
        # x1 = int(round((w - tw) / 2.)) + dx
        # y1 = int(round((h - th) / 2.)) + dy
        #
        # render = render.crop((x1, y1, x1 + tw, y1 + th))
        # mask = mask.crop((x1, y1, x1 + tw, y1 + th))
        #
        # render = self.aug_trans(render)
        #
        # # random blur
        # if self.opt.aug_blur > 0.00001:
        #     blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
        #     render = render.filter(blur)
        #
        # intrinsic = np.matmul(K, np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic)))
        intrinsic = K
        calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
        extrinsic = torch.Tensor(extrinsic).float()

        mask = transforms.Resize(self.load_size)(mask)
        mask = transforms.ToTensor()(mask).float()
        mask_list.append(mask)

        render = self.to_tensor(render)
        # render = mask.expand_as(render) * render

        render_list.append(render)
        calib_list.append(calib)
        extrinsic_list.append(extrinsic)

        return {
            'img': torch.stack(render_list, dim=0),
            'calib': torch.stack(calib_list, dim=0),
            'extrinsic': torch.stack(extrinsic_list, dim=0),
            'mask': torch.stack(mask_list, dim=0),
            'joints3d': torch.Tensor(xyz).float(),
        }

    def __len__(self):
        return len(self.subjects)

    @staticmethod
    def mano2niipts(mano_verts, spacing=[0.5, 0.5, 0.5]):
        pt = mano_verts.squeeze()
        xyz = np.array(pt)
        max_xyz = np.max(xyz, axis=0) + spacing[0] * 2
        min_xyz = np.min(xyz, axis=0) - spacing[0] * 2
        D, H, W = max_xyz
        SD, SH, SW = min_xyz
        ds, hs, ws = spacing
        x_ = np.arange(SD, D + 1e-5, step=ds, dtype=np.float32)
        y_ = np.arange(SH, H + 1e-5, step=hs, dtype=np.float32)
        z_ = np.arange(SW, W + 1e-5, step=ws, dtype=np.float32)
        px, py, pz = np.meshgrid(x_, y_, z_, indexing='ij')
        all_pts = np.stack([px, py, pz], -1)  # [D, H, W, 3]
        bounds = np.array([min_xyz, max_xyz])
        return all_pts, bounds

    def select_sampling_method(self, subject, res):
        if not self.is_train:
            random.seed(1991)
            np.random.seed(1991)
            torch.manual_seed(1991)
        mesh = self.mesh_dic[subject]

        vol = np.load(res['mri_vol_path'])
        D, H, W, C = vol.shape
        vol_flat = vol.reshape(-1, C)

        # move all root_joint to [0,0,0]
        vol_flat[:, :3] -= np.array(res['joints3d'][self.root_joint_id])

        random_choice = np.random.choice(D * H * W, size=self.num_sample_inout, replace=False)
        sample_points = vol_flat[random_choice, :]

        samples_p = sample_points[:, :3]
        samples_l = sample_points[:, 3]

        outside = np.logical_not(mesh.contains(samples_p))
        samples_l[outside] = 0

        samples_all = torch.Tensor(vol_flat[:, :3].T).float()
        labels_all = torch.Tensor(vol_flat[:, 3].reshape(1, -1)).float()
        samples = torch.Tensor(samples_p.T).float()
        labels = torch.Tensor(samples_l.reshape(1, self.num_sample_inout)).float()

        # mesh.export(self.opt.dataroot + "\\debug\\" + subject + "_mano.obj")

        del mesh

        if self.opt.eval_only:
            return {
                'samples': samples,
                'labels': labels,
                'samples_all': samples_all,
                'labels_all': labels_all,
            }
        else:
            return {
                'samples': samples,
                'labels': labels,
            }

    @staticmethod
    def projectPoints(xyz, K):
        """ Project 3D coordinates into image space. """
        xyz = np.array(xyz)
        K = np.array(K)
        uv = np.matmul(K, xyz.T).T
        return uv[:, :2] / uv[:, -1:]

    def get_item(self, index):
        # In case of a missing file or IO error, switch to a random sample instead
        # try:
        sid = index % len(self.subjects)

        # name of the subject 'rp_xxxx_xxx'
        subject = self.subjects[sid]
        res = {
            'name': subject,
            'mesh_path': os.path.join(self.MRI, subject + '_mano.obj'),
            'mri_vol_path': os.path.join(self.MRI, subject + '_mri_label.npy'),
            'mri_nii_path': os.path.join(self.MRI, subject + '.nii.gz'),
        }
        render_data = self.get_render(subject, num_views=self.num_views,
                                      random_sample=self.opt.random_multiview)
        res.update(render_data)

        if self.opt.num_sample_inout:
            sample_data = self.select_sampling_method(subject, res)
            res.update(sample_data)

        # img = np.uint8((np.transpose(rendeer_data['img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
        # K = render_data['calib'][0, :3, :3]
        # trans = render_data['calib'][0, :3, 3:4]
        # # pts = self.projectPoints(sample_data['samples'].T[sample_data['labels'][0] != 0], K)
        # pts = torch.addmm(trans, K, sample_data['samples'][:, sample_data['labels'][0] != 0])  # [3, N]
        # pts = pts.T[:, :2] / pts.T[:, 2:3]
        # for p in pts:
        #     if abs(p[0]) < img.shape[0] and abs(p[1]) < img.shape[0]:
        #         img[int(p[1]), int(p[0])] = (255, 0, 0)
        # plt.figure()
        # plt.imshow(img)
        # plt.title(subject)
        # plt.show()
        # cv2.imshow('test', img)
        # cv2.waitKey(1)

        # if self.num_sample_color:
        #     color_data = self.get_color_sampling(subject, yid=yid, pid=pid)
        #     res.update(color_data)

        return res
        # except Exception as e:
        #     print(e)
        #     return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)
