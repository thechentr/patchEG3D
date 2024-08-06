import os
import sys
import gin
import torch
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F
sys.path.append(os.path.join(os.path.dirname(__file__), 'eg3d', 'eg3d'))
import legacy
import dnnlib
from torch_utils import misc
from training.triplane import TriPlaneGenerator
from camera_utils import (
    math_utils,
    create_cam2world_matrix,
    LookAtPoseSampler, 
    FOV_to_intrinsics,
)


import os
import torch
import torchvision.transforms as transforms


def load_eg3d_model(
    model_path,
    neural_rendering_resolution=None,
    device=None
):
    if device is None:
        device = torch.device('cpu')

    with dnnlib.util.open_url(model_path) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].to(device)

    G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
    misc.copy_params_and_buffers(G, G_new, require_all=True)

    G_new.neural_rendering_resolution = (
        neural_rendering_resolution
        if neural_rendering_resolution is not None
        else G.neural_rendering_resolution
    )
    G_new.rendering_kwargs = G.rendering_kwargs
    G_new.rendering_kwargs['ray_start'] = G.rendering_kwargs['ray_end'] = 'auto'
    G_new.rendering_kwargs['avg_camera_radius'] = 5.0
    
    del G

    return G_new

@gin.configurable
class EG3DRender(object):
    """
    每个car都有angle_num个初始状态，每次reset只在这其中选择
    """
    def __init__(
        self, 
        device=torch.device('cpu'),
        imgsize=(256, 256),
        horizontal_range=[-60,60],
        vertical_range=[0, 30],
        angle_num=150,
    ):

        self.device = device  
        self.imgsize = imgsize  
        self.horizontal_range = horizontal_range  
        self.vertical_range = vertical_range  
        self.angle_num = angle_num  

        self.model = load_eg3d_model(
            model_path='./checkpoints/shapenetcars128-64.pkl',
            device=self.device
        )
        self.model.eval()
        self.cam_pivot = self.model.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0])
        self.cam_pivot = torch.tensor(self.cam_pivot, device=self.device)
        self.cam_radius = self.model.rendering_kwargs.get('avg_camera_radius', 5.0)
        intrinsics = FOV_to_intrinsics(18.837, self.device)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(  # NOTE 为什么要采样相机矩阵
            1e-5, 1e-5, self.cam_pivot,
            radius=self.cam_radius, device=self.device
        )
        self.conditioning_params = torch.cat(
            [conditioning_cam2world_pose.reshape(-1),
            intrinsics.reshape(-1)], dim=0)
        self.intrinsics = intrinsics.reshape(-1)


        

        self.cars_seeds = None
        self.states = None
        self.imgs_tensor = None
        self.cam2world_pose = None
        self.rotated_points = None
    

    def reset(self, seeds:list, states=None):
        """
        input:
        - states: unit degree

        return: 
        - imgs_tensor: [n, 3, w, h]
        - rotated_points: 
        """   
        self.cars_seeds = seeds
        self.states = states if (states is not None) else self._sample_state(len(seeds))
        self.imgs_tensor, self.cam2world_pose = self._render(self.cars_seeds, self.states)
        self.rotated_points = self._calculate_rpoints(self.cam2world_pose)
        return self.imgs_tensor, self.rotated_points
    
    def step(self, actions):
        """
        actions: degree
        return: imgs_tensor, rotated_points
        """
        self.states = self.states + actions
        self.imgs_tensor, self.cam2world_pose = self._render(self.cars_seeds, self.states)
        self.rotated_points = self._calculate_rpoints(self.cam2world_pose)
        return self.imgs_tensor, self.rotated_points
    

    def _render(self, seeds:list, states:torch.tensor): 
        """
        seed -> zs -> ws -> imgs
        just render, do not save to self
        states: degree
        """
        assert states.shape[0] == len(seeds)
        hs = states[:, 0]
        vs= states[:, 1]
        N = len(seeds)

        zs = self._seeds2zs(seeds)
        ws = self._zs2ws(zs)

        camera_origins = torch.stack([torch.sin(torch.deg2rad(hs+90))*self.cam_radius, 
                                -torch.cos(torch.deg2rad(hs+90))*self.cam_radius, 
                                torch.sin(torch.deg2rad(vs))*self.cam_radius], dim=1)
        cam2world_pose = create_cam2world_matrix(-camera_origins, camera_origins)
        
        # return torch.rand((1, 3, 256, 256), device=self.device)
    
        camera_params = torch.cat(
            [cam2world_pose.reshape(-1, 16),
             self.intrinsics.repeat(N, 1).detach()], dim=1
        )
        
        imgs_norm = self.model.synthesis(ws, camera_params)['image'].clamp(-1, 1)  # [-1, 1]
        imgs_tensor = (imgs_norm * 127.5 + 128).clamp(0, 255) / 255  # [0, 1]

        imgs_tensor = F.interpolate(input=imgs_tensor, size=self.imgsize, mode='bilinear', align_corners=False)
        
        return imgs_tensor, cam2world_pose

    def _calculate_rpoints(self, cam2world_pose: torch.tensor):
        """
        根据数据cam2world_pose 计算rotate points
        """
        self.rotated_points = []
        for i in range(len(cam2world_pose)):
            img_w, img_h = self.imgsize
            patch_radius, patch_w, patch_h = 0.19, 0.16, 0.08   # magic number
            p_world = torch.tensor([[patch_radius, -patch_w, patch_h],
                                    [patch_radius, -patch_w, -patch_h],
                                    [patch_radius, patch_w, -patch_h],
                                    [patch_radius, patch_w, patch_h]], dtype=torch.float, device=self.device)  
            rotated_points = self._world2img(p_world, self.cam2world_pose[i])
            rotated_points[:, 0] = rotated_points[:, 0] * img_w
            rotated_points[:, 1] = rotated_points[:, 1] * img_h
            self.rotated_points.append(rotated_points)
        self.rotated_points = torch.stack(self.rotated_points, dim=0)

        return self.rotated_points

        points_orig = [ [0, 0], 
                        [0, patch_tensor.shape[1]], 
                        [patch_tensor.shape[2], patch_tensor.shape[1]], 
                        [patch_tensor.shape[1], 0] ]
        
        patch_tensor = TF.perspective(patch_tensor, points_orig, self.rotated_points[i], interpolation=transforms.InterpolationMode.NEAREST, fill=-1)

        mask = patch_tensor.mean(0) == -1
        imgs_tensor[i] = torch.where(mask, imgs_tensor[i], patch_tensor)

        return imgs_tensor
    
    def _world2img(self, p_world, cam2world_pose):
        # 转成齐次坐标形式  
        p_world_h = torch.cat([p_world, torch.ones((p_world.shape[0], 1), dtype=torch.float, device=p_world.device)], dim=1)  
        world2cam = torch.inverse(cam2world_pose)  

        # 世界坐标到相机坐标转换  
        p_cam_h = world2cam @ p_world_h.unsqueeze(-1)  
        p_cam = p_cam_h[:, :3, 0] / p_cam_h[:, 3, 0].unsqueeze(1)  # 归一化以获得非齐次坐标  

        K = self.intrinsics.reshape(3, 3)

        p_cam_h = torch.cat([p_cam, torch.ones((p_cam.shape[0], 1), dtype=torch.float, device=p_cam.device)], dim=1)  

        # 相机坐标到像平面坐标  
        p_image_h = K @ p_cam.transpose(0, 1)  

        # 归一化以获得最终图像像素坐标  
        p_image = p_image_h[:2, :] / p_image_h[2, :]  
        p_image = p_image.transpose(0, 1)  # 转换回 (N, 2) 形式 
        
        return p_image


    def detach(self):
        if self.states is not None:
            self.states = self.states.detach()


    def _seeds2zs(self, seeds: list):
        zs_np = np.stack([np.random.RandomState(seed).rand(512) for seed in seeds])
        zs = torch.from_numpy(zs_np).to(self.device)
        return zs
    

    def _zs2ws(self, zs, truncation_psi=1.0, truncation_cutoff=14):
        ws = self.model.mapping(
            zs, self.conditioning_params.repeat(zs.size(0), 1),
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff
        )
        return ws
    
    def _sample_state(self, N):  
        h_tensor = torch.rand(N, device=self.device) * (self.horizontal_range[1] - self.horizontal_range[0]) + self.horizontal_range[0]  
        v_tensor = torch.rand(N, device=self.device) * (self.vertical_range[1] - self.vertical_range[0]) + self.vertical_range[0]    
        states = torch.stack([h_tensor, v_tensor], dim=1)  
        states.requires_grad_(True)  
        return states 