import torch
import torch.nn as nn
from torch.nn import functional as F
from HandOccNet.common.nets.backbone import FPN
from HandOccNet.common.nets.transformer import Transformer
from HandOccNet.common.nets.regressor import Regressor
from HandOccNet.common.utils.fitting import ScaleTranslationLoss, FittingMonitor
from HandOccNet.common.utils.optimizers import optim_factory
from HandOccNet.common.utils.camera import PerspectiveCamera
# from HandOccNet.main.config import cfg

class Model(nn.Module):
    def __init__(self, backbone, FIT, SET, regressor):
        super(Model, self).__init__()
        self.backbone = backbone
        self.FIT = FIT
        self.SET = SET
        self.regressor = regressor

        # fitting hand scale and translation 
        self.fitting_joint_idxs = list(range(0, 21))
        self.fitting_loss = ScaleTranslationLoss(self.fitting_joint_idxs)

    
    def forward(self, inputs, targets, meta_info, mode):
        p_feats, s_feats = self.backbone(inputs['img']) # primary, secondary feats
        feats = self.FIT(s_feats, p_feats)
        feats = self.SET(feats, feats)

        if mode == 'train':
            gt_mano_params = torch.cat([targets['mano_pose'], targets['mano_shape']], dim=1)
        else:
            gt_mano_params = None
        pred_mano_results, gt_mano_results, preds_joints_img = self.regressor(feats, gt_mano_params)
       
        if False and mode == 'train':
            # loss functions
            loss = {}
            loss['mano_verts'] = cfg.lambda_mano_verts * F.mse_loss(pred_mano_results['verts3d'], gt_mano_results['verts3d'])
            loss['mano_joints'] = cfg.lambda_mano_joints * F.mse_loss(pred_mano_results['joints3d'], gt_mano_results['joints3d'])
            loss['mano_pose'] = cfg.lambda_mano_pose * F.mse_loss(pred_mano_results['mano_pose'], gt_mano_results['mano_pose'])
            loss['mano_shape'] = cfg.lambda_mano_shape * F.mse_loss(pred_mano_results['mano_shape'], gt_mano_results['mano_shape'])
            loss['joints_img'] = cfg.lambda_joints_img * F.mse_loss(preds_joints_img[0], targets['joints_img'])
            return loss

        else:
            # test output
            out = {}
            out['joints_coord_img'] = preds_joints_img[0]
            out['mano_pose'] = pred_mano_results['mano_pose_aa']
            out['mano_shape'] = pred_mano_results['mano_shape']
            
            out['joints_coord_cam'] = pred_mano_results['joints3d']
            out['mesh_coord_cam'] = pred_mano_results['verts3d']

            out['mano_joints2cam'] = pred_mano_results['mano_joints2cam'] 
            out['mano_pose_aa'] = pred_mano_results['mano_pose_aa']

            return out

    def get_mesh_scale_trans(self, pred_joint_img, pred_joint_cam, camera=None, depth=None):
        """
        pred_joint_img: (batch_size, 21, 2)
        pred_joint_cam: (batch_size, 21, 3)
        """
        if camera is None:
            camera = PerspectiveCamera()

        dtype, device = pred_joint_cam.dtype, pred_joint_cam.device
        hand_scale = torch.tensor([1.0 / 1.0], dtype=dtype, device=device, requires_grad=False)
        hand_translation = torch.tensor([0, 0, .6], dtype=dtype, device=device, requires_grad=True)
        if depth is not None:
            tensor_depth = torch.tensor(depth, device=device, dtype=dtype)[
                None, None, :, :]
            grid = pred_joint_img.clone()
            grid[:, :, 0] /= tensor_depth.shape[-1]
            grid[:, :, 1] /= tensor_depth.shape[-2]
            grid = 2 * grid - 1
            joints_depth = torch.nn.functional.grid_sample(
                tensor_depth, grid[:, None, :, :])  # (1, 1, 1, 21)
            joints_depth = joints_depth.reshape(1, 21, 1)
            hand_translation = torch.tensor(
                [0, 0, joints_depth[0, self.fitting_joint_idxs, 0].mean() / 1000.], device=device, requires_grad=True)

        # intended only for demo mesh rendering
        batch_size = 1
        self.fitting_loss.trans_estimation = hand_translation.clone()

        params = []
        params.append(hand_translation)
        # params.append(hand_scale)
        optimizer, create_graph = optim_factory.create_optimizer(
            params, optim_type='lbfgsls', lr=1.0e-1)

        # optimization
        print("[Fitting]: fitting the hand scale and translation...")
        with FittingMonitor(batch_size=batch_size) as monitor:
            fit_camera = monitor.create_fitting_closure(
                optimizer, camera, pred_joint_cam, pred_joint_img, hand_translation, hand_scale, self.fitting_loss, create_graph=create_graph)

            loss_val = monitor.run_fitting(
                optimizer, fit_camera, params)


        print(f"[Fitting]: fitting finished with loss of {loss_val}")
        print(f"Scale: {hand_scale.detach().cpu().numpy()}, Translation: {hand_translation.detach().cpu().numpy()}")
        return hand_scale, hand_translation


def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(mode):
    backbone = FPN(pretrained=True)
    FIT = Transformer(injection=True) # feature injecting transformer
    SET = Transformer(injection=False) # self enhancing transformer
    regressor = Regressor()
    
    if mode == 'train':
        FIT.apply(init_weights)
        SET.apply(init_weights)
        regressor.apply(init_weights)
        
    model = Model(backbone, FIT, SET, regressor)
    
    return model