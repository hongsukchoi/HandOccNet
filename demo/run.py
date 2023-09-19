import sys
import glob
import os
import os.path as osp
import argparse
import json
import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
from tqdm import tqdm

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import load_img, process_bbox, generate_patch_image
from utils.vis import save_obj, vis_keypoints_with_skeleton
from utils.mano import MANO
from utils.camera import PerspectiveCamera
mano = MANO()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def load_camera(cam_path, cam_idx='0'):
    with open(cam_path, 'r') as f:
        cam_data = json.load(f)

        camera = PerspectiveCamera()

        camera.focal_length_x = torch.full([1], cam_data[cam_idx]['fx'])
        camera.focal_length_y = torch.full([1], cam_data[cam_idx]['fy'])
        camera.center = torch.tensor(
            [cam_data[cam_idx]['cx'], cam_data[cam_idx]['cy']]).unsqueeze(0)
        # only intrinsics
        # rotation, _ = cv2.Rodrigues(
        #     np.array(cam_data[cam_idx]['rvec'], dtype=np.float32))
        # camera.rotation.data = torch.from_numpy(rotation).unsqueeze(0)
        # camera.translation.data = torch.tensor(
        #     cam_data[cam_idx]['tvec']).unsqueeze(0) / 1000.
        camera.rotation.requires_grad = False
        camera.translation.requires_grad = False
        camera.name = str(cam_idx)

    return camera

if __name__ == '__main__':
    # argument parsing
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True
    transform = transforms.ToTensor()

    # model snapshot load
    model_path = '/home/hongsuk.c/Projects/HandOccNet/main/../output/model_dump/snapshot_70.pth.tar' #'./snapshot_demo.pth.tar'
    assert osp.exists(model_path), 'Cannot find model at ' + model_path
    print('Load checkpoint from {}'.format(model_path))
    model = get_model('test')

    model = DataParallel(model).cuda()
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['network'], strict=False)
    model.eval()

    # set directory and file paths
    subset = 'handnerf_training_test'
    data_dir = f'/home/hongsuk.c/Projects/HandNeRF_annotation/data/{subset}/cam_0'
    cam_path = '/home/hongsuk.c/Projects/MultiCamCalib/data/handnerf_calibration_0822/output/cam_params/cam_params_final.json'
    
    # this code assumes only right hand images. for left hands, flip the image
    images = sorted(glob.glob(data_dir + '/*.jpg'))
    annots = sorted(glob.glob(data_dir + '/*.json'))
    depths = sorted(glob.glob(
        f'/home/hongsuk.c/Projects/HandNeRF_annotation/data/{subset}/cam_0_depth/*.png'))

    # get camera for projection
    camera = load_camera(cam_path, cam_idx='0')
    camera.cuda()

    save_dir = f'/home/hongsuk.c/Projects/HandNeRF_annotation/data/{subset}/cam_0_handoccnet_finetuned'
    if not osp.exists(save_dir):
        os.mkdir(save_dir)

    for idx, (img_path, ann_path, depth_path) in enumerate(tqdm(zip(images, annots, depths))):

        original_img = load_img(img_path)
        with open(ann_path, 'r') as f:
            ann = json.load(f)
        print(img_path, ann_path)
        geo_anns = ann["shapes"]
        hand_bbox = None
        for g in geo_anns:
            if g["label"] == "right_hand" and g["shape_type"] == "rectangle":
                hb = g["points"] # xyxy
                hand_bbox = [hb[0][0], hb[0][1], hb[1][0] - hb[0][0], hb[1][1] - hb[0][1]]
                break
        
        if hand_bbox is None:
            import pdb; pdb.set_trace()

        # process input image
        original_img_height, original_img_width = original_img.shape[:2]
        bbox = process_bbox(hand_bbox, original_img_width, original_img_height)
        img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape) 
        img = transform(img.astype(np.float32))/255
        img = img.cuda()[None,:,:,:]

        # forward pass to the model
        inputs = {'img': img} # cfg.input_img_shape[1], cfg.input_img_shape[0], 3
        targets = {}
        meta_info = {}
        with torch.no_grad():
            out = model(inputs, targets, meta_info, 'test')
        img = (img[0].cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8) # 
        verts_out = out['mesh_coord_cam'][0].cpu().numpy()
        
        # get hand mesh's scale and translation by fitting joint cam to joint img

        joint_img, joint_cam = out['joints_coord_img'], out['joints_coord_cam']
    
        # denormalize joint_img from 0 ~ 1 to actual 0 ~ original height and width
        H, W = img.shape[:2]
        joint_img[:, :, 0] *= W
        joint_img[:, :, 1] *= H
        torch_bb2img_trans = torch.tensor(bb2img_trans).to(joint_img)
        homo_joint_img = torch.cat([joint_img, torch.ones_like(joint_img[:, :, :1])], dim=2)
        org_res_joint_img = homo_joint_img @ torch_bb2img_trans.transpose(0, 1)

        # depth initialization
        depth = np.asarray(Image.open(depth_path))

        hand_scale, hand_translation = model.module.get_mesh_scale_trans(
            org_res_joint_img, joint_cam, camera, depth)

        np_joint_img = org_res_joint_img[0].cpu().numpy()
        np_joint_img = np.concatenate([np_joint_img, np.ones_like(np_joint_img[:, :1])], axis=1)
        vis_img = original_img.astype(np.uint8)[:, :, ::-1]
        pred_joint_img_overlay = vis_keypoints_with_skeleton(vis_img, np_joint_img.T, mano.skeleton)
        # cv2.imshow('2d prediction', pred_joint_img_overlay)
        save_path = osp.join(
            save_dir, f'{osp.basename(img_path)[:-4]}_2d_prediction.png')

        cv2.imwrite(save_path, pred_joint_img_overlay)
        projected_joints = camera(
            hand_scale * joint_cam + hand_translation)
        np_joint_img = projected_joints[0].detach().cpu().numpy()
        np_joint_img = np.concatenate([np_joint_img, np.ones_like(np_joint_img[:, :1])], axis=1)

        vis_img = original_img.astype(np.uint8)[:, :, ::-1]
        pred_joint_img_overlay = vis_keypoints_with_skeleton(vis_img, np_joint_img.T, mano.skeleton)
        # cv2.imshow('projection', pred_joint_img_overlay)
        # cv2.waitKey(0)
        save_path = osp.join(save_dir, f'{osp.basename(img_path)[:-4]}_projection.png')
        cv2.imwrite(save_path, pred_joint_img_overlay)
        
        # data to save
        data_to_save = {
            'hand_scale': hand_scale.detach().cpu().numpy().tolist(), # 1
            'hand_translation': hand_translation.detach().cpu().numpy().tolist(), # 3
            'mano_pose': out['mano_pose'][0].detach().cpu().numpy().tolist(),  # 48
            'mano_shape': out['mano_shape'][0].detach().cpu().numpy().tolist(),  # 10
        }
        save_path = osp.join(
            save_dir, f'{osp.basename(img_path)[:-4]}_3dmesh.json')
        with open(save_path, 'w') as f:
            json.dump(data_to_save, f)

        """ Visualization """
        # bbox for input hand image
        # bbox_vis = np.array(bbox, int)
        # bbox_vis[2:] += bbox_vis[:2]
        # cvimg = cv2.rectangle(original_img.copy(),
        #                     bbox_vis[:2], bbox_vis[2:], (255, 0, 0), 3)
        # cv2.imwrite(f'test{idx}_hand_bbox.png', cvimg[:, :, ::-1])

        # ## input hand image
        # cv2.imwrite(f'test{idx}_hand_image.png', img[:, :, ::-1])

        # save mesh (obj)
        save_path = osp.join(
            save_dir, f'{osp.basename(img_path)[:-4]}_3dmesh.obj')
        save_obj(verts_out*np.array([1, -1, -1]),
                 mano.face, save_path)
