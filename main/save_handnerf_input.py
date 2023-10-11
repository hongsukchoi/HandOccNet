import torch
import argparse
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn
from config import cfg
from base import Tester

# # TEMP 
# from DEX_YCB import target_img_list_sum


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():

    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()
    
    total_to_save = {}
    cur_sample_idx = 0
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
        
        # forward
        with torch.no_grad():
            out = tester.model(inputs, targets, meta_info, 'test')
        
        to_save = {'/'.join(name.split('/')[-4:]): 
        [
            out['mesh_coord_cam'][idx].cpu().numpy(),
            out['joints_coord_cam'][idx].cpu().numpy(),
            out['mano_joints2cam'][idx].cpu().numpy(),
            out['mano_pose_aa'][idx].cpu().numpy(),
        ] for idx, name in enumerate(meta_info['img_path'])}
        total_to_save.update(to_save)

    np.save('DexYCB_HandNeRF_novel_object_testset_HandOccNet_pred.npy', total_to_save)


    # tester._print_eval_result(args.test_epoch)

if __name__ == "__main__":
    main()