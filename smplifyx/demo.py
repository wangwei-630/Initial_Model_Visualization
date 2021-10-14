import argparse
import pickle
import torch
import smplx
import numpy as np

from os import path as osp
from cmd_parser import parse_config

import pyrender
import trimesh


if __name__ == '__main__':
    '''debug查看smplx参数'''
    # models_dir = '../models'
    # bm_fname =  osp.join(models_dir,'smplx/SMPLX_NEUTRAL.npz')#'PATH_TO_SMPLX_model.pkl'  obtain from https://smpl-x.is.tue.mpg.de/downloads
    # smplx_dict = np.load(bm_fname, encoding='latin1')


    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', nargs='+', type=str, required=True,
                        help='The pkl files that will be read')

    args, remaining = parser.parse_known_args()
    pkl_paths = args.pkl
    args = parse_config(remaining)
    # print('args:', args.keys())
    # ['data_folder', 'max_persons', 'config', 'loss_type', 'interactive', 'save_meshes', 'visualize', 'degrees',
    #  'use_cuda', 'dataset', 'joints_to_ign', 'output_folder', 'img_folder', 'keyp_folder', 'summary_folder',
    #  'result_folder', 'mesh_folder', 'gender_lbl_type', 'gender', 'float_dtype', 'model_type', 'camera_type',
    #  'optim_jaw', 'optim_hands', 'optim_expression', 'optim_shape', 'model_folder', 'use_joints_conf', 'batch_size',
    #  'num_gaussians', 'use_pca', 'num_pca_comps', 'flat_hand_mean', 'body_prior_type', 'left_hand_prior_type',
    #  'right_hand_prior_type', 'jaw_prior_type', 'use_vposer', 'vposer_ckpt', 'init_joints_idxs', 'body_tri_idxs',
    #  'prior_folder', 'focal_length', 'rho', 'interpenetration', 'penalize_outside', 'data_weights',
    #  'body_pose_prior_weights', 'shape_weights', 'expr_weights', 'face_joints_weights', 'hand_joints_weights',
    #  'jaw_pose_prior_weights', 'hand_pose_prior_weights', 'coll_loss_weights', 'depth_loss_weight', 'df_cone_height',
    #  'max_collisions', 'point2plane', 'part_segm_fn', 'ign_part_pairs', 'use_hands', 'use_face', 'use_face_contour',
    #  'side_view_thsh', 'optim_type', 'lr', 'gtol', 'ftol', 'maxiters']

    dtype = torch.float32
    use_cuda = args.get('use_cuda', True)
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model_type = args.get('model_type', 'smplx')
    print('Model type:', model_type)
    print('Model folder:', args.get('model_folder'))

    model_params = dict(model_path=args.get('model_folder'),
                        #  joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=not args.get('use_vposer'),
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        dtype=dtype,
                        **args)

    model = smplx.create(**model_params)
    model = model.to(device=device)

    '''load SMPLX_NEUTRAL.pkl'''
    for pkl_path in pkl_paths:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            print("------------>>>>>>>>>>>>>>----------------")
            print('data.keys:', data.keys())
            '''
            data.keys：
            ['dynamic_lmk_bary_coords', 
            'hands_componentsl', 
            'ft', 
            'lmk_faces_idx', 
            'f', 
            'J_regressor', 
            'hands_componentsr', 
            'kintree_table', 
            'hands_coeffsr', 
            'joint2num', 
            'hands_meanl', 
            'lmk_bary_coords', 
            'weights', 
            'posedirs', 
            'dynamic_lmk_faces_idx', 
            'part2num', 
            'vt', 
            'hands_meanr', 
            'hands_coeffsl', 
            'v_template', 
            'shapedirs']
            '''
            print("------------>>>>>>>>>>>>>>----------------")

        '''SMPLX_NEUTRAL.pkl -> tensor'''
        est_params = {}
        for key, val in data.items():
            print(key)
            if key == "ft" or key == "f":
                val = val/1.0   #can't convert np.ndarray of type numpy.uint32
            elif key == "joint2num" or key == "part2num":
                continue    #can't convert np.ndarray of type numpy.object_.
            est_params[key] = torch.tensor(val, dtype=dtype, device=device)

        '''output 3D model'''
        model_output = model(**est_params)
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()

        out_mesh = trimesh.Trimesh(vertices, model.faces, process=False)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 1.0, 0.9, 1.0))
        mesh = pyrender.Mesh.from_trimesh(
            out_mesh,
            material=material)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')
        pyrender.Viewer(scene, use_raymond_lighting=True)