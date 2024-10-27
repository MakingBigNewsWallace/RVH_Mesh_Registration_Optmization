"""
fit smplh to scans

crated by Xianghui, 12, January 2022

the code is tested
"""

import sys, os
sys.path.append(os.getcwd())
import json
import pickle as pkl
import torch
import numpy as np
import cv2
import pdb
from pathlib import Path
from tqdm import tqdm
from os.path import exists
from pytorch3d.loss import point_mesh_face_distance
from pytorch3d.structures import Meshes, Pointclouds
from smpl_registration.base_fitter import BaseFitter
from lib.body_objectives import batch_get_pose_obj, batch_3djoints_loss, batch_2djoints_loss
from lib.smpl.priors.th_smpl_prior import get_prior
from lib.smpl.priors.th_hand_prior import HandPrior
from lib.smpl.wrapper_pytorch import SMPLPyTorchWrapperBatchSplitParams

from Render_tools.smpl_render import kaolin_SMPL_renderer, rotate_points

global OPENCV_TO_OPENGL_CAMERA_CONVENTION
OPENCV_TO_OPENGL_CAMERA_CONVENTION = torch.tensor([[-1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, -1, 0],
                                                    [0, 0, 0, 1]]).float()
global init_camera_params


##modified from fit_SMPLH.py
class SMPLHOptimizer(BaseFitter):
    #we don't have scans, but we have mesh/SMPL/SMPLH parameters or 3D mesh/verts
    def __init__(self, model_root, device='cuda:0', save_name='smplh', debug=False, hands=False,
                keypoint_format='coco',keypoint_type='2d',img_size=(960,720),K=None):
        super(SMPLHOptimizer, self).__init__(model_root, device, save_name, debug, hands)
        if keypoint_format == 'coco':
            self.smpl_regressor = np.load("J_regressor_coco.npy") #17,6890
            self.smpl_regressor = torch.from_numpy(self.smpl_regressor).float().to(device)
        self.img_size = img_size #H,W
        self.camera_K = K
        self.kaolin_render=kaolin_SMPL_renderer()
        

    def perspective_projection(self, verts):
        #assume verts in B,N,3
        cam_t = torch.eye(4).to(self.device)
        cam_t[:2]*=-1
        T_wc = cam_t @ OPENCV_TO_OPENGL_CAMERA_CONVENTION.to(self.device)
        camera_transform = T_wc[:3,:].T
        camera_transform = camera_transform[None]

        #the camera transform should be updated according to the optimized one
        projected_verts = rotate_points(verts,  self.img_size[0], self.img_size[1],
                                        self.camera_K,camera_transform=camera_transform)

        projected_verts[:,:,0]*=self.img_size[1]/2
        projected_verts[:,:,0]+=self.img_size[1]/2
        projected_verts[:,:,1]*=-self.img_size[0]/2
        projected_verts[:,:,1]+=self.img_size[0]/2
        
        # projected_format_joints = projected_format_joints[0].detach().cpu().numpy()
        return projected_verts

    def kaolin_smpl_render_image(self, smpl, K, img):
        verts, _, _, _ = smpl()
        cam_t = torch.eye(4).cuda()
        cam_t[:2]*=-1
        fitting_camera_params = {
                    'intrinsic': K,
                    'extrinsic': cam_t.cpu().detach().numpy(),
                    "img_size": (img.shape[0], img.shape[1]) #h,w
                }
        rendered_img = self.kaolin_render.render_smpl_mesh(verts, fitting_camera_params, img.copy())
        return rendered_img[:,:,::-1]*255
        # cv2.imwrite(args.image_path.replace(".jpg", "optimized_MESH.jpg"), OPT_rendered_img[:,:,::-1]*255)

    def fit(self, smpl_params_file, pose_files, gender='neutral', save_path=None,
                keypoint_format='coco',keypoint_type='2d',img_path=None):
        # Batch size
        batch_sz = 1

        # Load scans and center them. Once smpl is registered, move it accordingly.
        # th_scan_meshes, centers = self.load_scans(scans, ret_cent=True)

        # Set optimization hyper parameters
        iterations, pose_iterations, steps_per_iter, pose_steps_per_iter = 1, 3, 30, 30

        th_pose_3d = None
        input_img = cv2.imread(img_path)
        #can load from smpl params
        if smpl_params_file is not None:
            # pose, betas, trans = self.load_smpl_params(smpl_params_file)
            with open(args.smpl_file, 'rb') as f:
                smpl_params = pkl.load(f)
            pose=torch.tensor(smpl_params['pose']).reshape(-1,66).to(self.device)
            if smpl_params['left_hand_pose'] is not None:
                left_hand_pose=torch.tensor(smpl_params['left_hand_pose']).reshape(-1,45).to(self.device)
            else:
                print("left hand pose is none")
                left_hand_pose=torch.zeros(1,45).to(self.device)
            if smpl_params['right_hand_pose'] is not None:
                right_hand_pose=torch.tensor(smpl_params['right_hand_pose']).reshape(-1,45).to(self.device)
            else:
                print("right hand pose is none")
                right_hand_pose=torch.zeros(1,45).to(self.device)
            #load hand pkl file if exists
            mano_file_path = args.smpl_file.replace("smplh.pkl", "MANO_hand.pkl")
            #check if the file exists
            if os.path.exists(mano_file_path):
                print("MANO pkl exist, loading")
                with open(mano_file_path, 'rb') as f:
                    mano_params = pkl.load(f)
                    hand_keypoints = mano_params['hand_keyp_arr']
                    #check which hand is the keypoint
                    hand_side = mano_params['right']
                    print("hand_keypoints", hand_keypoints.shape, "hand_side", hand_side)
                #plot the hand keypoints
                hand_keypoints_img = input_img.copy()
                for i in range(hand_keypoints.shape[0]):
                    for j in range(21):
                        cv2.circle(hand_keypoints_img, (int(hand_keypoints[i,j,0]), int(hand_keypoints[i,j,1])), 2, (0, 128, 255), -1)
                
            else:
                print("MANO pkl does not exist, using zero hand pose")
                hand_keypoints = None
                hand_side = None
            # pdb.set_trace()
            body_pose=torch.cat([pose,left_hand_pose[:,:3],right_hand_pose[:,:3]],dim=1)
            total_pose=torch.cat([pose,left_hand_pose,right_hand_pose],dim=1)
            betas=smpl_params['betas'].to(self.device)
            trans=torch.tensor(smpl_params['trans']).reshape(-1,3).to(self.device)
            # print("total_pose", total_pose.shape, "betas", betas.shape,trans.shape)
            # trans = torch.tensor(trans)
            # init smpl
            smpl = self.init_smpl(batch_sz, gender, total_pose, betas, trans)
            verts, jtr, tposed, naked = smpl()
            print("verts", verts.shape, "jtr", jtr.shape, "tposed", tposed.shape, "naked", naked.shape)
            if self.hands and os.path.exists(mano_file_path):
                self.hand_regressor=smpl.hand_reg_torch.to(self.device)
                smplh_3d_hand_keypoints=torch.bmm(self.hand_regressor,verts)
                smplh_2d_hand_keypoints = self.perspective_projection(smplh_3d_hand_keypoints)
                print("smplh_3d_hand_keypoints", smplh_3d_hand_keypoints.shape, "smplh_2d_hand_keypoints", smplh_2d_hand_keypoints.shape)
                for i in range(smplh_2d_hand_keypoints.shape[0]):
                    for j in range(42):
                        cv2.circle(hand_keypoints_img, (int(smplh_2d_hand_keypoints[i,j,0]), int(smplh_2d_hand_keypoints[i,j,1])), 2, (255, 0, 128), -1)
                
            cv2.imwrite(img_path.replace(".jpg", "hand_keypoints.jpg"), hand_keypoints_img)
            
###################################### test the init smpl model
            
            format_joints = self.smpl_regressor@verts #bs,17,3
            projected_format_joints=self.perspective_projection(format_joints)
            projected_format_joints = projected_format_joints[0].detach().cpu().numpy()
            ###################

            rendered_img=self.kaolin_smpl_render_image(smpl, self.camera_K, input_img)
            cv2.imwrite(img_path.replace(".jpg", "init_smpl_before_optimized.jpg"), rendered_img)
            
            for j in range(len(projected_format_joints)):
                cv2.circle(rendered_img,(int(projected_format_joints[j][0]),int(projected_format_joints[j][1])),5,(255,0,0),-1)
                # cv2.circle(rendered_img,(int(th_pose_2d[0][j][0]),int(th_pose_2d[0][j][1])),5,(255,0,128),-1)
            save_path = os.path.join("/home/wenbo/RVH_Mesh_Registration/opt_demo_file/debug_image",img_path.split("/")[-1])
            cv2.imwrite(save_path.replace(".jpg", "_before_optimized_2d_kpts.jpg"), rendered_img)
            print("finished init smpl")
        if pose_files is not None:
            
            # batch_sz = len(scans)
            # init smpl
            #suppose to have a smpl file, either from the previous fitting or from the initial smpl
            #need to load from npz file, we only have 2d joints
            th_pose_2d_coco = np.load(pose_files)['coco_order'].reshape(batch_sz,-1,3)#17,3 x,y,vis
            th_pose_2d_coco = torch.tensor(th_pose_2d_coco).float().to(self.device)
            print("th_pose_2d_coco", th_pose_2d_coco.shape)
            # Optimize pose first
        if self.hands and os.path.exists(mano_file_path):
            hand_keypoints = torch.tensor(hand_keypoints).float().to(self.device)
            # mark left,right hand existance [T,F] 
            hand_side_exist = torch.tensor([smpl_params['left_hand_pose'] is not None,smpl_params['right_hand_pose'] is not None]).float().to(self.device)
            print("hand_keypoints", hand_keypoints, "hand_side_exist", hand_side_exist)
        else:
            hand_keypoints = None
            hand_side_exist = None
        # pdb.set_trace()
        print("Optimizing SMPL pose only")
        self.optimize_pose_only( smpl, pose_iterations, pose_steps_per_iter, th_pose_2d_coco,
                                    keypoint_format=keypoint_format,keypoint_type=keypoint_type,img_path=img_path,
                                    hand_2d_keypoints=hand_keypoints,hand_side_exist=hand_side_exist)
        
        return smpl
        # Optimize pose and shape
        # self.optimize_pose_shape(th_scan_meshes, smpl, iterations, steps_per_iter, th_pose_3d)

        # if save_path is not None:
        #     if not exists(save_path):
        #         os.makedirs(save_path)
        #     return self.save_outputs(save_path, scans, smpl, th_scan_meshes, save_name='smplh' if self.hands else 'smpl')

    def optimize_pose_shape(self, th_scan_meshes,smpl, iterations, steps_per_iter, th_pose_3d=None):
        #we don't have scans, but we have mesh/SMPL/SMPLH parameters or 3D mesh/verts
        # Optimizer
        optimizer = torch.optim.Adam([smpl.trans, smpl.betas, smpl.pose], 0.02, betas=(0.9, 0.999))
        # Get loss_weights
        weight_dict = self.get_loss_weights()

        for it in range(iterations):
            loop = tqdm(range(steps_per_iter))
            loop.set_description('Optimizing SMPL')
            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                loss_dict = self.forward_pose_shape(th_scan_meshes, smpl, th_pose_3d)
                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                tot_loss.backward()
                optimizer.step()

                l_str = 'Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                    loop.set_description(l_str)

                # if self.debug:
                #     self.viz_fitting(smpl, th_scan_meshes)

        print('** Optimised smpl pose and shape **')

    def forward_pose_shape(self, th_scan_meshes, smpl, th_pose_3d=None):
        # Get pose prior
        prior = get_prior(self.model_root, smpl.gender)

        # forward
        verts, _, _, _ = smpl()
        th_smpl_meshes = Meshes(verts=verts, faces=torch.stack([smpl.faces] * len(verts), dim=0))

        # losses
        loss = dict()
        loss['s2m'] = point_mesh_face_distance(th_smpl_meshes, Pointclouds(points=th_scan_meshes.verts_list()))
        loss['m2s'] = point_mesh_face_distance(th_scan_meshes, Pointclouds(points=th_smpl_meshes.verts_list()))
        loss['betas'] = torch.mean(smpl.betas ** 2)
        loss['pose_pr'] = torch.mean(prior(smpl.pose))
        if self.hands:
            hand_prior = HandPrior(self.model_root, type='grab')
            loss['hand'] = torch.mean(hand_prior(smpl.pose)) # add hand prior if smplh is used
        if th_pose_3d is not None:
            # 3D joints loss
            J, face, hands = smpl.get_landmarks()
            joints = self.compose_smpl_joints(J, face, hands, th_pose_3d)
            j3d_loss = batch_3djoints_loss(th_pose_3d, joints)
            loss['pose_obj'] = j3d_loss
            # loss['pose_obj'] = batch_get_pose_obj(th_pose_3d, smpl).mean()
        return loss

    def compose_smpl_joints(self, J, face, hands, th_pose_3d):
        if th_pose_3d.shape[1] == 25:
            joints = J
        else:
            joints = torch.cat([J, face, hands], 1)
        return joints

    def optimize_pose_only(self, smpl, iterations,
                           steps_per_iter, th_pose_2d, prior_weight=None,
                           keypoint_format='coco',keypoint_type='2d',img_path=None,
                           hand_2d_keypoints=None,hand_side_exist=None): 
        # we could only use the 2D joints to optimize the pose,in coco order
        """they didn't implement from_smplh method"""
        # split_smpl = SMPLHPyTorchWrapperBatchSplitParams.from_smplh(smpl).to(self.device)
        split_smpl = SMPLPyTorchWrapperBatchSplitParams.from_smpl(smpl).to(self.device)
        optimizer = torch.optim.Adam([split_smpl.trans, split_smpl.top_betas, split_smpl.global_pose], 0.02,
                                     betas=(0.9, 0.999))
        # pdb.set_trace()
        # Get loss_weights
        weight_dict = self.get_loss_weights()

        iter_for_global = 1
        th_pose_2d_numpy = th_pose_2d.detach().cpu().numpy()
        if hand_2d_keypoints is not None:
            hand_2d_keypoints_numpy = hand_2d_keypoints.detach().cpu().numpy()
            hand_2d_keypoints_numpy = hand_2d_keypoints_numpy.reshape(-1,3)

        for it in range(iter_for_global + iterations):
            loop = tqdm(range(steps_per_iter))
            if it < iter_for_global:
                # Optimize global orientation
                print('Optimizing SMPL global orientation')
                loop.set_description('Optimizing SMPL global orientation')
            elif it == iter_for_global:
                # Now optimize full SMPL pose
                print('Optimizing SMPL pose only')
                loop.set_description('Optimizing SMPL pose only')
                print("split_smpl.body_pose", split_smpl.body_pose.shape)
                optimizer = torch.optim.Adam([split_smpl.trans, split_smpl.top_betas, split_smpl.global_pose,
                                              split_smpl.body_pose], 0.02, betas=(0.9, 0.999))
            else:
                loop.set_description('Optimizing SMPL pose only')

            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                if keypoint_type=='3d':
                    loss_dict = self.forward_step_pose_only(split_smpl, th_pose_2d, prior_weight)
                elif keypoint_type=='2d':
                    loss_dict,optimized_2d_kpts,smplh_2d_hand_keypoints = self.forward_step_2d_pose_only(split_smpl, th_pose_2d, prior_weight,
                        keypoint_format='coco',hand_2d_keypoints=hand_2d_keypoints,hand_side_exist=hand_side_exist)
                    input_img = cv2.imread(img_path)
                    optimized_2d_kpts = optimized_2d_kpts[0].detach().cpu().numpy()
                    
                    rendered_img=self.kaolin_smpl_render_image(split_smpl, self.camera_K, input_img)
                    for j in range(len(optimized_2d_kpts)):
                        cv2.circle(rendered_img,(int(optimized_2d_kpts[j][0]),int(optimized_2d_kpts[j][1])),5,(255,0,0),-1)
                        cv2.circle(rendered_img,(int(th_pose_2d[0][j][0]),int(th_pose_2d[0][j][1])),5,(0,0,255),-1)
                    if smplh_2d_hand_keypoints is not None:
                        smplh_2d_hand_keypoints = smplh_2d_hand_keypoints[0].detach().cpu().numpy()
                        # print("hand_2d_keypoints in optimize loop", hand_2d_keypoints_numpy.shape)
                        for j in range(len(smplh_2d_hand_keypoints)):
                            cv2.circle(rendered_img,(int(smplh_2d_hand_keypoints[j][0]),int(smplh_2d_hand_keypoints[j][1])),2,(255,0,0),-1)
                        for k in range(len(hand_2d_keypoints_numpy)):
                            cv2.circle(rendered_img,(int(hand_2d_keypoints_numpy[k][0]),int(hand_2d_keypoints_numpy[k][1])),2,(0,0,255),-1)
                        
                    save_path = os.path.join("/home/wenbo/RVH_Mesh_Registration/opt_demo_file/debug_image",img_path.split("/")[-1])
                    cv2.imwrite(save_path.replace(".jpg", "_{}_{}optimized_2d_kpts.jpg".format(it,i)), rendered_img,)
                    # pdb.set_trace()
                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it/2)
                tot_loss.backward()
                optimizer.step()

                l_str = 'Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                    loop.set_description(l_str)

                # if self.debug:
                #     self.viz_fitting(split_smpl, th_scan_meshes)

        self.copy_smpl_params(smpl, split_smpl)

        print('** Optimised smpl pose **')

    def copy_smpl_params(self, smpl, split_smpl):
        # Put back pose, shape and trans into original smpl
        smpl.pose.data = split_smpl.pose.data
        smpl.betas.data = split_smpl.betas.data
        smpl.trans.data = split_smpl.trans.data

    def forward_step_pose_only(self, smpl, th_pose_3d, prior_weight):
        """
        Performs a forward step, given smpl and scan meshes.
        Then computes the losses.
        currently no prior weight implemented for smplh
        """
        # Get pose prior
        prior = get_prior(self.model_root, smpl.gender)

        # losses
        loss = dict()
        # loss['pose_obj'] = batch_get_pose_obj(th_pose_3d, smpl, init_pose=False)
        # 3D joints loss
        J, face, hands = smpl.get_landmarks()
        joints = self.compose_smpl_joints(J, face, hands, th_pose_3d)
        loss['pose_pr'] = torch.mean(prior(smpl.pose))
        loss['betas'] = torch.mean(smpl.betas ** 2)
        j3d_loss = batch_3djoints_loss(th_pose_3d, joints)
        loss['pose_obj'] = j3d_loss
        return loss

    def forward_step_2d_pose_only(self, smpl, th_pose_2d, prior_weight,keypoint_format='coco',
                                    hand_2d_keypoints=None,hand_side_exist=None):
        """
        Performs a forward step, given smpl and scan meshes.
        Then computes the losses.
        currently no prior weight implemented for smplh
        """
        # Get pose prior
        prior = get_prior(self.model_root, smpl.gender)

        # losses
        loss = dict()
        # loss['pose_obj'] = batch_get_pose_obj(th_pose_3d, smpl, init_pose=False)
        # 3D joints loss
        J, face, hands = smpl.get_landmarks()
        # joints = self.compose_smpl_joints(J, face, hands, th_pose_3d)
        verts, jtr, tposed, naked = smpl()
        format_joints = torch.matmul(self.smpl_regressor, verts)
        projected_format_joints = self.perspective_projection(format_joints)
        loss['pose_pr'] = torch.mean(prior(smpl.pose))
        loss['betas'] = torch.mean(smpl.betas ** 2)
        j2d_loss = batch_2djoints_loss(th_pose_2d, projected_format_joints)
        loss['pose_obj'] = j2d_loss
        if self.hands:
            hand_prior = HandPrior(self.model_root, type='grab')
            loss['hand'] = torch.mean(hand_prior(smpl.pose)) # add hand prior if smplh is used
            #check if the hand keypoints are valid
            if hand_2d_keypoints is not None and hand_side_exist is not None:
                # print("in forward_step_2d_pose_only hand loss")
                # print("hand_2d_keypoints", hand_2d_keypoints.shape, "hand_side_exist", hand_side_exist)
                hand_2d_keypoints_both_hands = torch.zeros(1,42,3).to(self.device)
                if sum(hand_side_exist) == 2:
                    hand_2d_keypoints_both_hands=hand_2d_keypoints
                elif hand_side_exist[0]:
                    hand_2d_keypoints_both_hands[0,:21] = hand_2d_keypoints[0]
                elif hand_side_exist[1]:
                    hand_2d_keypoints_both_hands[0,21:] = hand_2d_keypoints[0]
                
                smplh_3d_hand_keypoints=torch.bmm(self.hand_regressor,verts)
                smplh_2d_hand_keypoints = self.perspective_projection(smplh_3d_hand_keypoints)
                hand_loss = batch_2djoints_loss(hand_2d_keypoints_both_hands, smplh_2d_hand_keypoints)
                loss['hand_kpt'] = hand_loss
        else:
            smplh_2d_hand_keypoints=None

        return loss, projected_format_joints, smplh_2d_hand_keypoints

    def get_loss_weights(self):
        """Set loss weights"""
        loss_weight = {'s2m': lambda cst, it: 20. ** 2 * cst * (1 + it),
                       'm2s': lambda cst, it: 20. ** 2 * cst / (1 + it),
                       'betas': lambda cst, it: 10. ** 1.0 * cst / (1 + it),
                       'offsets': lambda cst, it: 10. ** -1 * cst / (1 + it),
                       'pose_pr': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'hand': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'hand_kpt': lambda cst, it: 10. ** 1 * cst / (1 + it),
                       'lap': lambda cst, it: cst / (1 + it),
                       'pose_obj': lambda cst, it: 10. ** 2 * cst / (1 + it)
                       }
        return loss_weight


def main(args):
    # smpl_params= pkl.load(open(args.smpl_file, 'rb'))
    # mano_files = pkl.load(open(args.mano_file, 'rb'))
    with open(args.smpl_file, 'rb') as f:
        smpl_params = pkl.load(f)
    with open(args.mano_file, 'rb') as f:
        mano_files = pkl.load(f)
    img = cv2.imread(args.image_path)
    img_cv2 = img.copy()
    K= np.asarray([[714.2279663085938, 0, 0],
                    [0, 714.2279663085938, 0], 
                    [360.54248046875, 483.0450134277344, 1]]).T
            
    #draw bbox for body and hand
    print("smpl_params", smpl_params.keys())
    #draw bbox for body
    batch_sz = len(smpl_params['bbox_center'])
    for i in range(batch_sz):
        body_bbox_center = smpl_params['bbox_center'][i]
        body_bbox_size = smpl_params['bbox_size'][i]
        print("body_bbox_center", body_bbox_center)
        print("body_bbox_size", body_bbox_size)
        #assume square bbox
        cv2.rectangle(img, (int(body_bbox_center[0]-body_bbox_size/2), int(body_bbox_center[1]-body_bbox_size/2)),
                        (int(body_bbox_center[0]+body_bbox_size/2), int(body_bbox_center[1]+body_bbox_size/2)), (255, 0, 0), 2)

        print("mano_files", mano_files.keys())
        #draw bbox for hand
        hand_count=len(mano_files['hand_boxes'])
        for j in range(hand_count):
            hand_bbox_xyxy = mano_files['hand_boxes'][j]
            print("hand_bbox_xyxy", hand_bbox_xyxy)
            cv2.rectangle(img, (int(hand_bbox_xyxy[0]), int(hand_bbox_xyxy[1])),
                        (int(hand_bbox_xyxy[2]), int(hand_bbox_xyxy[3])), (0, 255, 0), 2)
        # print("hand_bbox_center", hand_bbox_center)
    output_path = args.image_path.replace(".jpg", "_body_hand_read_bbox.jpg")
    cv2.imwrite(output_path, img)
    kaolin_render=kaolin_SMPL_renderer()
    np_cam_t = np.eye(4)
    np_cam_t[:3,3] = smpl_params["trans"]
    np_cam_t[0,3] *= -1
    print("init np_cam_t", np_cam_t,"image size", img_cv2.shape)
    # global init_camera_params
    init_camera_params = {
                'intrinsic': K,
                'extrinsic': np_cam_t,
                "img_size": (img_cv2.shape[0], img_cv2.shape[1])#h,w
            }
    cam_view = kaolin_render.render_smpl_mesh(smpl_params["vertices"], init_camera_params, img_cv2.copy())
    output_path = args.image_path.replace(".jpg", "ReRender.jpg")
    cv2.imwrite(output_path, 255*cam_view[:, :, ::-1])
    fitter = SMPLHOptimizer(args.model_root, debug=args.display, hands=args.hands,img_size=(img_cv2.shape[0], img_cv2.shape[1]),K=K)
    optimized_smpl= fitter.fit(args.smpl_file, args.pose_file, args.gender, args.save_path, img_path=args.image_path)
    #####################################################render the optimized smpl
    OPT_rendered_img=fitter.kaolin_smpl_render_image(optimized_smpl, K, img_cv2)
    # OPT_rendered_img = kaolin_render.render_smpl_mesh(verts, fitting_camera_params, img_cv2.copy())
    cv2.imwrite(args.image_path.replace(".jpg", "optimized_MESH.jpg"), OPT_rendered_img)

if __name__ == "__main__":
    #this is under the singel person setting
    import argparse
    from utils.configs import load_config
    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument('--image_path', type=str, help='path to the 3d scans')
    parser.add_argument('--smpl_file', default="593_smplh.pkl",type=str, help='smpl body file')
    parser.add_argument('--mano_file', default="593_MANO_hand.pkl",type=str, help='mano hand file')
    parser.add_argument('--pose_file', default="593_2dkeypoints.npz",type=str, help='mano hand file')
    parser.add_argument('--save_path', type=str, help='save path for all scans')
    parser.add_argument('--gender',  default='male',type=str)  # can be female
    parser.add_argument('--display', default=False, action='store_true')
    parser.add_argument("--config-path", "-c", type=Path, default="config.yml",
                        help="Path to yml file with config")
    parser.add_argument('-hands', default=True, action='store_true', help='use SMPL+hand model or not')
    args = parser.parse_args()

    # args.scan_path = 'data/mesh_1/scan.obj'
    # args.pose_file = 'data/mesh_1/3D_joints_all.json'
    # args.display = True
    # args.save_path = 'data/mesh_1'
    # args.gender = 'male'
    config = load_config(args.config_path)
    args.model_root = Path(config["SMPL_MODELS_PATH"])

    main(args)