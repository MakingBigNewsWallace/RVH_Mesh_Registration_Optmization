import os
import torch
import kaolin as kal
import cv2
import numpy as np
import einops
import PIL
import time
from kaolin import ops
from kaolin.render import camera as kaolin_camera
import smplx
from scipy.spatial.transform import Rotation
import tqdm

import trimesh
import copy

def aa_to_rotmat(theta: torch.Tensor):
    """
    Convert axis-angle representation to rotation matrix.
    Works by first converting it to a quaternion.
    Args:
        theta (torch.Tensor): Tensor of shape (B, 3) containing axis-angle representations.
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion representation to rotation matrix.
    Args:
        quat (torch.Tensor) of shape (B, 4); 4 <===> (w, x, y, z).
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def prepare_vertices(vertices, faces, 
                     width, height, K, camera_rot=None, camera_trans=None,
                     camera_transform=None):
    r"""Wrapper function to move and project vertices to cameras then index them with faces.

    Args:
        vertices (torch.Tensor):
            the meshes vertices, of shape :math:`(\text{batch_size}, \text{num_vertices}, 3)`.
        faces (torch.LongTensor):
            the meshes faces, of shape :math:`(\text{num_faces}, \text{face_size})`.
        camera_proj (torch.Tensor):
            the camera projection vector, of shape :math:`(3, 1)`.
        camera_rot (torch.Tensor, optional):
            the camera rotation matrices,
            of shape :math:`(\text{batch_size}, 3, 3)`.
        camera_trans (torch.Tensor, optional):
            the camera translation vectors,
            of  shape :math:`(\text{batch_size}, 3)`.
        camera_transform (torch.Tensor, optional):
            the camera transformation matrices,
            of shape :math:`(\text{batch_size}, 4, 3)`.
            Replace `camera_trans` and `camera_rot`.
    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor):
            The vertices in camera coordinate indexed by faces,
            of shape :math:`(\text{batch_size}, \text{num_faces}, \text{face_size}, 3)`.
            The vertices in camera plan coordinate indexed by faces,
            of shape :math:`(\text{batch_size}, \text{num_faces}, \text{face_size}, 2)`.
            The face normals, of shape :math:`(\text{batch_size}, \text{num_faces}, 3)`.
    """
    # Apply the transformation from camera_rot and camera_trans or camera_transform
    if camera_transform is None:
        assert camera_trans is not None and camera_rot is not None, \
            "camera_transform or camera_trans and camera_rot must be defined"
        vertices_camera = camera.rotate_translate_points(vertices, camera_rot,
                                                         camera_trans)
    else:
        assert camera_trans is None and camera_rot is None, \
            "camera_trans and camera_rot must be None when camera_transform is defined"
        padded_vertices = torch.nn.functional.pad(
            vertices, (0, 1), mode='constant', value=1.
        )
        vertices_camera = (padded_vertices @ camera_transform)
    
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    intrinsics = kaolin_camera.PinholeIntrinsics.from_focal(width, height,
                                                            focal_x=fx, focal_y=fy,
                                                            x0=cx-width/2, y0=cy-height/2,
                                                            ).to(vertices.device).float()
    vertices_image = intrinsics.transform(vertices_camera)[..., :2]
    
    face_vertices_camera = ops.mesh.index_vertices_by_faces(vertices_camera, faces)
    face_vertices_image = ops.mesh.index_vertices_by_faces(vertices_image, faces)
    face_normals = ops.mesh.face_normals(face_vertices_camera, unit=True)
    return face_vertices_camera, face_vertices_image, face_normals


def load_smpl_param(path):
    smpl_params = dict(np.load(str(path)))
    if "thetas" in smpl_params:
        smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]
    return {
        "betas": smpl_params["betas"].astype(np.float32).reshape(1, 10),
        "body_pose": smpl_params["body_pose"].astype(np.float32),
        "global_orient": smpl_params["global_orient"].astype(np.float32),
        "transl": smpl_params["transl"].astype(np.float32),
    }

def coordinate_transform(pose, original, target):
    if original == 'opengl':
        if target == 'unity':
            pose[2] *= -1
        elif target == 'blender':
            pose[2] *= -1
            pose[[1, 2]] = pose[[2, 1]]
        elif target in ['opencv', 'colmap']:
            pose[1:3] *= -1
    elif original == 'unity':
        if target == 'opengl':
            pose[2] *= -1
        elif target == 'blender':
            pose[[1, 2]] = pose[[2, 1]]
        elif target in ['opencv', 'colmap']:
            pose[1] *= -1
    elif original == 'blender':
        if target == 'opengl':
            pose[1] *= -1
            pose[[1, 2]] = pose[[2, 1]]
        elif target == 'unity':
            pose[[1, 2]] = pose[[2, 1]]
        elif target in ['opencv', 'colmap']:
            pose[2] *= -1
            pose[[1, 2]] = pose[[2, 1]]
    elif original in ['opencv', 'colmap']:
        if target == 'opengl':
            pose[1:3] *= -1
        elif target == 'unity':
            pose[1] *= -1
        elif target == 'blender':
            pose[1] *= -1
            pose[[1, 2]] = pose[[2, 1]]
    return pose


device = 'cuda'
obj_path = "smpl_assets/smplx_uv/smplx_uv.obj"
texture_path = "smpl_assets/smplx_uv/smplx_uv.png"

## load mesh
curr_mesh = kal.io.obj.import_mesh(obj_path, with_materials=False)
mesh = kal.io.obj.import_mesh(obj_path, with_materials=True)
vertices = curr_mesh.vertices.to(device)
faces = curr_mesh.faces.to(device)
## load from SMPLx parameters

avatar_id = "avatarrex_zzr"
# avatar_id = "avatarrex_lbn2"

pose_id = "thuman4__pose_00_free_view"
# pose_id = "thuman4__pose_01_free_view"
# pose_id = "thuman4__pose_02_free_view"

batch_id = "batch_700000"
# batch_id = "batch_800000"

root = f"/root/autodl-tmp/datazy/AnimatableGaussians/test_results/{avatar_id}/avatar/{pose_id}/{batch_id}/pca_20_sigma_2.00"
image_dir = os.path.join(root, "rgb_map")
cam_dir = os.path.join(root, "cam_params")

avatar_smpl_pose_path = "/root/autodl-tmp/datazy/AnimatableGaussians/datas/avatarrex/zzr/smpl_params.npz"
# avatar_smpl_pose_path = "/root/autodl-tmp/datazy/AnimatableGaussians/datas/avatarrex/lbn2/smpl_params.npz"
avatar_smpl_data = np.load(avatar_smpl_pose_path)
avatar_smpl_data = dict(avatar_smpl_data)
betas = avatar_smpl_data["betas"]

smpl_pose_path = f"/root/autodl-tmp/datazy/AnimatableGaussians/datas/thuman4/pose_00.npz"
# smpl_pose_path = f"/root/autodl-tmp/datazy/AnimatableGaussians/datas/thuman4/pose_01.npz"
# smpl_pose_path = f"/root/autodl-tmp/datazy/AnimatableGaussians/datas/thuman4/pose_02.npz"
smplx_params = dict(np.load(smpl_pose_path))
# betas = smplx_params["betas"]
transl = smplx_params["transl"]
global_orient = smplx_params["global_orient"]
body_pose = smplx_params["body_pose"]
left_hand_pose = smplx_params["left_hand_pose"]
right_hand_pose = smplx_params["right_hand_pose"]

print("global_orient.shape: ", global_orient.shape)
print("body_pose.shape: ", body_pose.shape)
print("betas.shape: ", betas.shape)
print("transl.shape: ", transl.shape)
print("left_hand_pose.shape: ", left_hand_pose.shape)
print("right_hand_pose.shape", right_hand_pose.shape)

model_path = "/root/autodl-tmp/datazy/Render_tools/smpl_assets/smplx_model"
smpl_model = smplx.SMPLX(model_path = model_path, gender = "neutral", 
                         use_pca = False, num_pca_comps = 45, flat_hand_mean = True, 
                         batch_size = 1).cuda().eval()

## load texture
texture_map = PIL.Image.open(texture_path)
texture_map = np.array(texture_map,dtype=np.float32) / 255.0
texture_map = torch.tensor(texture_map,dtype=torch.float).to(device).permute(2,0,1).unsqueeze(0)

image_list = sorted(os.listdir(image_dir))
cam_list = sorted(os.listdir(cam_dir))
print("image count: ", len(image_list))
assert len(image_list) == len(cam_list)

# original_list = ["opengl", "blender", "opencv", "colmap"]
# target_list = ["opengl", "blender", "opencv", "colmap"]
# original_list = ["opencv"]
# target_list = ["opengl"]

original = "opencv"
target = "opengl"

for framed_index in tqdm.tqdm(range(min(len(image_list), 800))):
    curr_cam_path = os.path.join(cam_dir, cam_list[framed_index])
    curr_cam = dict(np.load(curr_cam_path))
    height = curr_cam["img_h"]
    width = curr_cam["img_w"]
    K = curr_cam["intr"]
    cam_extr = curr_cam["extr"]
    H = height
    W = width

    # T_wc_ori = np.linalg.inv(cam_extr)
    T_wc = cam_extr
    T_wc = coordinate_transform(T_wc, original, target)

    # OpenCV coordinate to OpenGL coordinate
    K[1,2] = height - K[1,2]

    curr_global_orient = torch.from_numpy(global_orient[framed_index:framed_index+1]).float().to(device)
    curr_body_pose = torch.from_numpy(body_pose[framed_index:framed_index+1]).float().to(device)
    curr_betas = torch.from_numpy(betas).float().to(device)
    curr_transl = torch.from_numpy(transl[framed_index:framed_index+1]).float().to(device)
    curr_left_hand_pose = torch.from_numpy(left_hand_pose[framed_index:framed_index+1]).float().to(device)
    curr_right_hand_pose = torch.from_numpy(right_hand_pose[framed_index:framed_index+1]).float().to(device)

    smpl_output = smpl_model.forward(betas=curr_betas,
                                    global_orient=curr_global_orient,
                                    body_pose=curr_body_pose,
                                    transl=curr_transl,
                                    left_hand_pose=curr_left_hand_pose,
                                    right_hand_pose=curr_right_hand_pose)
    vertices = smpl_output.vertices[0]

    ### debug
    # vertices_np = smpl_output.vertices.detach().cpu().numpy()[0]
    # faces_np = smpl_model.faces
    # mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np)
    # mesh.export('z_debug_smplx_mesh_v2.obj')
    ### debug

    # ## set camera
    R = T_wc[:3, :3]
    T = T_wc[:3, 3]
    camera_transform = torch.from_numpy(np.concatenate([R.T, T[None]], axis=0)).to(device).float()
    camera_transform = camera_transform[None]

    ## get camera-based vertices
    face_vertices_camera, face_vertices_image, face_normals = prepare_vertices(vertices, faces, 
                                                                    W, H, K, 
                                                                    camera_transform=camera_transform)

    # print("mesh.face_uvs: ", mesh.face_uvs)
    num_faces = faces.shape[0]
    face_attributes = [
        mesh.face_uvs.repeat(1, 1, 1, 1).to(device),
        torch.ones((1, num_faces, 3, 1), device='cuda')
    ]
    image_features, face_idx = kal.render.mesh.rasterize(H, W, 
                                                        face_vertices_camera[:, :, :, -1],  
                                                        face_vertices_image, 
                                                        face_attributes, 
                                                        valid_faces=None, 
                                                        multiplier=None, 
                                                        eps=None, 
                                                        # backend='nvdiffrast',
                                                        backend='cuda',
                                                        )

    ## set texture
    start_time = time.time()
    texture_coords, mask = image_features
    image = kal.render.mesh.texture_mapping(texture_coords, 
                                            texture_map.repeat(1, 1, 1, 1), 
                                            mode='bilinear')
    image = torch.clamp(image, 0.0, 1.0)
    mask = torch.clamp(mask, 0.0, 1.0) > 0

    ## set light
    image_normals = face_normals[:, face_idx].squeeze(0)
    lights = torch.tensor([[1.0, 0.0, 1.0, 1.0, 0.0, 0, 1.0, 0.0, .5]], device=device)
    image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, lights)
    image = image[:, :, :, :3] * einops.repeat(image_lighting, 'a b c -> () b c (repeat a)', repeat=3).to(device)
    image = torch.clamp(image, 0.0, 1.0)

    ## conver type
    image_np = image[0].cpu().detach().numpy() * 255
    mask_np = mask[0].cpu().detach().numpy() * 255
    mesh_np = image_np[:, :, ::-1].astype(np.uint8)

    # id_str = str(framed_index).zfill(4)
    id_str = image_list[framed_index][:-4]
    img_path = os.path.join(image_dir, image_list[framed_index])
    # img_path = os.path.join(root, "images", str(framed_index).zfill(8)+".png")
    human_img = cv2.imread(img_path)[:, :, :3]

    mask_onehot = mask[0].cpu().detach().numpy()
    final_image_np = mesh_np * mask_onehot + human_img * (1 - mask_onehot)
    final_image_np = np.concatenate([human_img, final_image_np], axis=1)
    # final_image_np = np.concatenate([human_img, mesh_np], axis=1)

    ## save image
    output_path = f"./output/avatar/{avatar_id}_{pose_id}/{id_str}.jpg"
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, final_image_np)

# generate video
frames = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
frames.sort()

output_video_path = os.path.join(output_dir, f"render_smpl_{avatar_id}_{pose_id}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video_writer = cv2.VideoWriter(output_video_path, fourcc, 25, (768, 384))

for frame in frames:
    frame_path = os.path.join(output_dir, frame)
    img = cv2.imread(frame_path)
    h,w = img.shape[0:2]
    img = cv2.resize(img, (768,384))
    video_writer.write(img)

video_writer.release()