import smplx
import numpy as np
import torch
import trimesh

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

smpl_model_path = "smpl_assets/smpl_model/SMPL_FEMALE.pkl"
smpl_model = smplx.SMPL(model_path=smpl_model_path, batch_size = 1).cuda().eval()

pose_path = "datas/cuhk_garment/female-anran-apose/poses_refined.npz"
smpl_params = load_smpl_param(pose_path)

betas = torch.tensor(smpl_params["betas"]).cuda()
global_orient = torch.tensor(smpl_params["global_orient"]).cuda()
body_poses = torch.tensor(smpl_params["body_pose"]).cuda()
trans = torch.tensor(smpl_params["transl"]).cuda()

live_smpl = smpl_model.forward(betas=betas,
                global_orient=global_orient[0:1],
                body_pose=body_poses[0:1],
                transl=trans[0:1])
vertices = live_smpl.vertices.detach().cpu().numpy()[0]
faces = smpl_model.faces
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

mesh.export('z_debug_smpl_mesh.obj')