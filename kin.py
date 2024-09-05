import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def quatmul(a, b):
    return torch.cat([
        a[..., 0:1]*b[..., 0:1] - a[..., 1:2]*b[..., 1:2] - a[..., 2:3]*b[..., 2:3] - a[..., 3:4]*b[..., 3:4],
        a[..., 0:1]*b[..., 1:2] + a[..., 1:2]*b[..., 0:1] + a[..., 2:3]*b[..., 3:4] - a[..., 3:4]*b[..., 2:3],
        a[..., 0:1]*b[..., 2:3] - a[..., 1:2]*b[..., 3:4] + a[..., 2:3]*b[..., 0:1] + a[..., 3:4]*b[..., 1:2],
        a[..., 0:1]*b[..., 3:4] + a[..., 1:2]*b[..., 2:3] - a[..., 2:3]*b[..., 1:2] + a[..., 3:4]*b[..., 0:1]
    ], dim=-1)

def quat2mat(q):
    qw, qx, qy, qz = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]
    mat_3x3 = torch.stack([
        1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw,
        2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw,
        2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))
    
    mat_4x4 = torch.eye(4, dtype=mat_3x3.dtype, device=device).repeat(*q.shape[:-1], 1, 1)
    mat_4x4[..., :3, :3] = mat_3x3
    return mat_4x4

def ttmat_fn(t):
    mat = torch.eye(4, dtype=t.dtype, device=device).repeat(1, 1, 1)
    mat[0, :3, 3] = t
    return mat

def rxtmat_fn(angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack([
        torch.ones_like(angle), torch.zeros_like(angle), torch.zeros_like(angle), torch.zeros_like(angle),
        torch.zeros_like(angle), c, -s, torch.zeros_like(angle),
        torch.zeros_like(angle), s, c, torch.zeros_like(angle),
        torch.zeros_like(angle), torch.zeros_like(angle), torch.zeros_like(angle), torch.ones_like(angle)
    ], dim=-1).reshape(*angle.shape, 4, 4)

def rytmat_fn(angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack([
        c, torch.zeros_like(angle), s, torch.zeros_like(angle),
        torch.zeros_like(angle), torch.ones_like(angle), torch.zeros_like(angle), torch.zeros_like(angle),
        -s, torch.zeros_like(angle), c, torch.zeros_like(angle),
        torch.zeros_like(angle), torch.zeros_like(angle), torch.zeros_like(angle), torch.ones_like(angle)
    ], dim=-1).reshape(*angle.shape, 4, 4)

def rztmat_fn(angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack([
        c, -s, torch.zeros_like(angle), torch.zeros_like(angle),
        s, c, torch.zeros_like(angle), torch.zeros_like(angle),
        torch.zeros_like(angle), torch.zeros_like(angle), torch.ones_like(angle), torch.zeros_like(angle),
        torch.zeros_like(angle), torch.zeros_like(angle), torch.zeros_like(angle), torch.ones_like(angle)
    ], dim=-1).reshape(*angle.shape, 4, 4)

def quatmul_mat_fn(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return torch.stack([
        [qw, -qx, -qy, -qz],
        [qx,  qw, -qz,  qy],
        [qy,  qz,  qw, -qx],
        [qz, -qy,  qx,  qw]
    ], dim=-1).to(dtype=q.dtype, device=device)

# Palm transformation
palm_quat = torch.as_tensor([0., 1., 0., 1.], dtype=torch.float32, device=device)
t_palm = quat2mat(palm_quat / torch.norm(palm_quat))

def fftp_pos_fd_fn(ff_qpos):
    ff_t_base = t_palm @ ttmat_fn(torch.as_tensor([0., 0.0435, -0.001542], dtype=torch.float32, device=device)) @ quat2mat(torch.as_tensor([0.999048, -0.0436194, 0., 0.], dtype=torch.float32, device=device))
    ff_t_proximal = ff_t_base @ rztmat_fn(ff_qpos[..., 0]) @ ttmat_fn(torch.as_tensor([0., 0., 0.0164], dtype=torch.float32, device=device))
    ff_t_medial = ff_t_proximal @ rytmat_fn(ff_qpos[..., 1]) @ ttmat_fn(torch.as_tensor([0., 0., 0.054], dtype=torch.float32, device=device))
    ff_t_distal = ff_t_medial @ rytmat_fn(ff_qpos[..., 2]) @ ttmat_fn(torch.as_tensor([0., 0., 0.0384], dtype=torch.float32, device=device))
    ff_t_ftp = ff_t_distal @ rytmat_fn(ff_qpos[..., 3]) @ ttmat_fn(torch.as_tensor([0., 0., 0.0384], dtype=torch.float32, device=device))
    return ff_t_ftp[..., :3, 3]

def mftp_pos_fd_fn(mf_qpos):
    mf_t_base = t_palm @ ttmat_fn(torch.as_tensor([0., 0., 0.0007], dtype=torch.float32, device=device))
    mf_t_proximal = mf_t_base @ rztmat_fn(mf_qpos[..., 0]) @ ttmat_fn(torch.as_tensor([0., 0., 0.0164], dtype=torch.float32, device=device))
    mf_t_medial = mf_t_proximal @ rytmat_fn(mf_qpos[..., 1]) @ ttmat_fn(torch.as_tensor([0., 0., 0.054], dtype=torch.float32, device=device))
    mf_t_distal = mf_t_medial @ rytmat_fn(mf_qpos[..., 2]) @ ttmat_fn(torch.as_tensor([0., 0., 0.0384], dtype=torch.float32, device=device))
    mf_t_ftp = mf_t_distal @ rytmat_fn(mf_qpos[..., 3]) @ ttmat_fn(torch.as_tensor([0., 0., 0.0384], dtype=torch.float32, device=device))
    return mf_t_ftp[..., :3, 3]

def rftp_pos_fd_fn(rf_qpos):
    rf_t_base = t_palm @ ttmat_fn(torch.as_tensor([0., -0.0435, -0.001542], dtype=torch.float32, device=device)) @ quat2mat(torch.as_tensor([0.999048, 0.0436194, 0., 0.], dtype=torch.float32, device=device))
    rf_t_proximal = rf_t_base @ rztmat_fn(rf_qpos[..., 0]) @ ttmat_fn(torch.as_tensor([0., 0., 0.0164], dtype=torch.float32, device=device))
    rf_t_medial = rf_t_proximal @ rytmat_fn(rf_qpos[..., 1]) @ ttmat_fn(torch.as_tensor([0., 0., 0.054], dtype=torch.float32, device=device))
    rf_t_distal = rf_t_medial @ rytmat_fn(rf_qpos[..., 2]) @ ttmat_fn(torch.as_tensor([0., 0., 0.0384], dtype=torch.float32, device=device))
    rf_t_ftp = rf_t_distal @ rytmat_fn(rf_qpos[..., 3]) @ ttmat_fn(torch.as_tensor([0., 0., 0.0384], dtype=torch.float32, device=device))
    return rf_t_ftp[..., :3, 3]

def thtp_pos_fd_fn(th_qpos):
    th_t_base = t_palm @ ttmat_fn(torch.as_tensor([-0.0182, 0.019333, -0.045987], dtype=torch.float32, device=device)) @ quat2mat(torch.as_tensor([0.477714, -0.521334, -0.521334, -0.477714], dtype=torch.float32, device=device))
    th_t_proximal = th_t_base @ rxtmat_fn(-th_qpos[..., 0]) @ ttmat_fn(torch.as_tensor([-0.027, 0.005, 0.0399], dtype=torch.float32, device=device))
    th_t_medial = th_t_proximal @ rztmat_fn(th_qpos[..., 1]) @ ttmat_fn(torch.as_tensor([0., 0., 0.0177], dtype=torch.float32, device=device))
    th_t_distal = th_t_medial @ rytmat_fn(th_qpos[..., 2]) @ ttmat_fn(torch.as_tensor([0., 0., 0.0514], dtype=torch.float32, device=device))
    th_t_ftp = th_t_distal @ rytmat_fn(th_qpos[..., 3]) @ ttmat_fn(torch.as_tensor([0., 0., 0.054], dtype=torch.float32, device=device))
    return th_t_ftp[..., :3, 3]

if __name__ == "__main__":
    # Set symmetric joint angles for all fingers (except thumb)
    symmetric_qpos = torch.tensor([0.2, 0.2, 0.2, 0.2], requires_grad=True, device=device)
    
    # Apply the symmetric angles to the first, middle, and ring fingers
    ff_pos = fftp_pos_fd_fn(symmetric_qpos)
    mf_pos = mftp_pos_fd_fn(symmetric_qpos)
    rf_pos = rftp_pos_fd_fn(symmetric_qpos)
    
    # Set and apply separate angles for the thumb
    th_qpos = torch.tensor([0.2, 0.2, 0.2, 0.2], requires_grad=True, device=device)
    th_pos = thtp_pos_fd_fn(th_qpos)

    # Print the resulting positions
    print("First Finger Position:", ff_pos)
    print("Middle Finger Position:", mf_pos)
    print("Ring Finger Position:", rf_pos)
    print("Thumb Position:", th_pos)
