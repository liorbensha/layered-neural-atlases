import argparse
import os
import struct
import warnings
from subprocess import call

import cv2
import imageio
import numpy as np
import six
import torch
import torch.optim as optim
from PIL import Image


# TODO(Yakir): added it
def load_cameras_parameters():
    meta_file = "./data/ayush_depth/metadata_scaled.npz"
    with open(meta_file, "rb") as f:
        meta = np.load(f)
        extrinsics = np.array(meta["extrinsics"])
        intrinsics = np.array(meta["intrinsics"])
    assert (
        extrinsics.shape[0] == intrinsics.shape[0]
    ), f"#extrinsics({extrinsics.shape[0]}) != #intrinsics({intrinsics.shape[0]})"
    third_of_frames = []
    for i in range(len(intrinsics)):
        if i % 3 == 0:
            third_of_frames.append(intrinsics[i])

    return np.array(third_of_frames)


# (Lior&Yakir) We copied that from consistent depth code
def load_raw_float32_image(file_name):
    with open(file_name, "rb") as f:
        CV_CN_MAX = 512
        CV_CN_SHIFT = 3
        CV_32F = 5
        I_BYTES = 4
        Q_BYTES = 8

        h = struct.unpack("i", f.read(I_BYTES))[0]
        w = struct.unpack("i", f.read(I_BYTES))[0]

        cv_type = struct.unpack("i", f.read(I_BYTES))[0]
        pixel_size = struct.unpack("Q", f.read(Q_BYTES))[0]
        d = ((cv_type - CV_32F) >> CV_CN_SHIFT) + 1
        assert d >= 1
        d_from_pixel_size = pixel_size // 4
        if d != d_from_pixel_size:
            raise Exception("Incompatible pixel_size(%d) and cv_type(%d)" %
                            (pixel_size, cv_type))
        if d > CV_CN_MAX:
            raise Exception("Cannot save image with more than 512 channels")

        data = np.frombuffer(f.read(), dtype=np.float32)
        result = data.reshape(h, w) if d == 1 else data.reshape(h, w, d)
        return result


def compute_consistency(flow12, flow21):
    wflow21 = warp_flow(flow21, flow12)
    diff = flow12 + wflow21
    diff = (diff[:, :, 0]**2 + diff[:, :, 1]**2)**.5
    return diff


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = flow.copy()
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def get_consistency_mask(optical_flow, optical_flow_reverse):
    mask_flow = compute_consistency(optical_flow.numpy(),
                                    optical_flow_reverse.numpy()) < 1.0
    mask_flow_reverse = compute_consistency(optical_flow_reverse.numpy(),
                                            optical_flow.numpy()) < 1.0
    return torch.from_numpy(mask_flow), torch.from_numpy(mask_flow_reverse)


def resize_flow(flow, newh, neww):
    oldh, oldw = flow.shape[0:2]
    flow = cv2.resize(flow, (neww, newh), interpolation=cv2.INTER_LINEAR)
    flow[:, :, 0] *= newh / oldh
    flow[:, :, 1] *= neww / oldw
    return flow


def load_input_data(resy, resx, maximum_number_of_frames, data_folder,
                    use_mask_rcnn_bootstrapping, filter_optical_flow, vid_root,
                    vid_name):
    out_flow_dir = vid_root / f'{vid_name}_flow'
    maskrcnn_dir = vid_root / f'{vid_name}_maskrcnn'
    depth_dir = vid_root / f'{vid_name}_depth'

    depth_frames = torch.FloatTensor([
        cv2.resize(load_raw_float32_image(file_name), (resx, resy))
        for file_name in depth_dir.glob("*.raw")
    ])

    input_files = sorted(
        list(data_folder.glob('*.jpg')) + list(data_folder.glob('*.png')))

    number_of_frames = np.minimum(maximum_number_of_frames, len(input_files))
    video_frames = torch.zeros((resy, resx, 3, number_of_frames))
    video_frames_dx = torch.zeros((resy, resx, 3, number_of_frames))
    video_frames_dy = torch.zeros((resy, resx, 3, number_of_frames))

    mask_frames = torch.zeros((resy, resx, number_of_frames))

    optical_flows = torch.zeros((resy, resx, 2, number_of_frames, 1))
    optical_flows_mask = torch.zeros((resy, resx, number_of_frames, 1))
    optical_flows_reverse = torch.zeros((resy, resx, 2, number_of_frames, 1))
    optical_flows_reverse_mask = torch.zeros((resy, resx, number_of_frames, 1))

    mask_files = sorted(
        list(maskrcnn_dir.glob('*.jpg')) + list(maskrcnn_dir.glob('*.png')))
    for i in range(number_of_frames):
        file1 = input_files[i]
        im = np.array(Image.open(str(file1))).astype(np.float64) / 255.
        if use_mask_rcnn_bootstrapping:
            mask = np.array(Image.open(str(mask_files[i]))).astype(
                np.float64) / 255.
            mask = cv2.resize(mask, (resx, resy), cv2.INTER_NEAREST)
            mask_frames[:, :, i] = torch.from_numpy(mask)
        video_frames[:, :, :, i] = torch.from_numpy(
            cv2.resize(im[:, :, :3], (resx, resy)))
        video_frames_dy[:-1, :, :,
                        i] = video_frames[1:, :, :,
                                          i] - video_frames[:-1, :, :, i]
        video_frames_dx[:, :-1, :,
                        i] = video_frames[:, 1:, :,
                                          i] - video_frames[:, :-1, :, i]

    for i in range(number_of_frames - 1):
        file1 = input_files[i]
        j = i + 1
        file2 = input_files[j]

        fn1 = file1.name
        fn2 = file2.name

        flow12_fn = out_flow_dir / f'{fn1}_{fn2}.npy'
        flow21_fn = out_flow_dir / f'{fn2}_{fn1}.npy'
        flow12 = np.load(flow12_fn)
        flow21 = np.load(flow21_fn)

        if flow12.shape[0] != resy or flow12.shape[1] != resx:
            flow12 = resize_flow(flow12, newh=resy, neww=resx)
            flow21 = resize_flow(flow21, newh=resy, neww=resx)
        mask_flow = compute_consistency(flow12, flow21) < 1.0
        mask_flow_reverse = compute_consistency(flow21, flow12) < 1.0

        optical_flows[:, :, :, i, 0] = torch.from_numpy(flow12)
        optical_flows_reverse[:, :, :, j, 0] = torch.from_numpy(flow21)

        if filter_optical_flow:
            optical_flows_mask[:, :, i, 0] = torch.from_numpy(mask_flow)
            optical_flows_reverse_mask[:, :, j,
                                       0] = torch.from_numpy(mask_flow_reverse)
        else:
            optical_flows_mask[:, :, i, 0] = torch.ones_like(mask_flow)
            optical_flows_reverse_mask[:, :, j,
                                       0] = torch.ones_like(mask_flow_reverse)

    return optical_flows_mask, video_frames, optical_flows_reverse_mask, mask_frames, video_frames_dx, \
        video_frames_dy, optical_flows_reverse, optical_flows, depth_frames


def get_tuples(number_of_frames, video_frames, depth_frames):
    # video_frames shape: (resy, resx, 3, num_frames), mask_frames shape: (resy, resx, num_frames)
    jif_all = []
    depth_all = []
    for f in range(number_of_frames):
        mask = (video_frames[:, :, :, f] > -1).any(dim=2)
        relis, reljs = torch.where(mask > 0.5)
        depth = depth_frames[f][relis, reljs]
        #TODO try positional encoding?
        depth_all.append(depth)
        jif_all.append(torch.stack((reljs, relis, f * torch.ones_like(reljs))))
    return torch.cat(jif_all, dim=1), torch.cat(depth_all)


# See explanation in the paper, appendix A (Second paragraph)


def pre_train_mapping(model_F_mapping,
                      frames_num,
                      uvw_mapping_scale,
                      resx,
                      resy,
                      larger_dim,
                      device,
                      pretrain_iters=100):
    optimizer_mapping = optim.Adam(model_F_mapping.parameters(), lr=0.0001)
    for i in range(pretrain_iters):
        for f in range(frames_num):
            i_s_int = torch.randint(int(resy), (np.int64(10000), 1))
            j_s_int = torch.randint(int(resx), (np.int64(10000), 1))

            i_s = i_s_int / (larger_dim / 2) - 1
            j_s = j_s_int / (larger_dim / 2) - 1
            # TODO (Lior) I added depth at the 3rd coordinate, I changed the variable name from xyt
            xydt = torch.cat(
                (j_s, i_s, torch.ones_like(i_s),
                 (f / (frames_num / 2.0) - 1) * torch.ones_like(i_s)),
                dim=1).to(device)
            uv_temp = model_F_mapping(xydt)

            model_F_mapping.zero_grad()
            # TODO
            loss = (xydt[:, :3] * uvw_mapping_scale -
                    uv_temp).norm(dim=1).mean()
            print(f"pre-train loss: {loss.item()}")
            loss.backward()
            optimizer_mapping.step()
    return model_F_mapping


def save_mask_flow(optical_flows_mask, video_frames, results_folder):
    for j in range(optical_flows_mask.shape[3]):

        filter_flow_0 = imageio.get_writer("%s/filter_flow_%d.mp4" %
                                           (results_folder, j),
                                           fps=10)
        for i in range(video_frames.shape[3]):
            if torch.where(optical_flows_mask[:, :, i,
                                              j] == 1)[0].shape[0] == 0:
                continue
            cur_frame = video_frames[:, :, :, i].clone()
            # Put red color where mask=0.
            cur_frame[torch.where(optical_flows_mask[:, :, i, j] == 0)[0],
                      torch.where(optical_flows_mask[:, :, i, j] == 0)[1],
                      0] = 1
            cur_frame[torch.where(optical_flows_mask[:, :, i, j] == 0)[0],
                      torch.where(optical_flows_mask[:, :, i, j] == 0)[1],
                      1] = 0
            cur_frame[torch.where(optical_flows_mask[:, :, i, j] == 0)[0],
                      torch.where(optical_flows_mask[:, :, i, j] == 0)[1],
                      2] = 0

            filter_flow_0.append_data(
                (cur_frame.numpy() * 255).astype(np.uint8))

        filter_flow_0.close()
    # save the video in the working resolution
    input_video = imageio.get_writer("%s/input_video.mp4" % (results_folder),
                                     fps=10)
    for i in range(video_frames.shape[3]):
        cur_frame = video_frames[:, :, :, i].clone()

        input_video.append_data((cur_frame.numpy() * 255).astype(np.uint8))

    input_video.close()
