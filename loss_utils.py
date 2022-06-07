import torch

dtype = torch.FloatTensor
dtype_long = torch.LongTensor


def bilinear_interpolate_torch(depth_frames, frames, x, y):
    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1  # type: ignore

    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1  # type: ignore

    x0 = torch.clamp(x0, 0, depth_frames.shape[1] - 1)
    x1 = torch.clamp(x1, 0, depth_frames.shape[1] - 1)
    y0 = torch.clamp(y0, 0, depth_frames.shape[0] - 1)
    y1 = torch.clamp(y1, 0, depth_frames.shape[0] - 1)

    Ia = depth_frames[y0, x0, frames]
    Ib = depth_frames[y1, x0, frames]
    Ic = depth_frames[y0, x1, frames]
    Id = depth_frames[y1, x1, frames]

    wa = (x1.type(dtype) - x) * (y1.type(dtype) - y)
    wb = (x1.type(dtype) - x) * (y - y0.type(dtype))
    wc = (x - x0.type(dtype)) * (y1.type(dtype) - y)
    wd = (x - x0.type(dtype)) * (y - y0.type(dtype))

    return (Ia * wa) + (Ib * wb) + (Ic * wc) + (Id * wd)


# calculating the gradient loss as defined by Eq.7 in the paper
def get_gradient_loss(video_frames_dx, video_frames_dy, jif_current,
                      depth_at_jif_current, model_F_mapping1, model_F_mapping2,
                      model_F_atlas, rgb_output_foreground, device, resx,
                      number_of_frames, model_alpha):
    # TODO (Lior) We left the 2D rigid loss and added 3rd coordinate
    xplus1ydt_foreground = torch.cat(
        ((jif_current[0, :] + 1) / (resx / 2) - 1, jif_current[1, :] /
         (resx / 2) - 1, depth_at_jif_current, jif_current[2, :] /
         (number_of_frames / 2.0) - 1),
        dim=1).to(device)
    xyplus1dt_foreground = torch.cat(
        ((jif_current[0, :]) / (resx / 2) - 1, (jif_current[1, :] + 1) /
         (resx / 2) - 1, depth_at_jif_current, jif_current[2, :] /
         (number_of_frames / 2.0) - 1),
        dim=1).to(device)

    alphaxplus1 = 0.5 * (model_alpha(xplus1ydt_foreground) + 1.0)
    alphaxplus1 = alphaxplus1 * 0.99
    alphaxplus1 = alphaxplus1 + 0.001

    alphayplus1 = 0.5 * (model_alpha(xyplus1dt_foreground) + 1.0)
    alphayplus1 = alphayplus1 * 0.99
    alphayplus1 = alphayplus1 + 0.001

    # precomputed discrete derivative with respect to x,y direction
    rgb_dx_gt = video_frames_dx[jif_current[1, :], jif_current[0, :], :,
                                jif_current[2, :]].squeeze(1).to(device)
    rgb_dy_gt = video_frames_dy[jif_current[1, :], jif_current[0, :], :,
                                jif_current[2, :]].squeeze(1).to(device)

    # uvw coordinates for locations with offsets of 1 pixel
    uvw_foreground2_xyplus1t = model_F_mapping2(xyplus1dt_foreground)
    uvw_foreground2_xplus1yt = model_F_mapping2(xplus1ydt_foreground)
    uvw_foreground1_xyplus1t = model_F_mapping1(xyplus1dt_foreground)
    uvw_foreground1_xplus1yt = model_F_mapping1(xplus1ydt_foreground)

    # The RGB values (from the 2 layers) for locations with offsets of 1 pixel
    rgb_output1_xyplus1dt = (
        model_F_atlas(uvw_foreground1_xyplus1t * 0.5 + 0.5) + 1.0) * 0.5
    rgb_output1_xplus1ydt = (
        model_F_atlas(uvw_foreground1_xplus1yt * 0.5 + 0.5) + 1.0) * 0.5
    rgb_output2_xyplus1dt = (
        model_F_atlas(uvw_foreground2_xyplus1t * 0.5 - 0.5) + 1.0) * 0.5
    rgb_output2_xplus1ydt = (
        model_F_atlas(uvw_foreground2_xplus1yt * 0.5 - 0.5) + 1.0) * 0.5

    # Reconstructed RGB values:
    rgb_output_foreground_xyplus1dt = rgb_output1_xyplus1dt * alphayplus1 + rgb_output2_xyplus1dt * (
        1.0 - alphayplus1)
    rgb_output_foreground_xplus1ydt = rgb_output1_xplus1ydt * alphaxplus1 + rgb_output2_xplus1ydt * (
        1.0 - alphaxplus1)

    # Use reconstructed RGB values for computing derivatives:
    rgb_dx_output = rgb_output_foreground_xplus1ydt - rgb_output_foreground
    rgb_dy_output = rgb_output_foreground_xyplus1dt - rgb_output_foreground
    gradient_loss = torch.mean((rgb_dx_gt - rgb_dx_output).norm(dim=1)**2 +
                               (rgb_dy_gt - rgb_dy_output).norm(dim=1)**2)
    return gradient_loss


# get rigidity loss as defined in Eq. 9 in the paper


def get_rigidity_loss(jif_foreground,
                      depth_at_jif_current,
                      derivative_amount,
                      resx,
                      number_of_frames,
                      model_F_mapping,
                      uvw_foreground,
                      device,
                      uvw_mapping_scale=1.0,
                      return_all=False):
    # concatenating (x,y-derivative_amount,t) and (x-derivative_amount,y,t) to get xyt_p:
    # TODO (Yakir): We changed the loss so it runs - need to figure the loss we want for the best results
    is_patch = torch.cat((jif_foreground[1, :] - derivative_amount,
                          jif_foreground[1, :])) / (resx / 2) - 1
    js_patch = torch.cat(
        (jif_foreground[0, :],
         jif_foreground[0, :] - derivative_amount)) / (resx / 2) - 1
    ds_patch = torch.cat(
        (depth_at_jif_current, depth_at_jif_current - derivative_amount))
    fs_patch = torch.cat((jif_foreground[2, :],
                          jif_foreground[2, :])) / (number_of_frames / 2.0) - 1
    xydt_p = torch.cat((js_patch, is_patch, ds_patch, fs_patch),
                       dim=1).to(device)

    uvw_p = model_F_mapping(xydt_p)
    # u_p[0,:]= u(x,y-derivative_amount,t).  u_p[1,:]= u(x-derivative_amount,y,t)
    u_p = uvw_p[:, 0].view(2, -1)
    # v_p[0,:]= u(x,y-derivative_amount,t).  v_p[1,:]= v(x-derivative_amount,y,t)
    v_p = uvw_p[:, 1].view(2, -1)
    w_p = uvw_p[:, 2].view(2, -1)

    # u_p_d_[0,:]=u(x,y,t)-u(x,y-derivative_amount,t)   u_p_d_[1,:]= u(x,y,t)-u(x-derivative_amount,y,t).
    u_p_d_ = uvw_foreground[:, 0].unsqueeze(0) - u_p
    # v_p_d_[0,:]=u(x,y,t)-v(x,y-derivative_amount,t).  v_p_d_[1,:]= u(x,y,t)-v(x-derivative_amount,y,t).
    v_p_d_ = uvw_foreground[:, 1].unsqueeze(0) - v_p
    w_p_d_ = uvw_foreground[:, 2].unsqueeze(0) - w_p

    # to match units: 1 in uvw coordinates is resx/2 in image space.
    du_dx = u_p_d_[1, :] * resx / 2
    du_dy = u_p_d_[0, :] * resx / 2
    dv_dx = v_p_d_[1, :] * resx / 2
    dv_dy = v_p_d_[0, :] * resx / 2
    dw_dx = w_p_d_[1, :] * resx / 2
    dw_dy = w_p_d_[0, :] * resx / 2

    jacobians = torch.cat((torch.cat(
        (du_dx.unsqueeze(-1).unsqueeze(-1), du_dy.unsqueeze(-1).unsqueeze(-1)),
        dim=2),
                           torch.cat((dv_dx.unsqueeze(-1).unsqueeze(-1),
                                      dv_dy.unsqueeze(-1).unsqueeze(-1)),
                                     dim=2),
                           torch.cat((dw_dx.unsqueeze(-1).unsqueeze(-1),
                                      dw_dy.unsqueeze(-1).unsqueeze(-1)),
                                     dim=2)),
                          dim=1)
    jacobians = jacobians / uvw_mapping_scale
    jacobians = jacobians / derivative_amount

    # Apply a loss to constrain the Jacobian to be a rotation matrix as much as possible
    JtJ = torch.matmul(jacobians.transpose(1, 2), jacobians)

    # a = JtJ[:, 0, 0] + 0.001
    # b = JtJ[:, 0, 1]
    # c = JtJ[:, 1, 0]
    # d = JtJ[:, 1, 1] + 0.001

    # JTJinv = torch.zeros_like(jacobians).to(device)
    # JTJinv[:, 0, 0] = d
    # JTJinv[:, 0, 1] = -b
    # JTJinv[:, 1, 0] = -c
    # JTJinv[:, 1, 1] = a
    JTJinv = JtJ.inverse()

    # See Equation (9) in the paper:
    rigidity_loss = (JtJ ** 2).sum(1).sum(1).sqrt() + \
        (JTJinv ** 2).sum(1).sum(1).sqrt()

    if return_all:
        return rigidity_loss
    else:
        return rigidity_loss.mean()


# Compute optical flow loss (Eq. 11 in the paper) for all pixels without averaging. This is relevant for visualization of the loss.
def get_optical_flow_loss_all(jif_foreground,
                              depth_at_jif_current,
                              uvw_foreground,
                              resx,
                              number_of_frames,
                              model_F_mapping,
                              optical_flows,
                              optical_flows_mask,
                              uvw_mapping_scale,
                              device,
                              alpha=1.0):
    # (Lior) without d and its OK! :)
    xydt_foreground_forward_should_match, relevant_batch_indices_forward = get_corresponding_flow_matches_all(
        jif_foreground, depth_at_jif_current, optical_flows_mask,
        optical_flows, resx, number_of_frames)  # type: ignore
    uvw_foreground_forward_should_match = model_F_mapping(
        xydt_foreground_forward_should_match.to(device))

    errors = (uvw_foreground_forward_should_match - uvw_foreground).norm(dim=1)
    errors[relevant_batch_indices_forward == False] = 0
    errors = errors * (alpha.squeeze())

    return errors * resx / (2 * uvw_mapping_scale)


# Compute optical flow loss (Eq. 11 in the paper)
def get_optical_flow_loss(jif_foreground,
                          depth_frames,
                          uvw_foreground,
                          optical_flows_reverse,
                          optical_flows_reverse_mask,
                          resx,
                          number_of_frames,
                          model_F_mapping,
                          optical_flows,
                          optical_flows_mask,
                          uvw_mapping_scale,
                          device,
                          use_alpha=False,
                          alpha=1.0):
    # Forward flow:
    uvw_foreground_forward_relevant, xydt_foreground_forward_should_match, relevant_batch_indices_forward = \
        get_corresponding_flow_matches(
        jif_foreground, depth_frames, optical_flows_mask,
        optical_flows, resx, number_of_frames, True, uvw_foreground)
    uvw_foreground_forward_should_match = model_F_mapping(
        xydt_foreground_forward_should_match.to(device))
    loss_flow_next = (uvw_foreground_forward_should_match -
                      uvw_foreground_forward_relevant).norm(
                          dim=1) * resx / (2 * uvw_mapping_scale)

    # Backward flow:
    uvw_foreground_backward_relevant, xydt_foreground_backward_should_match, relevant_batch_indices_backward = \
    get_corresponding_flow_matches(
        jif_foreground, depth_frames, optical_flows_reverse_mask, optical_flows_reverse,
        resx, number_of_frames, False, uvw_foreground)
    uvw_foreground_backward_should_match = model_F_mapping(
        xydt_foreground_backward_should_match.to(device))
    loss_flow_prev = (uvw_foreground_backward_should_match -
                      uvw_foreground_backward_relevant).norm(
                          dim=1) * resx / (2 * uvw_mapping_scale)

    if use_alpha:
        flow_loss = (
            loss_flow_prev *
            alpha[relevant_batch_indices_backward].squeeze()).mean() * 0.5 + (
                loss_flow_next *
                alpha[relevant_batch_indices_forward].squeeze()).mean() * 0.5
    else:
        flow_loss = (loss_flow_prev).mean() * 0.5 + \
            (loss_flow_next).mean() * 0.5

    return flow_loss


# A helper function for get_optical_flow_loss to return matching points according to the optical flow
def get_corresponding_flow_matches(jif_foreground,
                                   depth_frames,
                                   optical_flows_mask,
                                   optical_flows,
                                   resx,
                                   number_of_frames,
                                   is_forward,
                                   uvw_foreground,
                                   use_uvw=True):
    batch_forward_mask = torch.where(
        optical_flows_mask[jif_foreground[1, :].squeeze(),
                           jif_foreground[0, :].squeeze(),
                           jif_foreground[2, :].squeeze(), :])
    forward_frames_amount = 2**batch_forward_mask[1]
    relevant_batch_indices = batch_forward_mask[0]
    jif_foreground_forward_relevant = jif_foreground[:, relevant_batch_indices,
                                                     0]
    forward_flows_for_loss = optical_flows[
        jif_foreground_forward_relevant[1],
        jif_foreground_forward_relevant[0], :,
        jif_foreground_forward_relevant[2], batch_forward_mask[1]]

    if is_forward:
        jif_foreground_forward_should_match = torch.stack(
            (jif_foreground_forward_relevant[0] + forward_flows_for_loss[:, 0],
             jif_foreground_forward_relevant[1] + forward_flows_for_loss[:, 1],
             jif_foreground_forward_relevant[2] + forward_frames_amount))

    else:
        jif_foreground_forward_should_match = torch.stack(
            (jif_foreground_forward_relevant[0] + forward_flows_for_loss[:, 0],
             jif_foreground_forward_relevant[1] + forward_flows_for_loss[:, 1],
             jif_foreground_forward_relevant[2] - forward_frames_amount))

    depth_frames = depth_frames.permute(1, 2, 0)
    forward_frames = jif_foreground_forward_should_match[2].long()
    x_y_of_frames = jif_foreground_forward_should_match[:-1]
    depth_foreground_forward_should_match = bilinear_interpolate_torch(
        depth_frames, forward_frames, x_y_of_frames[0], x_y_of_frames[1])

    xydt_foreground_forward_should_match = torch.stack((
        jif_foreground_forward_should_match[0] / (resx / 2) - 1,
        jif_foreground_forward_should_match[1] / (resx / 2) - 1,
        # TODO: (Yakir) make sure this is the correct depth ([Future Yakir] its not)
        depth_foreground_forward_should_match,
        jif_foreground_forward_should_match[2] / (number_of_frames / 2) - 1)).T
    if use_uvw:
        uvw_foreground_forward_relevant = uvw_foreground[batch_forward_mask[0]]
        return uvw_foreground_forward_relevant, xydt_foreground_forward_should_match, relevant_batch_indices
    else:
        return xydt_foreground_forward_should_match, relevant_batch_indices


# A helper function for get_optical_flow_loss_all to return matching points according to the optical flow
def get_corresponding_flow_matches_all(jif_foreground,
                                       depth_at_jif,
                                       optical_flows_mask,
                                       optical_flows,
                                       resx,
                                       number_of_frames,
                                       use_uvw=True):
    jif_foreground_forward_relevant = jif_foreground

    forward_flows_for_loss = optical_flows[
        jif_foreground_forward_relevant[1],
        jif_foreground_forward_relevant[0], :,
        jif_foreground_forward_relevant[2], 0].squeeze()
    forward_flows_for_loss_mask = optical_flows_mask[
        jif_foreground_forward_relevant[1], jif_foreground_forward_relevant[0],
        jif_foreground_forward_relevant[2], 0].squeeze()

    jif_foreground_forward_should_match = torch.stack(
        (jif_foreground_forward_relevant[0].squeeze() +
         forward_flows_for_loss[:, 0],
         jif_foreground_forward_relevant[1].squeeze() +
         forward_flows_for_loss[:, 1],
         jif_foreground_forward_relevant[2].squeeze() + 1))

    xydt_foreground_forward_should_match = torch.stack((
        jif_foreground_forward_should_match[0] / (resx / 2) - 1,
        jif_foreground_forward_should_match[1] / (resx / 2) - 1,
        depth_at_jif,  #TODO depth at jif isn't the correct frame
        jif_foreground_forward_should_match[2] / (number_of_frames / 2) - 1)).T
    if use_uvw:
        return xydt_foreground_forward_should_match, forward_flows_for_loss_mask > 0
    else:
        return 0


# Compute alpha optical flow loss (Eq. 12 in the paper)


def get_optical_flow_alpha_loss(model_alpha, jif_foreground, depth_frames,
                                alpha, optical_flows_reverse,
                                optical_flows_reverse_mask, resx,
                                number_of_frames, optical_flows,
                                optical_flows_mask, device):
    # Forward flow
    xyt_foreground_forward_should_match, relevant_batch_indices_forward = get_corresponding_flow_matches(
        jif_foreground,
        depth_frames,
        optical_flows_mask,
        optical_flows,
        resx,
        number_of_frames,
        True,
        0,
        use_uvw=False)
    alpha_foreground_forward_should_match = model_alpha(
        xyt_foreground_forward_should_match.to(device))
    alpha_foreground_forward_should_match = 0.5 * \
        (alpha_foreground_forward_should_match + 1.0)
    alpha_foreground_forward_should_match = alpha_foreground_forward_should_match * 0.99
    alpha_foreground_forward_should_match = alpha_foreground_forward_should_match + 0.001
    loss_flow_alpha_next = (
        alpha[relevant_batch_indices_forward] -
        alpha_foreground_forward_should_match).abs().mean()

    # Backward loss
    xyt_foreground_backward_should_match, relevant_batch_indices_backward = get_corresponding_flow_matches(
        jif_foreground,
        depth_frames,
        optical_flows_reverse_mask,
        optical_flows_reverse,
        resx,
        number_of_frames,
        False,
        0,
        use_uvw=False)
    alpha_foreground_backward_should_match = model_alpha(
        xyt_foreground_backward_should_match.to(device))
    alpha_foreground_backward_should_match = 0.5 * \
        (alpha_foreground_backward_should_match + 1.0)
    alpha_foreground_backward_should_match = alpha_foreground_backward_should_match * 0.99
    alpha_foreground_backward_should_match = alpha_foreground_backward_should_match + 0.001
    loss_flow_alpha_prev = (
        alpha_foreground_backward_should_match -
        alpha[relevant_batch_indices_backward]).abs().mean()

    return (loss_flow_alpha_next + loss_flow_alpha_prev) * 0.5


# Compute alpha optical flow loss (Eq. 12 in the paper) for all the pixels for visualization.
def get_optical_flow_alpha_loss_all(model_alpha, jif_foreground, depth_at_jif,
                                    alpha, resx, number_of_frames,
                                    optical_flows, optical_flows_mask, device):
    xydt_foreground_forward_should_match, relevant_batch_indices_forward = get_corresponding_flow_matches_all(
        jif_foreground, depth_at_jif, optical_flows_mask, optical_flows, resx,
        number_of_frames)  # type: ignore
    alpha_foreground_forward_should_match = model_alpha(
        xydt_foreground_forward_should_match.to(device))
    alpha_foreground_forward_should_match = 0.5 * \
        (alpha_foreground_forward_should_match + 1.0)
    alpha_foreground_forward_should_match = alpha_foreground_forward_should_match * 0.99
    alpha_foreground_forward_should_match = alpha_foreground_forward_should_match + 0.001

    loss_flow_alpha_next = (alpha -
                            alpha_foreground_forward_should_match).abs()
    loss_flow_alpha_next[relevant_batch_indices_forward == False] = 0

    return loss_flow_alpha_next
