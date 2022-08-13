import torch

from unwrap_utils import get_tuples


def get_tuples_undistorted(number_of_frames, video_frames, depth_frames,
                           intrisics):
    # video_frames shape: (resy, resx, 3, num_frames), mask_frames shape: (resy, resx, num_frames)
    jif_all = []
    depth_all = []
    for f in range(number_of_frames):
        mask = (video_frames[:, :, :, f] > -1).any(dim=2)
        current_intrisics = intrisics[f]
        relis, reljs = torch.where(mask > 0.5)
        depth = depth_frames[f][relis, reljs]
        #TODO try positional encoding?
        # now apply intrisic to new data strutcure
        depth_all.append(depth)
        jif_all.append(torch.stack((reljs, relis, f * torch.ones_like(reljs))))
    return torch.cat(jif_all, dim=1), torch.cat(depth_all)


def principal_point(intrinsics):
    """
    Args:
        intrinsics: (fx, fy, cx, cy)
        shape: (H, W)
    """
    return intrinsics[:, 2:]


def focal_length(intrinsics):
    return intrinsics[:, :2]


def pixels_to_rays(pixels, intrinsics):
    """Convert pixels to rays in camera space using intrinsics.

    Args:
        pixels (2, B)
        intrinsics (4, B): (fx, fy, cx, cy)

    Returns:
        rays: (3, B), where z component is -1, i.e., rays[:, -1] = -1

    """
    # Assume principal point is ((W-1)/2, (H-1)/2).
    _, N = pixels.shape
    cs = principal_point(
        intrinsics)  # hopefully will be able to have it for each point
    # Convert to [-(W-1)/2, (W-1)/2] x [-(H-1)/2, (H-1)/2)] and bottom left is (0, 0)
    uvs = pixels - cs.view(-1, 2, 1, 1)
    uvs[:, 1] = -uvs[:, 1]  # flip v

    # compute rays (u/fx, v/fy, -1)
    fxys = focal_length(intrinsics).view(-1, 2, 1, 1)
    rays = torch.cat((uvs / fxys, -torch.ones(
        (B, 1, H, W), dtype=uvs.dtype, device=_device)),
                     dim=1)
    return rays


def pixels_to_points(intrinsics, depths, pixels):
    """Convert pixels to 3D points in camera space. (Camera facing -z direction)

    Args:
        intrinsics:
        depths (B, 1)
        pixels (B, 2)

    Returns:
        points (B, 3)

    """
    rays = pixels_to_rays(pixels, intrinsics)
    points = rays * depths
    return points


class CoordinatesWithDepth():
    def __init__(self, video_frames, depth_frames, intrisic_parameters):
        """
        expecteing `intrisic_parameters` of shape (B, 4) (with repetitions of course)
        """

        # Save input
        self.video_frames = video_frames
        self.depth_frames = depth_frames

        # Compute the needed data
        self.jif_all, self.depth_at_jif = get_tuples(video_frames.shape[3],
                                                     video_frames,
                                                     depth_frames)
        # jif_all.shape = (3, num_points)
        # depth_at_jif.shape = (num_points) - figure out how to apply K^-1 efficently to not get memory out

        self.undistorted_jif_all, self.undistorted_depth_at_jif = self.undistort_jif_and_depth(
            intrisic_parameters)
        self.undistorted_dx, self.undistorted_dy = self.get_undistorted_gradient(
            intrisic_parameters)

    def undistort_jif_and_depth(self, intrisic_parameters):
        return None, None

    def get_undistorted_gradient(self, intrisic_parameters):
        return None, None
