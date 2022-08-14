import torch

from unwrap_utils import get_tuples


def get_tuples_undistorted(number_of_frames, video_frames, depth_frames,
                           intrisics):
    # video_frames shape: (resy, resx, 3, num_frames), mask_frames shape: (resy, resx, num_frames)
    jif_all = []
    undistorted_jif_all = []
    depth_all = []
    for f in range(number_of_frames):
        mask = (video_frames[:, :, :, f] > -1).any(dim=2)
        fx, fy, cx, cy = intrisics[f]
        relis, reljs = torch.where(mask > 0.5)
        depth = depth_frames[f][relis, reljs]
        #TODO try positional encoding?
        # now apply intrisic to new data strutcure
        relis_undistorted = (fx * relis - cx) * depth # TODO(Yakir): make sure that cx and cy are not swapped
        reljs_undistorted = (fy * reljs - cy) * depth
        depth_all.append(depth)
        jif_all.append(torch.stack((reljs, relis, f * torch.ones_like(reljs))))
        undistorted_jif_all.append(torch.stack((reljs_undistorted, relis_undistorted, f * torch.ones_like(reljs))))
    return torch.cat(jif_all, dim=1), torch.cat(depth_all), torch.cat(undistorted_jif_all)


class CoordinatesWithDepth():
    def __init__(self, video_frames, depth_frames, intrisic_parameters):
        """
        """

        # Save input
        self.video_frames = video_frames
        self.depth_frames = depth_frames

        # Compute the needed data
        self.jif_all, self.depth_at_jif, self.undistorted_jif_all = get_tuples_undistorted(video_frames.shape[3],
                                                     video_frames,
                                                     depth_frames, intrisic_parameters)
        # jif_all.shape = (3, num_points)
        # depth_at_jif.shape = (num_points)
        # self.undistorted_dx, self.undistorted_dy = self.get_undistorted_gradient(
        #     intrisic_parameters)

    # def get_undistorted_gradient(self, intrisic_parameters):
    #     video_frames_dx = torch.zeros((resy, resx, 3, number_of_frames))
    #     video_frames_dy = torch.zeros((resy, resx, 3, number_of_frames))
    #     video_frames = self.video_frames
    #     # TODO: need undistorted frames here... Or not? ask Ronen and Meirav
    #     for i in range(number_of_frames):
    #         video_frames_dy[:-1, :, :,i] = video_frames[1:, :, :, i] - video_frames[:-1, :, :, i]
    #         video_frames_dx[:, :-1, :,i] = video_frames[:, 1:, :, i] - video_frames[:, :-1, :, i]

    #     video_frames_dx, video_frames_dy
