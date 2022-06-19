import torch


class GeneralCamera:
    @classmethod
    def camera2image(cls, points, intrinsic, batch=False):
        raise NotImplementedError

    @classmethod
    def image2camera(cls, points, intrinsic, batch=False):
        raise NotImplementedError

    @classmethod
    def world2camera(cls, world_points, extrinsic, batch=False):
        raise NotImplementedError

    @classmethod
    def camera2world(cls, camera_points, extrinsic, depth, batch=False):
        raise NotImplementedError

    @classmethod
    def world2image(cls, world_points, intrinsic, extrinsic, batch=False):
        camera_points = cls.world2camera(world_points, extrinsic, batch)
        image_points = cls.camera2image(camera_points, intrinsic, batch)
        return image_points

    @classmethod
    def image2world(cls, image_points, intrinsic, extrinsic, depth, batch=False):
        camera_points = cls.image2camera(image_points, intrinsic, batch)
        world_points = cls.camera2world(camera_points, extrinsic, depth, batch)
        return world_points

    @classmethod
    def matmul_mat_points(cls, mat, points, batch):
        dim_shift = 1 if batch else 0
        h, w, axis = 0 + dim_shift, 1 + dim_shift, 2 + dim_shift
        if batch:
            mat = mat.unsqueeze(1)
            points = points.permute(0, h, axis, w)
            points = mat @ points
            points = points.permute(0, h, axis, w)
        else:
            points = points.permute(h, axis, w)
            points = mat @ points
            points = points.permute(h, axis, w)
        return points

    @classmethod
    def add_axis(cls, points):
        points = torch.cat(
            [
                points,
                torch.ones(points.shape[:-1]).unsqueeze(-1).to(points.device)
            ],
            axis=-1)
        return points

    @classmethod
    def intrinsic_inv(cls, K):
        return torch.stack(
            [
                torch.stack([1 / K[...,  0, 0], K[..., 0, 1], - K[..., 0, 2] / K[..., 0, 0]], dim=-1),
                torch.stack([K[..., 1, 0], 1 / K[..., 1, 1], - K[..., 1, 2] / K[..., 1, 1]], dim=-1),
                torch.stack([K[..., 2, 0], K[..., 2, 1], K[..., 2, 2]], dim=-1)
            ],
            dim=-2)

    @classmethod
    def extrinsic_inv(cls, RT, batch=False):
        RT_inv = torch.eye(4, device=RT.device, dtype=RT.dtype)
        if batch:
            RT_inv = RT_inv.repeat([len(RT), 1, 1])
        RT_inv[..., :3, :3] = torch.transpose(RT[..., :3, :3], -2, -1)
        if batch:
            RT_inv[..., :3, -1] = torch.bmm(-1. * RT_inv[..., :3, :3], RT[..., :3, -1].unsqueeze(-1)).squeeze(-1)
        else:
            RT_inv[..., :3, -1] = (-1. * RT_inv[..., :3, :3]) @ (RT[..., :3, -1])
        return RT_inv


class PinholeCamera(GeneralCamera):
    @classmethod
    def camera2image(cls, camera_points, intrinsic, batch=False):
        camera_points /= camera_points[..., [2]]  # to be homogeneous coords
        image_points = cls.matmul_mat_points(intrinsic, camera_points, batch)
        image_points = image_points[..., :2]
        return image_points

    @classmethod
    def image2camera(cls, image_points, intrinsic, batch=False):
        image_points = cls.add_axis(image_points)
        intrinsic_inv = cls.intrinsic_inv(intrinsic)
        camera_points = cls.matmul_mat_points(intrinsic_inv, image_points, batch)
        return camera_points

    @classmethod
    def world2camera(cls, world_points, extrinsic, batch=False):
        camera_points = cls.matmul_mat_points(extrinsic[..., :3, :], world_points, batch)
        return camera_points

    @classmethod
    def camera2world(cls, camera_points, extrinsic, depth, batch=False):
        camera_points *= depth.unsqueeze(-1)
        camera_points = cls.add_axis(camera_points)
        extrinsic_inv = cls.extrinsic_inv(extrinsic, batch=batch)
        world_points = cls.matmul_mat_points(extrinsic_inv, camera_points, batch)
        return world_points
