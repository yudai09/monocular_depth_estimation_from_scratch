from re import A
import torch


class GeneralCamera:
    @classmethod
    def camera2image(cls, points, intrinsic):
        raise NotImplementedError

    @classmethod
    def image2camera(cls, points, intrinsic):
        raise NotImplementedError

    @classmethod
    def world2camera(cls, world_points, extrinsic):
        raise NotImplementedError

    @classmethod
    def camera2world(cls, camera_points, extrinsic, depth):
        raise NotImplementedError

    @classmethod
    def world2image(cls, world_points, intrinsic, extrinsic):
        camera_points = cls.world2camera(world_points, extrinsic)
        image_points = cls.camera2image(camera_points, intrinsic)
        return image_points

    @classmethod
    def image2world(cls, image_points, intrinsic, extrinsic, depth):
        camera_points = cls.image2camera(image_points, intrinsic)
        world_points = cls.camera2world(camera_points, extrinsic, depth)
        return world_points


class PinholeCamera(GeneralCamera):
    @classmethod
    def camera2image(cls, camera_points, intrinsic):
        camera_points /= camera_points[..., [2]]  # to be homogeneous coords
        image_points = intrinsic @ camera_points.permute(0, 2, 1)
        image_points = image_points.permute(0, 2, 1)
        image_points = image_points[..., :2]
        return image_points

    @classmethod
    def image2camera(cls, image_points, intrinsic):
        # add z-axis
        image_points = torch.cat([image_points, torch.ones(image_points.shape[:2]).unsqueeze(-1)], axis=-1)
        intrinsic_inv = torch.linalg.inv(intrinsic)
        camera_points = intrinsic_inv @ image_points.permute(0, 2, 1)
        camera_points = camera_points.permute(0, 2, 1)
        return camera_points

    @classmethod
    def world2camera(cls, world_points, extrinsic):
        camera_points = extrinsic[:3, :] @ world_points.permute(0, 2, 1)
        camera_points = camera_points.permute(0, 2, 1)
        return camera_points

    @classmethod
    def camera2world(cls, camera_points, extrinsic, depth):
        camera_points *= depth.unsqueeze(-1)
        ones = torch.ones((camera_points.shape[0], camera_points.shape[1], 1))
        camera_points = torch.concat([camera_points, ones], axis=-1)
        extrinsic_inv = torch.linalg.inv(extrinsic)
        world_points = extrinsic_inv @ camera_points.permute(0, 2, 1)
        world_points = world_points.permute(0, 2, 1)
        return world_points
