"""
    Module Tests

"""
from camera import CameraSrc

print(CameraSrc().get_cam_src(resolution=(500, 370), output_format="HD", flip=12))