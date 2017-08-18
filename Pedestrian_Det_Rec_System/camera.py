

class CameraSrc:

    def __init__(self, resolution=None, output_format=None, flip=None):
        self._camera = 'nvcamerasrc'

        if isinstance(resolution, tuple) and resolution is not None:
            self._input_resolution = resolution
        else:
            self._input_resolution = (1920, 1080)

        if isinstance(output_format, str) and output_format is not None:
            self._output_format = output_format
        else:
            self._output_format = 'BGR'

        if isinstance(flip, int) and flip is not None:
            self._flip = str(flip)
        else:
            self._flip = 4

        # defaults
        self._frame_rate = 0
        self._input_format = 'I420'

        # src prefix
        self._prefix = [
            str(self._camera),
            ' ! video/x-raw(memory:NVMM)',
            ', width=(int)',
            str(self._input_resolution[0]),
            ', height=(int)',
            str(self._input_resolution[1]),
            ', format=(string)',
            self._input_format,
            ', framerate=(fraction)15/1 ! nvvidconv flip-method=',
            str(self._flip),
            ' ! video/x-raw,',
            ' format=(string)BGRx ! videoconvert ! video/x-raw',
            ', format=(string)',
            str(self._output_format),
            ' ! appsink'
        ]

    def get_cam_src(self, resolution=None, output_format=None, flip=None):
        """
        Get the camera prefix string
        :return:
        """
        if isinstance(resolution, tuple) and resolution is not None:
            self._prefix[3] = str(resolution[0]) # width
            self._prefix[5] = str(resolution[1]) # height

        if isinstance(output_format, str) and output_format is not None:
            self._prefix[13] = output_format # BGR

        if isinstance(flip, int) and flip is not None:
            self._prefix[9] = str(flip) # 4 is default/

        src = ''
        for txt in self._prefix:
            src += txt
        return src



