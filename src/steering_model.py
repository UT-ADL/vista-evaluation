import os
import socket
import onnxruntime as ort

DEVICE_ID = int(os.environ.get('CUDA_AVAILABLE_DEVICES', '0').split(',')[0])
IS_UT_HPC = 'falcon' in socket.gethostname()


class OnnxSteeringModel:
    def __init__(self, path_to_onnx_model):
        options = ort.SessionOptions()
        if IS_UT_HPC:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            # these options are necessary only for HPC, not sure why
            # https://github.com/microsoft/onnxruntime/issues/8313#issuecomment-876092511
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
        else:
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': DEVICE_ID,
                }),
                'CPUExecutionProvider',
            ]
        
        self.session = ort.InferenceSession(path_to_onnx_model, options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, input_frame):
        return self.session.run(None, { self.input_name: input_frame })[0]
