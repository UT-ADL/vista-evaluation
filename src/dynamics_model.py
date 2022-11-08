import os
import socket

import onnxruntime as ort
import numpy as np

DEVICE_ID = int(os.environ.get('CUDA_AVAILABLE_DEVICES', '0').split(',')[0])
IS_NEURON = socket.gethostname() == 'neuron'
WARMUP_FRAMES = 20


class OnnxDynamicsModel:
    def __init__(self, path_to_onnx_model):
        options = ort.SessionOptions()
        if IS_NEURON:
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': DEVICE_ID,
                }),
                'CPUExecutionProvider',
            ]
        else:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            # these options are necessary only for HPC, not sure why:
            # https://github.com/microsoft/onnxruntime/issues/8313#issuecomment-876092511
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(path_to_onnx_model, options, providers=providers)
        self.last_effective_steering = None
        self.hidden_state_shape = self.session.get_inputs()[2].shape
        self.hidden_state = np.zeros(self.hidden_state_shape).astype(np.float32)
        self.steps_done = 0

    def predict(self, steering_command):
        if self.last_effective_steering is None:
            self.last_effective_steering = steering_command

        inputs = {
            self.session.get_inputs()[0].name: np.array(steering_command).astype(np.float32).reshape(1, 1),
            self.session.get_inputs()[1].name: np.array(self.last_effective_steering).astype(np.float32).reshape(1, 1),
            self.session.get_inputs()[2].name: self.hidden_state,
        }

        outs, self.hidden_state = self.session.run(None, inputs)
        self.last_effective_steering = outs.item()
        self.steps_done += 1

        if self.steps_done < WARMUP_FRAMES:
            return steering_command

        return self.last_effective_steering

    def reset(self):
        self.last_effective_steering = None
        self.hidden_state = np.zeros(self.hidden_state_shape).astype(np.float32)
        self.steps_done = 0
    

class NaiveSmoothingModel:
    def __init__(self, alpha=0.075):
        self.alpha = alpha
        self.last_effective_steering = None

    def predict(self, steering_command):
        if self.last_effective_steering is None:
            self.last_effective_steering = steering_command

        self.last_effective_steering = self.alpha * steering_command + (1 - self.alpha) * self.last_effective_steering
        return self.last_effective_steering

    def reset(self):
        self.last_effective_steering = None
