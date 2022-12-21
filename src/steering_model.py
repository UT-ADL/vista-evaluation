import os
import socket

import onnxruntime as ort

from src.trajectory import calculate_steering_angle

DEVICE_ID = int(os.environ.get('CUDA_AVAILABLE_DEVICES', '0').split(',')[0])
IS_UT_HPC = 'falcon' in socket.gethostname()


class OnnxModel:
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


class SteeringModel:

    def __init__(self, path_to_onnx_model):
        self.steering_model = OnnxModel(path_to_onnx_model)

    def predict(self, input_frame, car):
        return self.steering_model.predict(input_frame).item(), None


class ConditionalSteeringModel(SteeringModel):

    def __init__(self, path_to_steering_model, path_to_speed_model):
        super().__init__(path_to_steering_model)
        self.speed_model = OnnxModel(path_to_speed_model)

    def predict(self, input_frame, car):
        predictions = self.steering_model.predict(input_frame)
        steering_angle = predictions[0][car.human_turn_signal].item()
        speed = self.speed_model.predict(input_frame)[0].item()
        return steering_angle, speed


class ConditionalWaypointsModel(SteeringModel):

    def __init__(self, path_to_steering_model, path_to_speed_model, num_waypoints=4, ref_distance=6.8,
                 use_vehicle_pos=True, latitudinal_correction=-0.30):
        super().__init__(path_to_steering_model)
        self.speed_model = OnnxModel(path_to_speed_model)
        self.num_waypoints = num_waypoints
        self.ref_distance = ref_distance
        self.use_vehicle_pos = use_vehicle_pos
        self.latitudinal_correction = latitudinal_correction

    def predict(self, input_frame, car):
        predictions = self.steering_model.predict(input_frame)
        predictions = predictions[0].reshape(3, -1)
        waypoints = predictions[car.human_turn_signal]
        steering_angle = calculate_steering_angle(waypoints, self.num_waypoints, self.ref_distance,
                                                  self.use_vehicle_pos, self.latitudinal_correction)
        speed = self.speed_model.predict(input_frame)[0].item()
        return steering_angle, speed
