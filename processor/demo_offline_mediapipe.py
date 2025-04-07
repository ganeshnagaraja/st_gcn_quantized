#!/usr/bin/env python
import os
import sys
import argparse
import json
import shutil
import time

import numpy as np
# import skvideo.io

from .io import IO
import tools
import tools.utils as utils

import cv2

import mediapipe as mp
import torch
import torch.ao.quantization as quantization


class DemoOffline(IO):

    def start(self):
        
        # initiate
        label_name_path = './resource/kinetics_skeleton/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
            self.label_name = label_name

        # pose estimation
        video, data_numpy = self.pose_estimation()
        # self.pose_estimation()

        # action recognition
        data = torch.from_numpy(data_numpy)
        data = data.unsqueeze(0)
        data = data.float().to(self.dev).detach()  # (1, channel, frame, joint, person)

        self.model.eval()

        quantize = True
        if quantize == True:
            # quantize and save the model
            quantized_model = self.quantize(self.model)
            torch.save(quantized_model.state_dict(), './checkpoints/quantized_model.pt')
            exit()
        
        # model predict
        voting_label_name, video_label_name, output, intensity = self.predict(data)

        # render the video
        images = self.render_video(data_numpy, voting_label_name,
                            video_label_name, intensity, video)

        # visualize
        for image in images:
            image = image.astype(np.uint8)
            cv2.imshow("ST-GCN", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def quantize(self, model):

        # qunatization
        print('quatization....')
        torch.backends.quantized.engine = "qnnpack"

        # 4. Custom quantization configuration with explicit ranges
        self.model.qconfig = quantization.QConfig(
            activation=quantization.HistogramObserver.with_args(
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                reduce_range=False
            ),
            weight=quantization.PerChannelMinMaxObserver.with_args(
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                reduce_range=False
            )
        )

        prepared_model = torch.quantization.prepare(self.model)

        # Calibrate the model (pass representative data through it)
        # Here we use random data, but in practice you should use real validation data
        with torch.no_grad():
            for _ in range(10):
                input_tensor = torch.randn(1, 3, 300, 18, 1)  # Batch size of 1, input size 784
                prepared_model(input_tensor)

        quantized_model = torch.quantization.convert(prepared_model, inplace=False)

        # number of parameters in the model
        print("-----------------------------------------")
        num_params = sum(p.numel() for p in quantized_model.parameters())
        print('Number of parameters in the model: {}'.format(num_params))
        print('Model: {}'.format(self.arg.model))
        # number of layers in the model
        num_layers = sum(1 for _ in quantized_model.modules())
        print('Number of layers in the model: {}'.format(num_layers))
        print("-----------------------------------------")

        return quantized_model

    def predict(self, data):
        # forward
        output, feature = self.model.extract_feature(data)
        output = output[0]
        feature = feature[0]
        intensity = (feature*feature).sum(dim=0)**0.5
        intensity = intensity.cpu().detach().numpy()

        # get result
        # classification result of the full sequence
        voting_label = output.sum(dim=3).sum(
            dim=2).sum(dim=1).argmax(dim=0)
        voting_label_name = self.label_name[voting_label]
        # classification result for each person of the latest frame
        num_person = data.size(4)
        latest_frame_label = [output[:, :, :, m].sum(
            dim=2)[:, -1].argmax(dim=0) for m in range(num_person)]
        latest_frame_label_name = [self.label_name[l]
                                   for l in latest_frame_label]

        num_person = output.size(3)
        num_frame = output.size(1)
        video_label_name = list()
        for t in range(num_frame):
            frame_label_name = list()
            for m in range(num_person):
                person_label = output[:, t, :, m].sum(dim=1).argmax(dim=0)
                person_label_name = self.label_name[person_label]
                frame_label_name.append(person_label_name)
            video_label_name.append(frame_label_name)
        return voting_label_name, video_label_name, output, intensity

    def render_video(self, data_numpy, voting_label_name, video_label_name, intensity, video):
        images = utils.visualization.stgcn_visualize(
            data_numpy,
            self.model.graph.edge,
            intensity, video,
            voting_label_name,
            video_label_name,
            self.arg.height)
        return images

    def pose_estimation(self):

        video_name = self.arg.video.split('/')[-1].split('.')[0]

        # # initiate
        pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5)

        '''
        # number of parameters in the model
        print("-----------------------------------------")
        num_params = sum(p.numel() for p in self.model.parameters())
        print('Number of parameters in the model: {}'.format(num_params))
        print('Model: {}'.format(self.arg.model))
        # number of layers in the model
        num_layers = sum(1 for _ in self.model.modules())
        print('Number of layers in the model: {}'.format(num_layers))
        print(self.model)
        print("-----------------------------------------")
        '''
        
        video_capture = cv2.VideoCapture(self.arg.video)
        video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        pose_tracker = naive_pose_tracker(data_frame=video_length)

        # # pose estimation
        start_time = time.time()
        frame_index = 0
        video = list()
        
        while(True):
            # get image
            ret, orig_image = video_capture.read()
            if orig_image is None:
                break
            source_H, source_W, _ = orig_image.shape
            orig_image = cv2.resize(
                orig_image, (256 * source_W // source_H, 256))
            H, W, _ = orig_image.shape
            video.append(orig_image)

            # pose estimation
            results = pose.process(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
            if not results.pose_landmarks:
                continue
            
            landmarks = results.pose_landmarks.landmark
            mediapipe_pose_order = self.arg.mediapipe_pose_points

            pose_lst = []
            for each in self.arg.openpose_17_keypoints:

                if each == "PoseLandmark.NECK":
                    right_shoulder = landmarks[mediapipe_pose_order["PoseLandmark.RIGHT_SHOULDER"]]
                    left_shoulder = landmarks[mediapipe_pose_order["PoseLandmark.LEFT_SHOULDER"]]
                    neck_x = (right_shoulder.x + left_shoulder.x) / 2
                    neck_y = (right_shoulder.y + left_shoulder.y) / 2
                    neck_z = (right_shoulder.z + right_shoulder.z) / 2
                    each_row = [neck_x, neck_y, neck_z]
                    pose_lst.append(each_row)
                else:
                    co_ords = landmarks[mediapipe_pose_order[each]]
                    each_row = [co_ords.x, co_ords.y, co_ords.z]
                    pose_lst.append(each_row)
                
            # convert list to numpy array
            pose_np = np.array(pose_lst)
            pose_np = pose_np.reshape(1, 18, 3)

        #     datum = op.Datum()
        #     datum.cvInputData = orig_image
        #     opWrapper.emplaceAndPop([datum])
        #     multi_pose = datum.poseKeypoints  # (num_person, num_joint, 3)
            if len(pose_np.shape) != 3:
                continue
            
        #     # normalization
        #     multi_pose[:, :, 0] = multi_pose[:, :, 0]/W
        #     multi_pose[:, :, 1] = multi_pose[:, :, 1]/H
            pose_np[:, :, 0:2] = pose_np[:, :, 0:2] - 0.5
            pose_np[:, :, 0][pose_np[:, :, 2] == 0] = 0
            pose_np[:, :, 1][pose_np[:, :, 2] == 0] = 0

        #     # pose tracking
            pose_tracker.update(pose_np, frame_index)
            frame_index += 1

            print('Pose estimation ({}/{}).'.format(frame_index, video_length))

        data_numpy = pose_tracker.get_skeleton_sequence()
        
        return video, data_numpy
        # exit()

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--video',
                            default='./resource/media/ta_chi.mp4',
                            help='Path to video')
        parser.add_argument('--openpose',
                            default=None,
                            help='Path to openpose')
        parser.add_argument('--model_input_frame',
                            default=128,
                            type=int)
        parser.add_argument('--model_fps',
                            default=30,
                            type=int)
        parser.add_argument('--height',
                            default=1080,
                            type=int,
                            help='height of frame in the output video.')
        parser.set_defaults(
            config='./config/st_gcn/kinetics-skeleton/demo_offline.yaml')
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser

class naive_pose_tracker():
    """ A simple tracker for recording person poses and generating skeleton sequences.
    For actual occasion, I recommend you to implement a robuster tracker.
    Pull-requests are welcomed.
    """

    def __init__(self, data_frame=128, num_joint=18, max_frame_dis=np.inf):
        self.data_frame = data_frame
        self.num_joint = num_joint
        self.max_frame_dis = max_frame_dis
        self.latest_frame = 0
        self.trace_info = list()

    def update(self, multi_pose, current_frame):
        # multi_pose.shape: (num_person, num_joint, 3)

        if current_frame <= self.latest_frame:
            return

        if len(multi_pose.shape) != 3:
            return

        score_order = (-multi_pose[:, :, 2].sum(axis=1)).argsort(axis=0)
        for p in multi_pose[score_order]:

            # match existing traces
            matching_trace = None
            matching_dis = None
            for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
                # trace.shape: (num_frame, num_joint, 3)
                if current_frame <= latest_frame:
                    continue
                mean_dis, is_close = self.get_dis(trace, p)
                if is_close:
                    if matching_trace is None:
                        matching_trace = trace_index
                        matching_dis = mean_dis
                    elif matching_dis > mean_dis:
                        matching_trace = trace_index
                        matching_dis = mean_dis

            # update trace information
            if matching_trace is not None:
                trace, latest_frame = self.trace_info[matching_trace]

                # padding zero if the trace is fractured
                pad_mode = 'interp' if latest_frame == self.latest_frame else 'zero'
                pad = current_frame-latest_frame-1
                new_trace = self.cat_pose(trace, p, pad, pad_mode)
                self.trace_info[matching_trace] = (new_trace, current_frame)

            else:
                new_trace = np.array([p])
                self.trace_info.append((new_trace, current_frame))

        self.latest_frame = current_frame

    def get_skeleton_sequence(self):

        # remove old traces
        valid_trace_index = []
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            if self.latest_frame - latest_frame < self.data_frame:
                valid_trace_index.append(trace_index)
        self.trace_info = [self.trace_info[v] for v in valid_trace_index]

        num_trace = len(self.trace_info)
        if num_trace == 0:
            return None

        data = np.zeros((3, self.data_frame, self.num_joint, num_trace))
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            end = self.data_frame - (self.latest_frame - latest_frame)
            d = trace[-end:]
            beg = end - len(d)
            data[:, beg:end, :, trace_index] = d.transpose((2, 0, 1))

        return data

    # concatenate pose to a trace
    def cat_pose(self, trace, pose, pad, pad_mode):
        # trace.shape: (num_frame, num_joint, 3)
        num_joint = pose.shape[0]
        num_channel = pose.shape[1]
        if pad != 0:
            if pad_mode == 'zero':
                trace = np.concatenate(
                    (trace, np.zeros((pad, num_joint, 3))), 0)
            elif pad_mode == 'interp':
                last_pose = trace[-1]
                coeff = [(p+1)/(pad+1) for p in range(pad)]
                interp_pose = [(1-c)*last_pose + c*pose for c in coeff]
                trace = np.concatenate((trace, interp_pose), 0)
        new_trace = np.concatenate((trace, [pose]), 0)
        return new_trace

    # calculate the distance between a existing trace and the input pose

    def get_dis(self, trace, pose):
        last_pose_xy = trace[-1, :, 0:2]
        curr_pose_xy = pose[:, 0:2]

        mean_dis = ((((last_pose_xy - curr_pose_xy)**2).sum(1))**0.5).mean()
        wh = last_pose_xy.max(0) - last_pose_xy.min(0)
        scale = (wh[0] * wh[1]) ** 0.5 + 0.0001
        is_close = mean_dis < scale * self.max_frame_dis
        return mean_dis, is_close
