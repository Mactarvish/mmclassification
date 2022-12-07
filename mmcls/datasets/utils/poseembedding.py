import numpy as np
import time


# 手部姿态编码模块
class FullBodyPoseEmbedder(object):
    """Converts 3D pose landmarks into 3D embedding."""

    def __init__(self, torso_size_multiplier=2.5):
        # Multiplier to apply to the torso to get minimal body size.
        # 乘数应用于躯干以获得最小的身体尺寸
        self._torso_size_multiplier = torso_size_multiplier

        # Names of the landmarks as they appear in the prediction.
        # 出现在预测中的landmarks名称。
        # self._landmark_names = [
        #     'nose',
        #     'left_eye_inner', 'left_eye', 'left_eye_outer',
        #     'right_eye_inner', 'right_eye', 'right_eye_outer',
        #     'left_ear', 'right_ear',
        #     'mouth_left', 'mouth_right',
        #     'left_shoulder', 'right_shoulder',
        #     'left_elbow', 'right_elbow',
        #     'left_wrist', 'right_wrist',
        #     'left_pinky_1', 'right_pinky_1',
        #     'left_index_1', 'right_index_1',
        #     'left_thumb_2', 'right_thumb_2',
        #     'left_hip', 'right_hip',
        #     'left_knee', 'right_knee',
        #     'left_ankle', 'right_ankle',
        #     'left_heel', 'right_heel',
        #     'left_foot_index', 'right_foot_index',
        # ]

        self._landmark_names = [
            "wrist",
            "thumb1",
            "thumb2",
            "thumb3",
            "thumb4",
            "forefinger1",
            "forefinger2",
            "forefinger3",
            "forefinger4",
            "middle_finger1",
            "middle_finger2",
            "middle_finger3",
            "middle_finger4",
            "ring_finger1",
            "ring_finger2",
            "ring_finger3",
            "ring_finger4",
            "pinky_finger1",
            "pinky_finger2",
            "pinky_finger3",
            "pinky_finger4"
        ]

    def __call__(self, landmarks):
        """Normalizes pose landmarks and converts to embedding
        归一化姿势landmarks并转换为embedding

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances defined in `_get_pose_distance_embedding`.
          具有形状 (M, 3) 的姿势embedding的 Numpy 数组，其中“M”是“_get_pose_distance_embedding”中定义的成对距离的数量。
        """
        # assert landmarks.shape[0] == len(self._landmark_names), 'Unexpected number of landmarks: {}'.format(
        #     landmarks.shape[0])

        # 获取 landmarks.
        landmarks = np.copy(landmarks)

        # Normalize landmarks.
        landmarks = self._normalize_pose_landmarks(landmarks)
        return landmarks

    def _normalize_pose_landmarks(self, landmarks):
        """Normalizes landmarks translation and scale.归一化landmarks的平移和缩放"""
        landmarks = np.copy(landmarks)
        # 减去均值
        landmarks -= np.mean(landmarks, axis=0)
        # 除以关键点两两之间的最大距离（2范数）
        max_inner_distance = np.linalg.norm(landmarks[None, ...] - landmarks[:, None, ...], axis=2).max()
        landmarks /= max_inner_distance

        return landmarks
    
    def calc_max_inner_distance(self, landmarks):
        max_dist = 0
        for i in range(landmarks.shape[0]):
            for j in range(i + 1, landmarks.shape[0]):
                max_dist = max(max_dist, np.sqrt(((landmarks[i] - landmarks[j])**2).sum()))
        
        return max_dist

    def _get_pose_center(self, landmarks):
        """Calculates pose center as point between hips.将姿势中心计算为臀部之间的点。"""
        left_hip = landmarks[self._landmark_names.index('forefinger1')]
        right_hip = landmarks[self._landmark_names.index('pinky_finger1')]
        center = (left_hip + right_hip) * 0.5
        return center

    def _get_pose_size(self, landmarks, torso_size_multiplier):
        """Calculates pose size.计算姿势大小。

        它是下面两个值的最大值:
          * 躯干大小乘以`torso_size_multiplier`
          * 从姿势中心到任何姿势地标的最大距离
        """
        # 这种方法仅使用 2D landmarks来计算姿势大小.
        landmarks = landmarks[:, :2]

        # 臀部中心。
        left_hip = landmarks[self._landmark_names.index('forefinger1')]
        right_hip = landmarks[self._landmark_names.index('pinky_finger1')]
        hips = (left_hip + right_hip) * 0.5

        # # 两肩中心。
        # left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
        # right_shoulder = landmarks[self._landmark_names.index('right_shoulder')]
        # shoulders = (left_shoulder + right_shoulder) * 0.5

        # 两肩中心。
        left_shoulder = landmarks[self._landmark_names.index('wrist')]
        right_shoulder = landmarks[self._landmark_names.index('wrist')]
        shoulders = (left_shoulder + right_shoulder) * 0.5

        # 躯干尺寸作为最小的身体尺寸。
        torso_size = np.linalg.norm(shoulders - hips)

        # 到姿势中心的最大距离。
        pose_center = self._get_pose_center(landmarks)
        max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

        return max(torso_size * torso_size_multiplier, max_dist)

    def _get_pose_distance_embedding(self, landmarks):
        """Converts pose landmarks into 3D embedding.
            将姿势landmarks转换为 3D embedding.
        我们使用几个成对的 3D 距离来形成姿势embedding。 所有距离都包括带符号的 X 和 Y 分量。
        我们使用不同类型的对来覆盖不同的姿势类别。 Feel free to remove some or add new.

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances.
        """
        # embedding = np.array([
        #     # One joint.

        #     self._get_distance(
        #         self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
        #         self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder')),

        #     self._get_distance_by_names(landmarks, 'left_shoulder', 'left_elbow'),
        #     self._get_distance_by_names(landmarks, 'right_shoulder', 'right_elbow'),

        #     self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist'),
        #     self._get_distance_by_names(landmarks, 'right_elbow', 'right_wrist'),

        #     self._get_distance_by_names(landmarks, 'left_hip', 'left_knee'),
        #     self._get_distance_by_names(landmarks, 'right_hip', 'right_knee'),

        #     self._get_distance_by_names(landmarks, 'left_knee', 'left_ankle'),
        #     self._get_distance_by_names(landmarks, 'right_knee', 'right_ankle'),

        #     # Two joints.

        #     self._get_distance_by_names(landmarks, 'left_shoulder', 'left_wrist'),
        #     self._get_distance_by_names(landmarks, 'right_shoulder', 'right_wrist'),

        #     self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle'),
        #     self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle'),

        #     # Four joints.

        #     self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
        #     self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

        #     # Five joints.

        #     self._get_distance_by_names(landmarks, 'left_shoulder', 'left_ankle'),
        #     self._get_distance_by_names(landmarks, 'right_shoulder', 'right_ankle'),

        #     self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
        #     self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

        #     # Cross body.

        #     self._get_distance_by_names(landmarks, 'left_elbow', 'right_elbow'),
        #     self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'),

        #     self._get_distance_by_names(landmarks, 'left_wrist', 'right_wrist'),
        #     self._get_distance_by_names(landmarks, 'left_ankle', 'right_ankle'),

        #     # Body bent direction.

        #     # self._get_distance(
        #     #     self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
        #     #     landmarks[self._landmark_names.index('left_hip')]),
        #     # self._get_distance(
        #     #     self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
        #     #     landmarks[self._landmark_names.index('right_hip')]),
        # ])

        embedding = np.array([
            # One joint.

            self._get_distance(
                self._get_average_by_names(landmarks, 'forefinger1', 'pinky_finger1'),
                self._get_average_by_names(landmarks, 'wrist', 'wrist')),

            self._get_distance_by_names(landmarks, 'wrist', 'thumb1'),
            self._get_distance_by_names(landmarks, 'wrist', 'forefinger1'),
            self._get_distance_by_names(landmarks, 'wrist', 'middle_finger1'),
            self._get_distance_by_names(landmarks, 'wrist', 'ring_finger1'),
            self._get_distance_by_names(landmarks, 'wrist', 'pinky_finger1'),

            self._get_distance_by_names(landmarks, 'thumb1', 'thumb2'),
            self._get_distance_by_names(landmarks, 'forefinger1', 'forefinger2'),
            self._get_distance_by_names(landmarks, 'middle_finger1', 'middle_finger2'),
            self._get_distance_by_names(landmarks, 'ring_finger1', 'ring_finger2'),
            self._get_distance_by_names(landmarks, 'pinky_finger1', 'pinky_finger2'),

            self._get_distance_by_names(landmarks, 'thumb2', 'thumb3'),
            self._get_distance_by_names(landmarks, 'forefinger2', 'forefinger3'),
            self._get_distance_by_names(landmarks, 'middle_finger2', 'middle_finger3'),
            self._get_distance_by_names(landmarks, 'ring_finger2', 'ring_finger3'),
            self._get_distance_by_names(landmarks, 'pinky_finger2', 'pinky_finger3'),

            self._get_distance_by_names(landmarks, 'thumb3', 'thumb4'),
            self._get_distance_by_names(landmarks, 'forefinger3', 'forefinger4'),
            self._get_distance_by_names(landmarks, 'middle_finger3', 'middle_finger4'),
            self._get_distance_by_names(landmarks, 'ring_finger3', 'ring_finger4'),
            self._get_distance_by_names(landmarks, 'pinky_finger3', 'pinky_finger4'),

            # Two joints.
            self._get_distance_by_names(landmarks, 'wrist', 'thumb2'),
            self._get_distance_by_names(landmarks, 'wrist', 'forefinger2'),
            self._get_distance_by_names(landmarks, 'wrist', 'middle_finger2'),
            self._get_distance_by_names(landmarks, 'wrist', 'ring_finger2'),
            self._get_distance_by_names(landmarks, 'wrist', 'pinky_finger2'),

            self._get_distance_by_names(landmarks, 'thumb1', 'thumb3'),
            self._get_distance_by_names(landmarks, 'forefinger1', 'forefinger3'),
            self._get_distance_by_names(landmarks, 'middle_finger1', 'middle_finger3'),
            self._get_distance_by_names(landmarks, 'ring_finger1', 'ring_finger3'),
            self._get_distance_by_names(landmarks, 'pinky_finger1', 'pinky_finger3'),

            self._get_distance_by_names(landmarks, 'thumb2', 'thumb4'),
            self._get_distance_by_names(landmarks, 'forefinger2', 'forefinger4'),
            self._get_distance_by_names(landmarks, 'middle_finger2', 'middle_finger4'),
            self._get_distance_by_names(landmarks, 'ring_finger2', 'ring_finger4'),
            self._get_distance_by_names(landmarks, 'pinky_finger2', 'pinky_finger4'),

            # Three joints.
            self._get_distance_by_names(landmarks, 'wrist', 'thumb3'),
            self._get_distance_by_names(landmarks, 'wrist', 'forefinger3'),
            self._get_distance_by_names(landmarks, 'wrist', 'middle_finger3'),
            self._get_distance_by_names(landmarks, 'wrist', 'ring_finger3'),
            self._get_distance_by_names(landmarks, 'wrist', 'pinky_finger3'),

            self._get_distance_by_names(landmarks, 'thumb1', 'thumb4'),
            self._get_distance_by_names(landmarks, 'forefinger1', 'forefinger4'),
            self._get_distance_by_names(landmarks, 'middle_finger1', 'middle_finger4'),
            self._get_distance_by_names(landmarks, 'ring_finger1', 'ring_finger4'),
            self._get_distance_by_names(landmarks, 'pinky_finger1', 'pinky_finger4'),


            # Four joints.
            self._get_distance_by_names(landmarks, 'wrist', 'thumb4'),
            self._get_distance_by_names(landmarks, 'wrist', 'forefinger4'),
            self._get_distance_by_names(landmarks, 'wrist', 'middle_finger4'),
            self._get_distance_by_names(landmarks, 'wrist', 'ring_finger4'),
            self._get_distance_by_names(landmarks, 'wrist', 'pinky_finger4'),


            # Cross body.
            self._get_distance_by_names(landmarks, 'thumb1', 'forefinger2'),
            self._get_distance_by_names(landmarks, 'forefinger1', 'middle_finger2'),
            self._get_distance_by_names(landmarks, 'middle_finger1', 'ring_finger2'),
            self._get_distance_by_names(landmarks, 'ring_finger1', 'pinky_finger2'),

            self._get_distance_by_names(landmarks, 'thumb2', 'forefinger3'),
            self._get_distance_by_names(landmarks, 'forefinger2', 'middle_finger3'),
            self._get_distance_by_names(landmarks, 'middle_finger2', 'ring_finger3'),
            self._get_distance_by_names(landmarks, 'ring_finger2', 'pinky_finger3'),

            
            self._get_distance_by_names(landmarks, 'thumb3', 'forefinger4'),
            self._get_distance_by_names(landmarks, 'forefinger3', 'middle_finger4'),
            self._get_distance_by_names(landmarks, 'middle_finger3', 'ring_finger4'),
            self._get_distance_by_names(landmarks, 'ring_finger3', 'pinky_finger4'),


        ])

        return embedding

    def _get_average_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) * 0.5

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)

    def _get_distance(self, lmk_from, lmk_to):
        return lmk_to - lmk_from

