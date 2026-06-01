# Review

## Khan

- MDE mostly rely on CNN based model
- There are no reliable cue for MDE (temporal information and stereo correspondences)
- Classical method rely on multi view geometry for cues and different camera parameters
- Classical method require aligment and calibration
- Commputational time and memory requirements are important challenges for classical method
- Advancement of Convolutional Neural Networks (CNN) and publicly available
datasets have signiﬁcantly improved the performance of MDE
- MDE aims to estimate distances between scene objects and the camera from one
viewpoint.
- MDE requires low cost embedded system.
- Sensor such as ToF and LiDAR is impractical because of their processing power, computational time, range limitation and cost
- MDE can play an important role for cost-sensitive application
- MDE is ill-posed as there is an ambiguity in the scale of the depth
- MDE can be considered as regression problem using standard loss function such as MSE
- Supervised model address this issue  by approximately learning the scale from
a set of training images.
- unsupervised and semi-supervised methods often utilise an extra input for training such as stereo image sets, visual odometry and 6D camera pose estimation to tackle the scale ambiguity issue
- Traditional method rely on the assumtion of having observation of the scene in space or time (stereo, muti-view and structure from motion)
- DE active method using sensor, and pasive method extract depth information from multiple image
- DL method can be categorized into supervised, semi-supervised and self-supervised.
- Supervised method is expensive
- Semi supervised unable to correct their own bias and require additional domain information such as camera focal lenght and sensor data.
- Self supervised methods suffer from generalization issues, can only perform on very limited scenarios with similar distribution as training set.
- Architecture consist of Fully convolutional, Encoder decoder, Auto-decoder, and CNN
- Yin et al taking advantage of 3D geometric constraints. A simple type of geometric constraint known as "virtual norm" is implementred which is determined by randomly sampled three points in the 3D reconstruction to obtain a high quality depth estimation. The method can estimate 3D structures of the scene and surface normals directly from depth maps.
- Many self-supervised method use 640 × 192 image dimension
- Shanshan et al proposed geometry-aware symmetric domain adaption which target the generalisation issue of training on synthetic data. The method uses symmetric style iamge translation and monocular depth prediction. Utilizing the CycleGAN, GASDA involves both real to unreal and unreal to real image translations together with an epipolar geometry of the real stereo images.
- Evaluation metrics using Absolute relative difference (AbsRel), Root mean square error (RMSE), RMSE (log), Square relative error (SqRel) and accuracy with threshold.
- Feature extraction in encoder model using pretrained model such as VGG, ResNet or DenseNet while desired depth prediction is obtained using the decoder network architecture

## Zhao

- 