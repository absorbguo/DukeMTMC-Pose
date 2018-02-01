# DukeMTMC-Pose
Pedestrian Pose Annotation for [DukeMTMC-reID](https://github.com/layumi/DukeMTMC-reID_evaluation) by machine(Python and Matlab API).

It is a small work but we think it may help the community to focus on the body structure of the pedestrian.
The application can be pedestrain hallucination (generation), person re-identification and tracking. 

### Data
18 pedestrian body points have been included in `result` folder.

|File  | Description | 
| --------   | -----  |
|gallery_points.json  | The points of gallery images. |
|train_points.json  | The points of training images.|
|query_points.json  | The points of query images.|

We generated the pose by [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose). 
Thank their great works. 

**(Note that value is null if the point has not been detected.)**

![](https://github.com/layumi/DukeMTMC-Pose/blob/master/demo.png) 

### API

python: `python_demo.ipynb`

matlab: `matlab_demo.m`


### License

Please refer to `LICENSE_DukeMTMC.txt` and `LICENSE_DukeMTMC-reID.txt`.
