# Learning to Detect Scene Landmarks for Camera Localization

This repository contains the source code for our paper:

**Learning to Detect Scene Landmarks for Camera Localization**  
Tien Do, Ondrej Miksik, Joseph DeGol, Hyun Soo Park, and Sudipta N. Sinha  
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022  
[Project webpage](???) | [Dataset](???) | [arXiv](???) 

# Abstract

Modern camera localization methods that use image retrieval, feature matching, and 3D structure-based pose 
estimation require long-term storage of numerous scene images or a vast amount of image features. That can make them 
unsuitable for resource constrained VR/AR devices and also raises serious privacy concerns. In this paper, we 
present a new learned camera localization technique that eliminates the need to store features or a detailed 3D 
point cloud. Our key idea is to %implicitly encode a sparse set of of 3D scene points, implicitly encode the 
appearance of a sparse yet salient set of 3D scene points into a convolutional neural network (CNN) that can detect 
them in query images whenever they are visible. We refer to these points as {\emph scene landmarks}. We also show 
that a CNN can be trained to regress bearing vectors for such landmarks even when they are not within the camera's 
field-of-view. We demonstrate that the predicted landmarks yield accurate pose estimates and that our method 
outperforms DSAC*, the state-of-the-art in learned localization. Furthermore, extending HLoc (an accurate method) 
by combining its correspondences with our predictions boosts its accuracy even further.

![teaser](media/teaser_wide.png)
**Figure 1: We present a new method to recognize scene-specific _scene landmarks_ to localize a camera, which 
preserves privacy and achieves high accuracy. [Left] Scene landmark detections in a query image obtained from a 
heatmap-based CNN architecture. [Middle] A visualization of the predicted heatmap scores. [Right] The 3D scene 
landmarks (in red) and the estimated camera pose (in blue) are shown over the 3D point cloud (in gray). The 3D point 
cloud is shown only for the purpose of visualization.**

# Indoor-6 dataset

### Description
Our Indoor-6 dataset was created from multiple RGB 30 fps videos captured in six indoor scenes over multiple days. 
We extracted frames at 3 fps and split them into training and test sets: 

1. scene1 (24 videos; 6289/799 images)
2. scene2 (12 videos; 3021/284 images) 
3. scene3 (18 videos; 4181/315 images) 
4. scene4 (4 videos; 1942/272 images) 
5. scene5 (15 videos; 4946/424 images) 
6. scene6 (6 videos; 1761/323 images) 

The pseudo ground truth (pGT) 3D point clouds and camera poses for each scene are computed using COLMAP. Figure 2 
shows the camera poses (in red) and point clouds (in gray) and for each scene, the number of video and images in the 
training and test split respectively. Compared to 7-scenes, the scenes in Indoor-6 are larger, 
have multiple rooms, contains illumination variations as the images span multiple days and different times of day.

![indoor6_sfm](media/SfM_indoor_6.png)
**Figure 2: Indoor-6 dataset SfM reconstructions.**

### Download
scene1 scene2 scene3 scene4 scene5 scene6

### Organization


# Code (coming soon)

[comment]: <> (### Installation)

[comment]: <> (### Training)

[comment]: <> (### Evaluation)

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Legal Notices

Microsoft and any contributors grant you a license to the Microsoft documentation and other content
in this repository under the [Creative Commons Attribution 4.0 International Public License](https://creativecommons.org/licenses/by/4.0/legalcode),
see the [LICENSE](LICENSE) file, and grant you a license to any code in the repository under the [MIT License](https://opensource.org/licenses/MIT), see the
[LICENSE-CODE](LICENSE-CODE) file.

Microsoft, Windows, Microsoft Azure and/or other Microsoft products and services referenced in the documentation
may be either trademarks or registered trademarks of Microsoft in the United States and/or other countries.
The licenses for this project do not grant you rights to use any Microsoft names, logos, or trademarks.
Microsoft's general trademark guidelines can be found at http://go.microsoft.com/fwlink/?LinkID=254653.

Privacy information can be found at https://privacy.microsoft.com/en-us/

Microsoft and any contributors reserve all other rights, whether under their respective copyrights, patents,
or trademarks, whether by implication, estoppel or otherwise.
