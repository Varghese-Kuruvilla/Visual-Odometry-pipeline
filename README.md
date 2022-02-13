<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<br />



<h3 align="center">Visual Odometry Pipeline</h3>

  <p align="center">
    Provides a basic visual odometry pipeline (feature extraction, feature matching, motion estimation) with several configurable options. 
    <br />
    <!-- <a href=""><strong>Explore the docs »</strong></a> -->
    <!-- <br /> -->
    <br />
    <a href="https://github.com/Varghese-Kuruvilla/Visual-Odometry-pipeline/issues">Report Bug</a>
    ·
    <a href="https://github.com/Varghese-Kuruvilla/Visual-Odometry-pipeline/issues">Request Feature</a>
  </p>

</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#a-short-introduction-to-visual-odometry">A Short Introduction to Visual Odometry</a></li>
        <li><a href="#Description">A Brief Description of the Algorithm</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
This project aims to build a simple visual odometry pipeline that can be used as a baseline for further research work. The current documentation focuses specifically on feature based stereo visual odometry.

## A Short Introduction to Visual Odometry
**What is Visual Odometry?** 

It is the estimation of the robot pose based on the sequence of images that it captures as it moves through the environment.

**The basic VO pipeline** 

This project follows a basic visual odometry pipeline, which consists of the following steps:
- **Feature Extraction:** Reliable and repeatable features need to be extracted from the image. The current version of the code support SIFT, SURF, ORB and R2D2.

- **Feature Matching:** The extracted features are matched across frames using the mutual nearest neighbour algorithm.

- **Motion Estimation:** Motion estimation is carried out by using 3D points and their 2D correspondences (the solvePnPRansac function from OpenCV is used). For our experiments on the real robot, a ZED Mini camera was used which directly gives us the 3D points. For the Kitti dataset, we used Monodepth2 to find the corresponding depth maps.


<!--GETTING STARTED -->
## Getting Started
## Installation
Clone the repository
```
git clone https://github.com/Varghese-Kuruvilla/Visual-Odometry-pipeline.git
```

The packages required for getting started can be installed from the requirements file.
```
pip3 install -r requirements.txt
```

## Usage
### Running the VO code
- The current version of the code requires the RGB images and the corresponding depth images stored in a single folder in the following manner:
  - 000000_depth.npy 
  - 000000.png
  - 000001_depth.npy
  - 000001.png and so on.


- The configuration is specified through the file vo_params.yaml. A minimal example is shown below:
```
vo_method: "rgbd" #Choose from monocular or rgbd. The monocular method is work in progress, rgbd is stable
feature_extractor: "r2d2" #Choose from sift, orb, r2d2

#For offline mode
#The code for realtime VO using a depth camera will be shortly added
#Folder containing the rgb images: should be of the form *.png

image_path: ""

#Camera intrinsic matrix of the form [fx,0,cx,0,fy,cy,0,0,1]
#For Kitti

camera_intrinsic_matrix:
  - 721.53 
  - 0.0 
  - 609.55
  - 0.0 
  - 721.53
  - 172.85
  - 0.0 
  - 0.0 
  - 1.0

output_filename: ../global_poses #Saved as a .npy file
visualize_results: True #Visualize the extracted features and the matches between the images

##Parameters for plotting and evaluating the ATE, RPE
#GT should be in the KITTI ground truth format

gt_txt_file_path : "../plot_utils/data/03.txt"
#The poses file is automatically generated on running vo_runner.py
poses_file_path : "../plot_utils/data/global_poses.npy"
```

- The VO pipeline can be run from the vo_runner python script:
```
python3 vo_runner.py
```
### Plotting the trajectories and computing the absolute translational error(ATE) and relative position error(RPE)
- Run the script plot_traj.py(in plot_utils) to plot the ground truth and position of the vehicle as estimated by visual odometry
```
python3 plot_traj.py
```
- Run the script prepare_data.py(in plot_utils) to prepare the data for estimating the ATE and the RPE.
```
python3 prepare_data.py
```
- Run the script kittievalodom.py to estimate both the ATE and the RPE

<p align="right">(<a href="#top">back to top</a>)</p>

<p align="right">(<a href="#top">back to top</a>)</p>

<!--ROADMAP -->
## Roadmap
- [x] Add offline stereo visual odometry
- [x] Add scripts for plotting trajectories
- [x] Add scripts for error computation
- [ ] Add realtime stereo visual odometry
- [ ] Add monocular visual odometry

<!--CONTRIBUTING-->
## Contributing
Any contributions to this project are encouraged and are greatly appreciated.
If you have a suggestion that would improve the project, please fork the project and create a pull request. You can also open as issue.
Don't forget to give the project a star! Thanks!

11. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!--LICENSE-->
Distributed under the MIT License. See ```License``` for more information.

<!-- CONTACT -->
## Contact
- Shrutheesh Raman - shrutheesh99@gmail.com (https://github.com/ShrutheeshIR)
- Varghese Kuruvilla  - vkuruvilla789@gmail.com (https://github.com/Varghese-Kuruvilla)


Project Link: [Project Link](https://github.com/Varghese-Kuruvilla/Visual-Odometry-pipeline)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

* [R2D2 Naver labs](https://github.com/naver/r2d2)
* [David Scaramuzza visual odometry tutorial](https://rpg.ifi.uzh.ch/docs/Visual_Odometry_Tutorial.pdf)
* [Github Readme template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
<!-- [contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/Varghese-Kuruvilla/Visual-Odometry-pipeline/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png -->
