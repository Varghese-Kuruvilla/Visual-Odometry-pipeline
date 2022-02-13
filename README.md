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
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />



<h3 align="center">Visual Odometry Pipeline</h3>

  <p align="center">
    Provides a basic visual odometry pipeline (feature extraction, feature matching, motion estimation) with several configurable options. 
    <br />
    <a href="https://github.com/Varghese-Kuruvilla/Visual_Odometry"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Varghese-Kuruvilla/Visual_Odometry">View Demo</a>
    ·
    <a href="https://github.com/Varghese-Kuruvilla/Visual_Odometry/issues">Report Bug</a>
    ·
    <a href="https://github.com/Varghese-Kuruvilla/Visual_Odometry/issues">Request Feature</a>
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
The packages required for getting started can be installed from the requirements file.
```
pip3 install -r requirements.txt
```

## Usage
- The current version of the code requires the RGB images and the corresponding depth images stored in a single folder in the following manner:
  - 000000_depth.npy 
  - 000000.png
  - 000001_depth.npy
  - 000001.png and so on.


- The configuration is specified through the file vo_params.yaml. A minimal example is shown below:
```
vo_method: "rgbd" #Choose from monocular or rgbd. The monocular method is work in progress, rgbd is stable
feature_extractor: "r2d2" #Choose from sift, orb, r2d2
feature_matcher: "ratio_mutual_nn_matcher" #Choose from mnn_matcher, similarity_matcher, ratio_mutual_nn_matcher
#Folder containing the rgb images: should be of the form *.png
image_path: "/media/artpark/ELIZABETH/VO/data/03/image_2/"
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

visualize_results: True #Visualize the extracted features and the matches between the images
```

- The VO pipeline can be run from the vo_runner python script:
```
python3 vo_runner.py
```

<p align="right">(<a href="#top">back to top</a>)</p>

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTACT -->
## Contact
- Shrutheesh Raman - shrutheesh99@gmail.com
- Varghese Kuruvilla  - vkuruvilla789@gmail.com


Project Link: [https://github.com/Varghese-Kuruvilla/Visual_Odometry](https://github.com/Varghese-Kuruvilla/Visual_Odometry)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
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
[product-screenshot]: images/screenshot.png
