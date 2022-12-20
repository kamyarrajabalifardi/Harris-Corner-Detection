# Harris-Corner-Detection

In this problem we try to implement Harris corner detection algoroithm using gradients of image and non maximum supression on harris score. The corners are detected for two images shown below:

<p align="center">
<img width = "250" src="https://user-images.githubusercontent.com/46090276/195120404-2361e582-f97b-4923-a1a1-a70cf6b7ad18.jpg" alt="im01">
<img width = "250" src="https://user-images.githubusercontent.com/46090276/195120928-d77f8dd4-6bd9-4877-a385-1f49a0fb25b2.jpg" alt="19">
</p>
<p align="center">
<img width = "250" src="https://user-images.githubusercontent.com/46090276/195120818-66e76181-fa80-47d8-b572-9fdbb3f3adb0.jpg" alt="im02">
<img width = "250" src="https://user-images.githubusercontent.com/46090276/195121142-6cdd502a-dae3-434a-8745-474e3acec947.jpg" alt="im02">
</p>

After extracting Corners we extract intensity descriptors and use KNN approach to find corresponding keypoints between the two images. The final result is shown below:
<p align="center">
<img width = "600" src="https://user-images.githubusercontent.com/46090276/195119811-091316d7-2a9f-46e5-b1e0-f8e651ef75c0.jpg" alt="21">
</p>
