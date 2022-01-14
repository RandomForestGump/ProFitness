# ProFitness

![image](https://user-images.githubusercontent.com/20444857/149428449-9052ecc9-3fe8-47ad-a432-c314ec523519.png)


ProFitness is a Computer Vision based Strength training app built using Streamlit, OpenCV and Mediapipe.

https://user-images.githubusercontent.com/20444857/149428028-f68fb959-0b42-4c1a-9603-dfa70e197316.mp4

## Abstract
Human activity recognition has emerged as an active research area in recent years. With the advancement in mobile and wearable devices, various sensors are ubiquitous and widely available gathering data from a broad spectrum of peoples’ daily life activities. Research studies thoroughly assessed lifestyle activities and are increasingly concentrated on a variety of sport exercises. In this project, we examine strength training exercises and whether a product can be constructed to analyze the patterns of exercises and critique poses during it.


## MediaPipe: Training, Landmark Detection and Classification
Human pose estimation from video plays a critical role in various applications such as quantifying physical exercises, sign language recognition, and full-body gesture control. For example, it can form the basis for yoga, dance, and fitness applications. It can also enable the overlay of digital content and information on top of the physical world in augmented reality.

MediaPipe Pose is a ML solution for high-fidelity body pose tracking, inferring 33 3D landmarks and background segmentation mask on the whole body from RGB video frames utilizing our BlazePose research that also powers the ML Kit Pose Detection API. Current state-of-the-art approaches rely primarily on powerful desktop environments for inference, whereas our method achieves real-time performance on most modern mobile phones, desktops/laptops, in python and even on the web.

The detector is inspired by our own lightweight BlazeFace model, used in MediaPipe Face Detection, as a proxy for a person detector. It explicitly predicts two additional virtual key points that firmly describe the human body center, rotation and scale as a circle. Inspired by Leonardo’s Vitruvian man, we predict the midpoint of a person’s hips, the radius of a circle circumscribing the whole person, and the incline angle of the line connecting the shoulder and hip midpoints. 


## Repetition Counter



To count the repetitions, the algorithm monitors the probability of a target pose class. Let’s take push-ups with its “up” and “down” terminal states:

- When the probability of the “down” pose class passes a certain threshold for the first time, the algorithm marks that the “down” pose class is entered.
- Once the probability drops below the threshold, the algorithm marks that the “down” pose class has been exited and increases the counter.

To avoid cases when the probability fluctuates around the threshold (e.g., when the user pauses between “up” and “down” states) causing phantom counts, the threshold used to detect when the state is exited is actually slightly lower than the one used to detect when the state is entered. It creates an interval where the pose class and the counter can’t be changed.


## Pose Correction


![image](https://user-images.githubusercontent.com/20444857/149428399-a9053560-f4f5-4a03-bfea-6d03209c5bdf.png)

A novel algorithm was developed for Pose Correction. This involves the following steps:

- Define the initial periodic curve of classification for terminal states of exercise
- Calculate the initial wavelength of this curve (x)
- For filtering and practical purposes, we consider a wavelength of 3x and call it as our window.
- We slide through the temporal classification curve and detect the length of the window where there is little to no change in the classification pattern.
- We call this an area of abruption, mark the timestamps and export the video of this time step plus a threshold.
- There are 2 areas of abruption as displayed in Figure

## FUTURE WORK

Some ideas on the business side of the product are:
- Can be a one stop shop for online training
- A revenue share model can be incorporated similar to youtube and it’s streamers
- Yoga and introduction of HIIT.
- Integration with food health apps for daily nutrition intake.




