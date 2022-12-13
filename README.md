# MC-Recognizer
A statistic information generator for videos using facial recognition and python’s computer vision library (Tested in Python 3.10.4)

Developed for Unix-like Systems.
#### This program is mainly intended for videos longer than 15 minutes, but can be modified to be used for shorter or longer videos.


# Installation
### Clone the Repository
> git clone "x"
### Create a new Python virtual environment
> python -m venv /path/to/cloned/repository
### Activate your virtual environment
> source /path/to/venv/bin/activate
### Install requirements with pip
> pip install -r requirements.txt
### Create your face recognition folder
> Make a new directory named "Face_Images"
### Gather your face photos in jpeg or png format
> Place these images in your Face_Images folder, the filename will be the face's identity.
> One image per face to identify is sufficient
### Get a source video in 'mp4' format
> This video will contain frames with the identies in your Face_Images folder.



# Running the program
> The program receives 4 arguments:
#### "--videopath" followed by the path to your source video
> Ex. --videopath path/to/myvideo/video.mp4
#### "--closeup_percentage" followed by a value, 1 to 100, representing the percentage of area of the screen the identity's face takes up
> This is useful if you are getting false positives, or you want a specific use case to gather statistics on closeup frames
>> Ex. --closeup_percentage 15
#### "--scale_factor" followed by a value, 0.1 to 1, a factor that will scale down your source video for faster processing
> Note that smaller values will likely speed up processing, but can elicit more false positive identifications.
>> Ex. --scale_factor 0.25
#### "--check_every" followed by an integer, representing the number of frames skipped between a frame being processed
> Will speed up processing, but results will be less accurate.
>> Ex. --check_every 5
>>> will process every 5 frames in the source video

## Example usage
> python mc_recognizer.py --videopath /path/to/video/video.mp4 --closeup_percentage 15 --scale_factor 0.25 --check_every 4
>> will run the program on the video "video.mp4", only identifying frames with face taking up 15% of the frame's area, scaling down the video to 1/4th of its original size, and processing every 4 frames.
