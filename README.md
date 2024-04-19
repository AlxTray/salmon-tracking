# Running tracking

First, make sure Docker is first running on the system you are wishing to use this prototype system on.
Secondly, make sure all package requirements have been met, this can be easily done by running the command below in a terminal at the root project directory.
```
pip install -r requirements.txt
```
Make sure to be in a virtual environment before running.

To start the local inference Docker container run:
```
sudo inference server start
```
This will then pull the correct image for the target system.

Once the Docker image has been sucessfully pulled and started, simply run the *salmon_tracking.py* file:

For inferring on a video:
```
python salmon_tracking.py --video path\to\video\file.mp4
```

For inferring on a camera stream:
```
python salmon_tracking.py --webcam
```

Two windows will then be shown, one will showcase the current video stream from file or camera, with annotations and speed drawn on. And a second window that contains a 3-D plot showcasing the trajectories of all Salmon in the frame.
