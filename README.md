# 2019-RaspberryPi
FRC Team 4930's codebase for off-board vision processing on a Raspberry Pi, based off of the Java demo included in FRC's Raspberry Pi image. We use OpenCV to determine the robot's pose relative to the retroreflective tape on the cargo ship and the rocket.

![vision demo](https://github.com/NicholsSchool/2019-RaspberryPi/blob/master/demo%20images/vision%20demo.png?raw=true)

## Cool Features
- Contour analysis
- Sub-pixel corner detection
- 6D pose esimation with 3D-2D point correspondence
- Headings and distances to multiple custom waypoints

## Building & Deploying
Java 11 is required to build.  Set your path and/or JAVA_HOME environment
variable appropriately.

1) Run "./gradlew build"

On the rPi web dashboard:

1) Make the rPi writable by selecting the "Writable" tab
2) In the rPi web dashboard Application tab, select the "Uploaded Java jar"
   option for Application
3) Click "Browse..." and select the "2019-RaspberryPi-all.jar" file in
   your desktop project directory in the build/libs subdirectory
4) Click Save

The application will be automatically started.  Console output can be seen by
enabling console output in the Vision Status tab.
