
/*----------------------------------------------------------------------------*/
/* Copyright (c) 2018 FIRST. All Rights Reserved.                             */
/* Open Source Software - may be modified and shared by FRC teams. The code   */
/* must be accompanied by the FIRST BSD license file in the root directory of */
/* the project.                                                               */
/*----------------------------------------------------------------------------*/

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import edu.wpi.cscore.CvSource;
import edu.wpi.cscore.MjpegServer;
import edu.wpi.cscore.UsbCamera;
import edu.wpi.cscore.VideoSource;
import edu.wpi.first.cameraserver.CameraServer;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.vision.VisionPipeline;
import edu.wpi.first.vision.VisionThread;
import edu.wpi.first.vision.VisionRunner.Listener;

import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

/*
   JSON format:
   {
       "team": <team number>,
       "ntmode": <"client" or "server", "client" if unspecified>
       "cameras": [
           {
               "name": <camera name>
               "path": <path, e.g. "/dev/video0">
               "pixel format": <"MJPEG", "YUYV", etc>   // optional
               "width": <video mode width>              // optional
               "height": <video mode height>            // optional
               "fps": <video mode fps>                  // optional
               "brightness": <percentage brightness>    // optional
               "white balance": <"auto", "hold", value> // optional
               "exposure": <"auto", "hold", value>      // optional
               "properties": [                          // optional
                   {
                       "name": <property name>
                       "value": <property value>
                   }
               ],
               "stream": {                              // optional
                   "properties": [
                       {
                           "name": <stream property name>
                           "value": <stream property value>
                       }
                   ]
               }
           }
       ]
   }
 */

public final class Main {
    private static String configFile = "/boot/frc.json";

    private static final int CAMERA_RESOLUTION_X = 176, CAMERA_RESOLUTION_Y = 144, CAMERA_FPS = 30;
    private static final int NUM_OF_CAMERAS = 2;
    private static final Scalar BLUE = new Scalar(255, 0, 0), GREEN = new Scalar(0, 255, 0),
            RED = new Scalar(0, 0, 255);

    // @SuppressWarnings("MemberName")
    public static class CameraConfig {
        public String name;
        public String path;
        public JsonObject config;
        public JsonElement streamConfig;
    }

    public static int team;
    public static boolean server;
    public static List<CameraConfig> cameraConfigs = new ArrayList<>();

    private Main() {
    }

    /**
     * Report parse error.
     */
    public static void parseError(String str) {
        System.err.println("config error in '" + configFile + "': " + str);
    }

    /**
     * Read single camera configuration.
     */
    public static boolean readCameraConfig(JsonObject config) {
        CameraConfig cam = new CameraConfig();

        // name
        JsonElement nameElement = config.get("name");
        if (nameElement == null) {
            parseError("could not read camera name");
            return false;
        }
        cam.name = nameElement.getAsString();

        // path
        JsonElement pathElement = config.get("path");
        if (pathElement == null) {
            parseError("camera '" + cam.name + "': could not read path");
            return false;
        }
        cam.path = pathElement.getAsString();

        // stream properties
        cam.streamConfig = config.get("stream");

        cam.config = config;

        cameraConfigs.add(cam);
        return true;
    }

    /**
     * Read configuration file.
     */
    // @SuppressWarnings("PMD.CyclomaticComplexity")
    public static boolean readConfig() {
        // parse file
        JsonElement top;
        try {
            top = new JsonParser().parse(Files.newBufferedReader(Paths.get(configFile)));
        } catch (IOException ex) {
            System.err.println("could not open '" + configFile + "': " + ex);
            return false;
        }

        // top level must be an object
        if (!top.isJsonObject()) {
            parseError("must be JSON object");
            return false;
        }
        JsonObject obj = top.getAsJsonObject();

        // team number
        JsonElement teamElement = obj.get("team");
        if (teamElement == null) {
            parseError("could not read team number");
            return false;
        }
        team = teamElement.getAsInt();

        // ntmode (optional)
        if (obj.has("ntmode")) {
            String str = obj.get("ntmode").getAsString();
            if ("client".equalsIgnoreCase(str)) {
                server = false;
            } else if ("server".equalsIgnoreCase(str)) {
                server = true;
            } else {
                parseError("could not understand ntmode value '" + str + "'");
            }
        }

        // cameras
        JsonElement camerasElement = obj.get("cameras");
        if (camerasElement == null) {
            parseError("could not read cameras");
            return false;
        }
        JsonArray cameras = camerasElement.getAsJsonArray();
        for (JsonElement camera : cameras) {
            if (!readCameraConfig(camera.getAsJsonObject())) {
                return false;
            }
        }

        return true;
    }

    /**
     * Start running the camera.
     */
    public static VideoSource startCamera(CameraConfig config) {
        System.out.println("Starting camera '" + config.name + "' on " + config.path);
        CameraServer inst = CameraServer.getInstance();
        UsbCamera camera = new UsbCamera(config.name, config.path);
        MjpegServer server = inst.startAutomaticCapture(camera);

        Gson gson = new GsonBuilder().create();

        camera.setConfigJson(gson.toJson(config.config));
        camera.setConnectionStrategy(VideoSource.ConnectionStrategy.kKeepOpen);

        if (config.streamConfig != null) {
            server.setConfigJson(gson.toJson(config.streamConfig));
        }

        camera.setResolution(CAMERA_RESOLUTION_X, CAMERA_RESOLUTION_Y);
        camera.setFPS(CAMERA_FPS);

        return camera;
    }

    /**
     * Example pipeline.
     */
    public static class MyPipeline implements VisionPipeline {
        public int val;

        @Override
        public void process(Mat mat) {
            val += 1;
        }
    }

    /**
     * Get cargo lines using contour analysis.
     * 
     * @author Junqi Wu
     */
    public static class CargoPipeline implements VisionPipeline {

        public Mat dst;

        public double x;
        public double y;
        public double vx;
        public double vy;

        @Override
        public void process(Mat src) {
            if (src.empty()) {
                return;
            }

            dst = new Mat();

            // Convert to grayscale
            Imgproc.cvtColor(src, dst, Imgproc.COLOR_BGR2GRAY);

            // Separate bright areas from dark areas
            Imgproc.threshold(dst, dst, 127, 255, Imgproc.THRESH_BINARY);

            // Find all external contours
            ArrayList<MatOfPoint> contours = new ArrayList<>();
            try {
                Imgproc.findContours(dst, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            } catch (CvException e) {
                System.out.println(e.getMessage());
            }

            dst = src;

            // Approximate contours with polygons
            for (MatOfPoint contour : contours) {

                // Only include large contours
                if (Imgproc.contourArea(contour) > CAMERA_RESOLUTION_X * CAMERA_RESOLUTION_Y / 60) {
                    // Convert format
                    MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());

                    // Epsilon will be 4% of the perimeter, a lower epsilon will result in more
                    // vertices
                    double epsilon = 0.04 * Imgproc.arcLength(contour2f, true);

                    // Approximate contour to polygon using the Ramer-Douglas-Peucker algorithm
                    Imgproc.approxPolyDP(contour2f, contour2f, epsilon, true);

                    contour = new MatOfPoint(contour2f.toArray());

                    // Process the polygon if it has 4 vertices and is convex
                    if (contour.rows() == 4 && Imgproc.isContourConvex(contour)) {

                        // Get the best fit line
                        Mat line = new Mat();
                        Imgproc.fitLine(contour2f, line, Imgproc.DIST_L2, 0, 0.01, 0.01);
                        vx = line.get(0, 0)[0];
                        vy = line.get(1, 0)[0];
                        x = line.get(2, 0)[0];
                        y = line.get(3, 0)[0];

                        // Draw the contour
                        Imgproc.drawContours(dst, Arrays.asList(contour), -1, GREEN, 2);
                        // Draw the line
                        Imgproc.line(dst, new Point(x, y), new Point(x + vx * 100, y + vy * 100), GREEN, 2);
                    } else {
                        // Draw the contour
                        Imgproc.drawContours(dst, Arrays.asList(contour), -1, RED, 2);
                    }
                }
            }

        }

    }

    /**
     * Main.
     */
    public static void main(String... args) {
        if (args.length > 0) {
            configFile = args[0];
        }

        // read configuration
        if (!readConfig()) {
            return;
        }

        // start NetworkTables
        NetworkTableInstance ntinst = NetworkTableInstance.getDefault();
        if (server) {
            System.out.println("Setting up NetworkTables server");
            ntinst.startServer();
        } else {
            System.out.println("Setting up NetworkTables client for team " + team);
            ntinst.startClientTeam(team);
        }

        System.out.print(
                "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nH E L L O ! ! !\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");

        // start cameras
        List<VideoSource> cameras = new ArrayList<>();
        for (CameraConfig cameraConfig : cameraConfigs) {
            cameras.add(startCamera(cameraConfig));
        }

        // start image processing on camera 0 if present
        if (cameras.size() == NUM_OF_CAMERAS) {
            CvSource outputStream = CameraServer.getInstance().putVideo("Vision Pipline Output", CAMERA_RESOLUTION_X,
                    CAMERA_RESOLUTION_Y);

            VideoSource videoSource = cameras.get(0);
            CargoPipeline visionPipeline = new CargoPipeline();
            Listener<CargoPipeline> callback = pipeline -> {

                // x, y, vx, vy values should all be between 0 and 1
                NetworkTableInstance.getDefault().getTable("vision").getEntry("x")
                        .setDouble(pipeline.x / CAMERA_RESOLUTION_X);
                NetworkTableInstance.getDefault().getTable("vision").getEntry("y")
                        .setDouble(pipeline.y / CAMERA_RESOLUTION_Y);

                NetworkTableInstance.getDefault().getTable("vision").getEntry("vx").setDouble(pipeline.vx);
                NetworkTableInstance.getDefault().getTable("vision").getEntry("vy").setDouble(pipeline.vy);

                outputStream.putFrame(pipeline.dst);
            };

            VisionThread visionThread = new VisionThread(videoSource, visionPipeline, callback);
            /*
             * something like this for GRIP: VisionThread visionThread = new
             * VisionThread(cameras.get(0), new GripPipeline(), pipeline -> { ... });
             */
            visionThread.start();

            // loop forever
            while (true) {
                System.out.println("Switching cameras in 5 seconds");

                try {
                    Thread.sleep(5000);
                } catch (InterruptedException ex) {
                    return;
                }

                if (videoSource == cameras.get(0)) {
                    // videoSource = cameras.get(1);
                } else {
                    videoSource = cameras.get(0);
                }

                visionThread.interrupt();
                visionThread = new VisionThread(videoSource, visionPipeline, callback);
                visionThread.start();
            }
        } else {
            System.out.println("ERROR: ONLY " + cameras.size() + " CAMERA(S) ARE CONNECTED");
        }
    }
}
