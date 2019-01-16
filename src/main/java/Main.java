
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

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

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

    @SuppressWarnings("MemberName")
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
    @SuppressWarnings("PMD.CyclomaticComplexity")
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
     */
    public static class CargoPipeline implements VisionPipeline {

        public int detected;
        public Mat dst;

        @Override
        public void process(Mat src) {
            if (src.empty()) {
                return;
            }

            detected = 0;

            dst = new Mat();

            // Convert to grayscale
            Imgproc.cvtColor(src, dst, Imgproc.COLOR_BGR2GRAY);

            // Separate bright areas from dark areas
            Imgproc.threshold(dst, dst, 127, 255, Imgproc.THRESH_BINARY);

            // Find all external contours
            ArrayList<MatOfPoint> contours = new ArrayList<>();
            Imgproc.findContours(dst, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            // Approximate contours with polygons
            src.copyTo(dst);
            for (MatOfPoint contour : contours) {

                // Only include large contours
                if (Imgproc.contourArea(contour) > 750) {
                    // Convert format
                    MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());

                    // Epsilon will be 5% of the perimeter, a lower epsilon will result in more
                    // vertices
                    double epsilon = 0.05 * Imgproc.arcLength(contour2f, true);

                    // Approximate contour to polygon using the Ramer-Douglas-Peucker algorithm
                    Imgproc.approxPolyDP(contour2f, contour2f, epsilon, true);

                    // Draw the polygon if it has 4 vertices
                    if (contour2f.rows() == 4) {
                        detected++;
                        Imgproc.drawContours(dst, Arrays.asList(new MatOfPoint(contour2f.toArray())), -1,
                                new Scalar(0, 0, 255), 2);
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

        System.out.print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nH E L L O ! ! !\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");

        // start cameras
        List<VideoSource> cameras = new ArrayList<>();
        for (CameraConfig cameraConfig : cameraConfigs) {
            cameras.add(startCamera(cameraConfig));
        }

        // start image processing on camera 0 if present
        if (cameras.size() >= 1) {
            CvSource outputStream = CameraServer.getInstance().putVideo("Vision Pipline Output", 400, 300);

            VisionThread visionThread = new VisionThread(cameras.get(0), new CargoPipeline(), pipeline -> {
                // do something with pipeline results
                System.out.println("Detected " + pipeline.detected + " line(s)!");
                outputStream.putFrame(pipeline.dst);
            });
            /*
             * something like this for GRIP: VisionThread visionThread = new
             * VisionThread(cameras.get(0), new GripPipeline(), pipeline -> { ... });
             */
            visionThread.start();
        }

        // loop forever
        for (;;) {
            try {
                Thread.sleep(10000);
            } catch (InterruptedException ex) {
                return;
            }
        }
    }
}
