
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
import edu.wpi.first.networktables.EntryListenerFlags;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.vision.VisionThread;
import edu.wpi.first.vision.VisionRunner.Listener;

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

    public static final int CAMERA_RESOLUTION_X = 320, CAMERA_RESOLUTION_Y = 240, CAMERA_FPS = 30;
    private static final int NUM_OF_CAMERAS = 2;

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

    private static VisionThread visionThread;

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

        // start cameras
        List<VideoSource> cameras = new ArrayList<>();
        for (CameraConfig cameraConfig : cameraConfigs) {
            cameras.add(startCamera(cameraConfig));
        }

        // Start image processing if the correct number of camera streams are open
        if (cameras.size() == NUM_OF_CAMERAS) {
            CvSource cvStream = CameraServer.getInstance().putVideo("Pi Output", CAMERA_RESOLUTION_X,
                    CAMERA_RESOLUTION_Y);

            NetworkTable table = NetworkTableInstance.getDefault().getTable("vision");

            LinePipeline linePipeline = new LinePipeline();
            Listener<LinePipeline> lineCallback = pipeline -> {
                table.getEntry("angleToLine").setDouble(pipeline.angleToLine);
                table.getEntry("distanceToLine").setDouble(pipeline.distanceToLine);
                table.getEntry("angleToWall").setDouble(pipeline.angleToWall);

                cvStream.putFrame(pipeline.dst);
            };

            // EmptyPipeline emptyPipeline = new EmptyPipeline();
            // Listener<EmptyPipeline> emptyCallback = pipeline -> {
            //     cvStream.putFrame(pipeline.dst);
            // };

            visionThread = new VisionThread(cameras.get(0), linePipeline, lineCallback);
            visionThread.start();

            table.getEntry("camera").addListener(event -> {
                int camera = (int) event.value.getDouble();

                if (visionThread != null) {
                    visionThread.interrupt();
                }

                System.out.println("Switching to camera " + camera + "...");

                switch (camera) {
                case 0:
                    visionThread = new VisionThread(cameras.get(camera), linePipeline, lineCallback);
                    break;
                case 1:
                    visionThread = new VisionThread(cameras.get(camera), linePipeline, lineCallback);
                    break;
                }

                visionThread.start();
            }, EntryListenerFlags.kNew | EntryListenerFlags.kUpdate | EntryListenerFlags.kImmediate
                    | EntryListenerFlags.kLocal);

            // double num = 0;
            // loop forever
            while (true) {
                try {
                    System.out.println("Angle To Line: " + table.getEntry("angleToLine").getDouble(0));
                    System.out.println("Distance To Line: " + table.getEntry("distanceToLine").getDouble(0));
                    System.out.println("Angle To Wall: " + table.getEntry("angleToWall").getDouble(0));

                    // table.getEntry("camera").setDouble(num);
                    // if (num == 0) {
                    //     num = 1;
                    // } else {
                    //     num = 0;
                    // }
                    Thread.sleep(5000);
                } catch (InterruptedException e) {
                    System.out.println("Main thread sleep interrupted, are you switching cameras?");
                }
            }
        } else {
            System.out.println(
                    "ERROR: " + NUM_OF_CAMERAS + " expected, but " + cameras.size() + " camera(s) were detected!");
        }
    }
}
