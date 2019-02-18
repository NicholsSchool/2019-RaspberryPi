import java.util.ArrayList;
import java.util.Arrays;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import edu.wpi.first.vision.VisionPipeline;

/**
 * LinePipline gets the heading of alignment lines using contour analysis and
 * 3D-2D point correspondence.
 */
public class LinePipeline2 implements VisionPipeline {

    private static final int THRESHOLD = 160;
    private static final Scalar BLUE = new Scalar(255, 0, 0), GREEN = new Scalar(0, 255, 0),
            RED = new Scalar(0, 0, 255), YELLOW = new Scalar(0, 255, 255), ORANGE = new Scalar(0, 165, 255),
            MAGENTA = new Scalar(255, 0, 255);

    private static final double FOCAL_LENGTH = 420; // in pixels, needs tuning if res is changed
    private static final double CAMERA_ANGLE_OFFSET = -23 * Math.PI / 180;
    private static final double CAMERA_X_OFFSET = 0; // in inches
    private static final double CAMERA_Y_OFFSET = 0;

    public Mat dst;
    public double angleToLine;
    public double distanceToLine;
    public double angleToWall;

    private Mat src;
    private ArrayList<MatOfPoint> contours;
    private MatOfPoint line;
    private MatOfPoint2f line2f;
    private Mat rotationVector;
    private Mat translationVector;
    private double deltaX;
    private double deltaY;

    @Override
    public void process(Mat src) {
        if (src.empty()) {
            return;
        }
        this.src = src;

        getLines();
        getRealLine();

        getVectors();
        System.out.println("Rotation Vector: " + vtos(rotationVector));
        System.out.println("Translation Vector: " + vtos(translationVector));

        getWorldPos();
        getHeading();
    }

    private void getLines() {
        dst = new Mat();

        // Extract whites
        Core.inRange(src, new Scalar(THRESHOLD, THRESHOLD, THRESHOLD), new Scalar(255, 255, 255), dst);

        // Find all external contours
        contours = new ArrayList<>();
        try {
            Imgproc.findContours(dst, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        } catch (CvException e) {
            // Sometimes the Mat format gets messed up when switching cameras
            System.out.println(e.getMessage());
        }

        dst = src;
    }

    private void getRealLine() {
        MatOfPoint realLine = null;
        MatOfPoint2f realLine2f = null;
        double closest = 1;

        // Approximate contours with polygons
        for (MatOfPoint contour : contours) {
            // Only include contours larger than 1/300 of the screen
            if (Imgproc.contourArea(contour) > dst.width() * dst.height() / 300) {
                // Convert format
                MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());

                // Epsilon will be 2% of the perimeter, a lower epsilon will result in more
                // vertices
                double epsilon = 0.02 * Imgproc.arcLength(contour2f, true);

                // Approximate contour to polygon
                Imgproc.approxPolyDP(contour2f, contour2f, epsilon, true);

                // Convert format back
                contour = new MatOfPoint(contour2f.toArray());

                // Do the checks if it has 4 vertices and is convex
                if (contour.rows() == 4 && Imgproc.isContourConvex(contour)) {
                    Imgproc.drawContours(dst, Arrays.asList(contour), -1, YELLOW, 1);

                    // Get the bounding rect
                    RotatedRect rect = Imgproc.minAreaRect(contour2f);
                    double h = rect.size.height;
                    double w = rect.size.width;
                    double ratio = Math.max(h, w) / Math.min(h, w);

                    // distance to center of the screen as a percentage
                    double distanceToCenter = Math.abs(rect.center.x / dst.width() - 0.5);

                    // The "real" line will be the rectangle greater than a certain length to width
                    // ratio that is closest to the center of the screen
                    if (ratio > 4 && distanceToCenter < closest) {
                        realLine = contour;
                        realLine2f = contour2f;
                        closest = distanceToCenter;
                    }
                } else {
                    // Draw the contour
                    Imgproc.drawContours(dst, Arrays.asList(contour), -1, ORANGE, 1);
                }
            } else {
                // Draw the contour
                Imgproc.drawContours(dst, Arrays.asList(contour), -1, RED, 1);
            }
        }

        line = reorderPoints(realLine);
        line2f = reorderPoints(realLine2f);
    }

    private void getVectors() {
        if (line == null) {
            return;
        }

        // all intrinsics are in pixel values
        final double principalOffsetX = dst.width() / 2;
        final double principalOffsetY = dst.height() / 2;
        Mat cameraIntrinsics = Mat.zeros(3, 3, CvType.CV_64FC1);
        cameraIntrinsics.put(0, 0, FOCAL_LENGTH);
        cameraIntrinsics.put(0, 2, principalOffsetX);
        cameraIntrinsics.put(1, 1, FOCAL_LENGTH);
        cameraIntrinsics.put(1, 2, principalOffsetY);
        cameraIntrinsics.put(2, 2, 1);

        Point3[] worldSpacePointsArr = new Point3[4];
        // (-1, 0, 0) will be the bottom left corner, points are in counterclockwise
        // order
        worldSpacePointsArr[0] = new Point3(-1, 0, 0);
        worldSpacePointsArr[1] = new Point3(1, 0, 0);
        worldSpacePointsArr[2] = new Point3(1, 18, 0);
        worldSpacePointsArr[3] = new Point3(-1, 18, 0);
        MatOfPoint3f realSpacePoints = new MatOfPoint3f(worldSpacePointsArr);

        rotationVector = new Mat();
        translationVector = new Mat();
        Calib3d.solvePnP(realSpacePoints, line2f, cameraIntrinsics, new MatOfDouble(), rotationVector,
                translationVector);

        Point3[] shiftedWorldSpacePointsArr = new Point3[4];
        shiftedWorldSpacePointsArr[0] = new Point3(-1, 0, 2);
        shiftedWorldSpacePointsArr[1] = new Point3(1, 0, 2);
        shiftedWorldSpacePointsArr[2] = new Point3(1, 18, 2);
        shiftedWorldSpacePointsArr[3] = new Point3(-1, 18, 2);
        MatOfPoint3f shiftedRealSpacePoints = new MatOfPoint3f(shiftedWorldSpacePointsArr);
        MatOfPoint2f shiftedImagePoints = new MatOfPoint2f();
        Calib3d.projectPoints(shiftedRealSpacePoints, rotationVector, translationVector, cameraIntrinsics,
                new MatOfDouble(), shiftedImagePoints);

        drawBox(line2f, shiftedImagePoints);
    }

    // world pos relative to camera
    private void getWorldPos() {
        Mat rotationInv = new Mat();
        Calib3d.Rodrigues(rotationVector, rotationInv);
        Core.transpose(rotationInv, rotationInv);

        Mat camWorldPos = new Mat();
        Core.multiply(rotationInv, new Scalar(-1), rotationInv);
        // use Core.gemm() instead of Core.multiply() for matrices of different
        // dimensions
        Core.gemm(rotationInv, translationVector, 1, new Mat(), 0, camWorldPos);

        System.out.println("cam world X: " + camWorldPos.get(0, 0)[0]);
        System.out.println("cam world Y: " + camWorldPos.get(1, 0)[0]);
        System.out.println("cam world Z: " + camWorldPos.get(2, 0)[0]);

        // account for camera offset, rotate about x axis
        Mat angleOffset = Mat.zeros(3, 3, CvType.CV_64FC1);
        angleOffset.put(0, 0, 1);
        angleOffset.put(1, 1, Math.cos(CAMERA_ANGLE_OFFSET));
        angleOffset.put(1, 2, -Math.sin(CAMERA_ANGLE_OFFSET));
        angleOffset.put(2, 1, Math.sin(CAMERA_ANGLE_OFFSET));
        angleOffset.put(2, 2, Math.cos(CAMERA_ANGLE_OFFSET));

        Mat positionOffset = Mat.zeros(3, 1, CvType.CV_64FC1);
        positionOffset.put(0, 0, CAMERA_X_OFFSET);
        positionOffset.put(1, 0, CAMERA_Y_OFFSET);

        Mat worldPos = new Mat();
        Core.gemm(angleOffset, camWorldPos, 1, new Mat(), 0, worldPos);
        Core.add(worldPos, positionOffset, worldPos);

        System.out.println("world X: " + worldPos.get(0, 0)[0]);
        System.out.println("world Y: " + worldPos.get(1, 0)[0]);
        System.out.println("world Z: " + worldPos.get(2, 0)[0]);
        deltaX = worldPos.get(0, 0)[0];
        deltaY = worldPos.get(2, 0)[0];
    }

    private void getHeading() {
        angleToLine = Math.tan(deltaX / deltaY) * 180 / Math.PI;
        distanceToLine = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
        // angle to wall is the z rotation of the camera to the line
        Mat rotationInv = new Mat();
        Calib3d.Rodrigues(rotationVector, rotationInv);
        Core.transpose(rotationInv, rotationInv);
        angleToWall = rotationInv.get(2, 0)[0];
    }

    private String vtos(Mat v) {
        String s = "<";
        for (int i = 0; i < v.rows(); i++) {
            if (i != 0) {
                s += ", ";
            }
            s += v.get(i, 0)[0];
        }
        s += ">";

        return s;
    }

    private void drawBox(MatOfPoint2f imagePoints, MatOfPoint2f shiftedImagePoints) {
        Imgproc.drawContours(dst, Arrays.asList(new MatOfPoint(imagePoints.toArray())), -1, GREEN, 2);

        for (int i = 0; i < imagePoints.rows(); i++) {
            Imgproc.line(dst, new Point(imagePoints.get(i, 0)), new Point(shiftedImagePoints.get(i, 0)), BLUE, 2);
        }

        Imgproc.drawContours(dst, Arrays.asList(new MatOfPoint(shiftedImagePoints.toArray())), -1, MAGENTA, 2);

    }

    private MatOfPoint reorderPoints(MatOfPoint m) {
        Point[] points = m.toArray();

        // Get the bottom two vertices
        int lowest = 0;
        int second = 0;
        for (int i = 0; i < points.length; i++) {
            if (points[i].y > points[lowest].y) {
                second = lowest;
                lowest = i;
            } else if (points[i].y > points[second].y) {
                second = i;
            }
        }

        // Get the bottom left vertex
        int bl = points[lowest].x < points[second].x ? lowest : second;

        Point[] reordered = new Point[points.length];
        for (int i = 0; i < reordered.length; i++) {
            reordered[i] = points[(i + bl) % points.length];
        }

        return new MatOfPoint(reordered);
    }

    private MatOfPoint2f reorderPoints(MatOfPoint2f m) {
        Point[] points = m.toArray();

        // Get the bottom two vertices
        int lowest = 0;
        int second = 0;
        for (int i = 0; i < points.length; i++) {
            if (points[i].y > points[lowest].y) {
                second = lowest;
                lowest = i;
            } else if (points[i].y > points[second].y) {
                second = i;
            }
        }

        // Get the bottom left vertex
        int bl = points[lowest].x < points[second].x ? lowest : second;

        Point[] reordered = new Point[points.length];
        for (int i = 0; i < reordered.length; i++) {
            reordered[i] = points[(i + bl) % points.length];
        }

        return new MatOfPoint2f(reordered);
    }
}
