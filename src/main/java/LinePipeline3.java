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
 * the direct vectors from 3D-2D point correspondence.
 */
public class LinePipeline3 implements VisionPipeline {

    @SuppressWarnings("unused")
    private static final Scalar BLUE = new Scalar(255, 0, 0), GREEN = new Scalar(0, 255, 0),
            RED = new Scalar(0, 0, 255), YELLOW = new Scalar(0, 255, 255), ORANGE = new Scalar(0, 165, 255),
            MAGENTA = new Scalar(255, 0, 255);

    private static final int THRESHOLD = 160;

    private static final double FOCAL_LENGTH = 310; // In pixels, needs tuning if res is changed
    private static final int TAPE_DISTANCE_BUFFER = 6; // Distance padding from tip of tape

    private Mat dst;

    private double tapeLength;
    private Mat camOffset;

    private MatOfPoint2f line;

    private Mat rvec;
    private Mat tvec;

    private double angleToLine;
    private double distanceToLine;
    private double angleToWall;

    public LinePipeline3(double tapeLength, double xOffset, double yOffset, double zOffset) {
        this.tapeLength = tapeLength;

        camOffset = Mat.zeros(3, 1, CvType.CV_64FC1);
        camOffset.put(0, 0, xOffset);
        camOffset.put(1, 0, yOffset);
        camOffset.put(2, 0, zOffset - TAPE_DISTANCE_BUFFER);
    }

    @Override
    public void process(Mat src) {
        if (src.empty()) {
            return;
        }

        getLine(src);

        if (line == null) {
            return;
        }

        getVectors();

        offsetAdjustment();

        setHeading();
    }

    // Get the contours of bright objects
    private void getLine(Mat src) {
        dst = new Mat();

        // Extract whites
        Core.inRange(src, new Scalar(THRESHOLD, THRESHOLD, THRESHOLD), new Scalar(255, 255, 255), dst);

        // Find all external contours
        ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        try {
            Imgproc.findContours(dst, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        } catch (CvException e) {
            // Sometimes the Mat format gets messed up when switching cameras
            System.out.println(e.getMessage());
        }

        dst = src;

        line = null;
        double closest = 1;

        // Approximate contours with polygons
        for (MatOfPoint contour : contours) {
            // Only include contours larger than 1/300 of the screen
            if (Imgproc.contourArea(contour) > dst.width() * dst.height() / 300) {
                // Convert format
                MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());

                // Epsilon will be 1.25% of the perimeter, a lower epsilon will result in more
                // vertices
                double epsilon = 0.0125 * Imgproc.arcLength(contour2f, true);

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

                    // Distance to center of the screen as a percentage
                    double distanceToCenter = Math.abs(rect.center.x / dst.width() - 0.5);

                    // The "real" line will be the rectangle greater than a certain length to width
                    // ratio that is closest to the center of the screen
                    if (ratio > 2 && distanceToCenter < closest) {
                        line = contour2f;
                        closest = distanceToCenter;
                    }
                } else {
                    // Draw the contour
                    // Imgproc.drawContours(dst, Arrays.asList(contour), -1, ORANGE, 1);
                }
            } else {
                // Draw the contour
                // Imgproc.drawContours(dst, Arrays.asList(contour), -1, RED, 1);
            }
        }
    }

    private void getVectors() {
        line = reorderPoints(line);

        // All camera intrinsics are in pixel values
        final double principalOffsetX = dst.width() / 2;
        final double principalOffsetY = dst.height() / 2;
        Mat camIntrinsics = Mat.zeros(3, 3, CvType.CV_64FC1);
        camIntrinsics.put(0, 0, FOCAL_LENGTH);
        camIntrinsics.put(0, 2, principalOffsetX);
        camIntrinsics.put(1, 1, FOCAL_LENGTH);
        camIntrinsics.put(1, 2, principalOffsetY);
        camIntrinsics.put(2, 2, 1);

        // 3D axes is same as 2D image axes, right is positive x, down is positive y,
        // foward is positive z (a clockwise axes system)
        // Points start with bottom left corner and run counter-clockwise
        // Bottom of the line as (0, 0, 0)
        Point3[] worldSpaceArr = new Point3[4];
        worldSpaceArr[0] = new Point3(-1, 0, 0);
        worldSpaceArr[1] = new Point3(1, 0, 0);
        worldSpaceArr[2] = new Point3(1, 0, tapeLength);
        worldSpaceArr[3] = new Point3(-1, 0, tapeLength);
        MatOfPoint3f worldSpacePts = new MatOfPoint3f(worldSpaceArr);

        rvec = new Mat();
        tvec = new Mat();
        Calib3d.solvePnP(worldSpacePts, line, camIntrinsics, new MatOfDouble(), rvec, tvec);

        // Shift the points up 2 inches to draw the 3D box
        Point3[] shiftedWorldSpaceArr = new Point3[4];
        shiftedWorldSpaceArr[0] = new Point3(-1, -2, 0);
        shiftedWorldSpaceArr[1] = new Point3(1, -2, 0);
        shiftedWorldSpaceArr[2] = new Point3(1, -2, tapeLength);
        shiftedWorldSpaceArr[3] = new Point3(-1, -2, tapeLength);
        MatOfPoint3f shiftedWorldSpacePts = new MatOfPoint3f(shiftedWorldSpaceArr);
        MatOfPoint2f shiftedImgPts = new MatOfPoint2f();
        Calib3d.projectPoints(shiftedWorldSpacePts, rvec, tvec, camIntrinsics, new MatOfDouble(), shiftedImgPts);

        drawBox(line, shiftedImgPts);
    }

    private void offsetAdjustment() {

        double camRotation = rvec.get(0, 0)[0];
        // Account for camera rotation, rotate counter-clockwise about x axis with
        // left-hand rule
        Mat rotationMat = Mat.zeros(3, 3, CvType.CV_64FC1);
        rotationMat.put(0, 0, 1);
        rotationMat.put(1, 1, Math.cos(camRotation));
        rotationMat.put(1, 2, Math.sin(camRotation));
        rotationMat.put(2, 1, -Math.sin(camRotation));
        rotationMat.put(2, 2, Math.cos(camRotation));
        Core.gemm(rotationMat, tvec, 1, new Mat(), 0, tvec);

        Core.add(tvec, camOffset, tvec);

        Core.multiply(rvec, new Scalar(180 / Math.PI), rvec);
    }

    private void setHeading() {
        double x = tvec.get(0, 0)[0];
        double z = tvec.get(2, 0)[0];

        angleToLine = Math.atan(x / z) * 180 / Math.PI;
        distanceToLine = Math.sqrt(x * x + z * z);
        distanceToLine /= 12;
        angleToWall = rvec.get(1, 0)[0];
    }

    private void drawBox(MatOfPoint2f imagePoints, MatOfPoint2f shiftedImagePoints) {
        Imgproc.drawContours(dst, Arrays.asList(new MatOfPoint(imagePoints.toArray())), -1, GREEN, 2);

        for (int i = 0; i < imagePoints.rows(); i++) {
            Imgproc.line(dst, new Point(imagePoints.get(i, 0)), new Point(shiftedImagePoints.get(i, 0)), BLUE, 2);
        }

        Imgproc.drawContours(dst, Arrays.asList(new MatOfPoint(shiftedImagePoints.toArray())), -1, MAGENTA, 2);

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

    public void setTapeLength(double inches) {
        tapeLength = inches;
    }

    public void setOffset(double x, double y, double z) {
        camOffset = Mat.zeros(3, 1, CvType.CV_64FC1);
        camOffset.put(0, 0, x);
        camOffset.put(1, 0, y);
        camOffset.put(2, 0, z - TAPE_DISTANCE_BUFFER);
    }

    public Mat getDst() {
        return dst;
    }

    public String getRotationVector() {
        String s = "<";
        for (int i = 0; i < rvec.rows(); i++) {
            if (i != 0) {
                s += ", ";
            }
            s += rvec.get(i, 0)[0];
        }
        s += ">";

        return s;
    }

    public String getTranslationVector() {
        String s = "<";
        for (int i = 0; i < rvec.rows(); i++) {
            if (i != 0) {
                s += ", ";
            }
            s += tvec.get(i, 0)[0];
        }
        s += ">";

        return s;
    }

    public double getAngleToLine() {
        return angleToLine;
    }

    public double getDistanceToLine() {
        return distanceToLine;
    }

    public double getAngleToWall() {
        return angleToWall;
    }
}
