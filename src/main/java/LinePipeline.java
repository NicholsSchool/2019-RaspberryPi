import java.util.ArrayList;
import java.util.Arrays;

import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import edu.wpi.first.vision.VisionPipeline;

/**
 * LinePipline gets the heading of alignment lines using contour analysis.
 */
public class LinePipeline implements VisionPipeline {

    private static final int THRESHOLD = 160;
    private static final Scalar BLUE = new Scalar(255, 0, 0), GREEN = new Scalar(0, 255, 0),
            RED = new Scalar(0, 0, 255), YELLOW = new Scalar(0, 255, 255), ORANGE = new Scalar(0, 165, 255);
    private static final int HORIZONTAL_FOV = 57;
    private static final double TAPE_WIDTH = 2 / 12d; // in feet
    private static final double CAMERA_HEIGHT = 21 / 12d; // in feet

    public Mat dst;

    public double angleToLine;
    public double distanceToLine;
    public double angleToWall;

    @Override
    public void process(Mat src) {
        if (src.empty()) {
            return;
        }

        ArrayList<MatOfPoint> contours = getLines(src);
        MatOfPoint realLine = getRealLine(contours);

        setHeading(realLine);
    }

    private ArrayList<MatOfPoint> getLines(Mat src) {
        dst = new Mat();

        // Extract whites
        Core.inRange(src, new Scalar(THRESHOLD, THRESHOLD, THRESHOLD), new Scalar(255, 255, 255), dst);

        // Find all external contours
        ArrayList<MatOfPoint> contours = new ArrayList<>();
        try {
            Imgproc.findContours(dst, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        } catch (CvException e) {
            // Sometimes the Mat format gets messed up when switching cameras
            System.out.println(e.getMessage());
        }

        dst = src;

        return contours;
    }

    private MatOfPoint getRealLine(ArrayList<MatOfPoint> contours) {
        MatOfPoint realLine = null;
        // RotatedRect realLineRect = null;
        double closest = 1;

        // Approximate contours with polygons
        for (MatOfPoint contour : contours) {
            // Only include contours larger than 1/200 of the screen
            if (Imgproc.contourArea(contour) > dst.width() * dst.height() / 200) {
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
                    double distanceToCenter = Math.abs(rect.center.x / Main.CAMERA_RESOLUTION_X - 0.5);

                    // The "real" line will be the rectangle greater than a certain length to width
                    // ratio that is closest to the center of the screen
                    if (ratio > 2 && distanceToCenter < closest) {
                        realLine = contour;
                        // realLineRect = rect;
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

        return realLine;
    }

    private void setHeading(MatOfPoint line) {
        if (line == null) {
            return;
        }
        // Draw the contour
        Imgproc.drawContours(dst, Arrays.asList(line), -1, GREEN, 2);

        // Get the best fit line
        Mat fit = new Mat();
        Imgproc.fitLine(line, fit, Imgproc.DIST_L2, 0, 0.01, 0.01);
        double vx = fit.get(0, 0)[0];
        double vy = fit.get(1, 0)[0];
        // Change the vector so that it is facing upwards
        if (vy > 0) {
            vx *= -1;
            vy *= -1;
        }

        // Get the rotation of the line which is the angle to the wall
        double angle = Math.atan(-vy / vx);
        // Convert cartesian angle to angle from vertical
        if (angle < 0) {
            angle += Math.PI;
        }
        angleToWall = -(angle * 180 / Math.PI - 90);

        // Get the bottom two vertices
        double[] lowest = { 0, 0 };
        double[] second = { 0, 0 };
        for (int i = 0; i < line.rows(); i++) {
            double[] p = line.get(i, 0);
            if (p[1] > lowest[1]) {
                second = lowest;
                lowest = p;
            } else if (p[1] > second[1]) {
                second = p;
            }
        }

        // Get the midpoint of the bottom two vertices
        double x = (lowest[0] + second[0]) / 2;
        double y = (lowest[1] + second[1]) / 2;

        // The angle to the lline is the distance from the screen center multiplied by
        // the camera FOV
        angleToLine = (x / Main.CAMERA_RESOLUTION_X - 0.5) * HORIZONTAL_FOV;

        // Get the distance using the distance between the bottom vertices as the line
        // width
        double deltaX = lowest[0] - second[0];
        double deltaY = lowest[1] - second[1];
        double width = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
        double angularWidth = width / dst.width() * HORIZONTAL_FOV;
        distanceToLine = TAPE_WIDTH / Math.tan(angularWidth * Math.PI / 180);
        distanceToLine = Math.sqrt(distanceToLine * distanceToLine - CAMERA_HEIGHT * CAMERA_HEIGHT);

        // Get the distance using the width of the bounding rect as the line width
        // double width = Math.min(realLineRect.size.height, realLineRect.size.width);
        // double angularWidth = width / dst.width() * HORIZONTAL_FOV;
        // distanceToLine = TAPE_WIDTH / Math.tan(angularWidth * Math.PI / 180);
        // distanceToLine = Math.sqrt(distanceToLine * distanceToLine - CAMERA_HEIGHT *
        // CAMERA_HEIGHT);

        // Draw the best fit line
        Imgproc.line(dst, new Point(x, y), new Point(x + vx * 100, y + vy * 100), BLUE, 1);

        // Draw the bounding rect
        // Point[] vertices = new Point[4];
        // realLineRect.points(vertices);
        // for (int i = 0; i < vertices.length; i++) {
        // Imgproc.line(dst, vertices[i], vertices[(i + 1) % 4], YELLOW, 1);
        // }

    }

}
