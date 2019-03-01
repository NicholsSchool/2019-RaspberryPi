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
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;

import edu.wpi.first.vision.VisionPipeline;

/**
 * LinePipline gets the heading of alignment lines using contour analysis and
 * 3D-2D point correspondence.
 */
public class RetroPipeline implements VisionPipeline {

    @SuppressWarnings("unused")
    private static final Scalar BLUE = new Scalar(255, 0, 0), GREEN = new Scalar(0, 255, 0),
            RED = new Scalar(0, 0, 255), YELLOW = new Scalar(0, 255, 255), ORANGE = new Scalar(0, 165, 255),
            MAGENTA = new Scalar(255, 0, 255);

    private static final int THRESHOLD = 40;

    private static final double FOCAL_LENGTH = 320; // In pixels, needs tuning if res is changed
    private static final int DISTANCE_BUFFER = 36; // Distance padding from tip of tape

    private Mat dst;
    private Mat bitmask;

    private Mat camOffset;

    private RotatedRect leftRect;
    private RotatedRect rightRect;

    private MatOfPoint2f target;

    private Mat rvec;
    private Mat tvec;

    private double angleToTarget;
    private double distanceToTarget;
    private double angleToWall;

    public RetroPipeline(double xOffset, double yOffset, double zOffset) {
        camOffset = Mat.zeros(3, 1, CvType.CV_64FC1);
        camOffset.put(0, 0, xOffset);
        camOffset.put(1, 0, yOffset);
        camOffset.put(2, 0, zOffset - DISTANCE_BUFFER);
    }

    @Override
    public void process(Mat src) {
        if (src.empty()) {
            return;
        }

        getRects(src);

        if (leftRect == null || rightRect == null) {
            return;
        }

        getTarget();

        getVectors();

        offsetAdjustment();

        setHeading();
    }

    // Get the contours of bright objects
    private void getRects(Mat src) {
        // dst = src;
        dst = new Mat();
        src.copyTo(dst);
        bitmask = new Mat();

        // Extract whites
        Core.inRange(src, new Scalar(0, THRESHOLD, 0), new Scalar(255 - THRESHOLD, 255, 255 - THRESHOLD), bitmask);

        // Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2HSV);
        // Core.inRange(src, new Scalar(0, 0, 215), new Scalar(255, 40, 255), bitmask);
        // Imgproc.cvtColor(src, src, Imgproc.COLOR_HSV2BGR);

        // Find all external contours
        ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        try {
            Imgproc.findContours(bitmask, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        } catch (CvException e) {
            // Sometimes the Mat format gets messed up when switching cameras
            System.out.println(e.getMessage());
        }

        ArrayList<RotatedRect> leftTargets = new ArrayList<RotatedRect>();
        ArrayList<RotatedRect> rightTargets = new ArrayList<RotatedRect>();

        for (MatOfPoint contour : contours) {
            // Only include contours larger than 1/400 of the screen
            if (Imgproc.contourArea(contour) < dst.width() * dst.height() / 400) {
                // Draw the contour only
                Imgproc.drawContours(dst, Arrays.asList(contour), -1, RED, 1);
            } else {
                Imgproc.drawContours(dst, Arrays.asList(contour), -1, ORANGE, 1);

                // Get the bounding rect
                RotatedRect rect = Imgproc.minAreaRect(new MatOfPoint2f(contour.toArray()));
                double h = rect.size.height;
                double w = rect.size.width;
                double ratio = Math.max(h, w) / Math.min(h, w);

                // The "real" line will be the rectangle greater than a certain length to width
                // ratio that is closest to the center of the screen
                if (ratio > 2) {
                    if (rect.angle < -65 && rect.angle > -85) {
                        leftTargets.add(rect);
                        drawRect(rect, YELLOW);
                    } else if (rect.angle < -5 && rect.angle > -25) {
                        rightTargets.add(rect);
                        drawRect(rect, YELLOW);
                    }
                }
            }
        }

        leftRect = null;
        rightRect = null;

        double closest = 1;
        for (RotatedRect rect : leftTargets) {
            // Distance to center of the screen as a percentage
            double distanceToCenter = Math.abs(rect.center.x / dst.width() - 0.5);

            // The most likely candidate is closest to the center of the screen
            if (distanceToCenter < closest) {
                leftRect = rect;
                closest = distanceToCenter;
            }
        }

        if (leftRect == null) {
            return;
        }

        closest = 1;
        for (RotatedRect rect : rightTargets) {
            if (rect.center.x < leftRect.center.x) {
                continue;
            }

            // Distance to the left rect as a percentage
            double distanceToCenter = Math.abs((rect.center.x - leftRect.center.x) / dst.width());

            // The most likely candidate is closest to the left rect
            if (distanceToCenter < closest) {
                rightRect = rect;
                closest = distanceToCenter;
            }
        }
    }

    private void drawRect(RotatedRect rect, Scalar color) {
        Point[] pts = new Point[4];
        rect.points(pts);

        for (int i = 0; i < pts.length; i++) {
            Imgproc.line(dst, pts[i], pts[(i + 1) % 4], color, 1);
        }
    }

    private void getTarget() {
        // Get the top and side points of the two rects
        Point[] leftPts = new Point[4];
        leftRect.points(leftPts);

        Point topLeft = leftPts[0];
        Point leftMost = leftPts[0];
        for (Point p : leftPts) {
            if (p.y < topLeft.y) {
                topLeft = p;
            }
            if (p.x < leftMost.x) {
                leftMost = p;
            }
        }

        Point[] rightPts = new Point[4];
        rightRect.points(rightPts);

        Point topRight = rightPts[0];
        Point rightMost = rightPts[0];
        for (Point p : rightPts) {
            if (p.y < topRight.y) {
                topRight = p;
            }
            if (p.x > rightMost.x) {
                rightMost = p;
            }
        }

        target = new MatOfPoint2f(new Point[] { leftMost, topLeft, topRight, rightMost });

        // Get subpixel locations of the corners
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.COUNT, 40, 0.001);
        Imgproc.cornerSubPix(bitmask, target, new Size(5, 5), new Size(-1, -1), criteria);
    }

    private void getVectors() {
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
        worldSpaceArr[0] = new Point3(-7.313, -0.5008, 0);
        worldSpaceArr[1] = new Point3(-5.9363, -5.826, 0);
        worldSpaceArr[2] = new Point3(5.9363, -5.826, 0);
        worldSpaceArr[3] = new Point3(7.313, -0.5008, 0);
        MatOfPoint3f worldSpacePts = new MatOfPoint3f(worldSpaceArr);

        rvec = new Mat();
        tvec = new Mat();
        Calib3d.solvePnP(worldSpacePts, target, camIntrinsics, new MatOfDouble(), rvec, tvec);

        // 3D box with the corners on the outside of the target
        Point3[] boxBottomWorldSpaceArr = new Point3[4];
        boxBottomWorldSpaceArr[0] = new Point3(-7.313, 0, 0);
        boxBottomWorldSpaceArr[1] = new Point3(-7.313, -5.826, 0);
        boxBottomWorldSpaceArr[2] = new Point3(7.313, -5.826, 0);
        boxBottomWorldSpaceArr[3] = new Point3(7.313, 0, 0);
        MatOfPoint3f boxBottomWorldSpacePts = new MatOfPoint3f(boxBottomWorldSpaceArr);
        MatOfPoint2f boxBottomImgPts = new MatOfPoint2f();
        Calib3d.projectPoints(boxBottomWorldSpacePts, rvec, tvec, camIntrinsics, new MatOfDouble(), boxBottomImgPts);

        Point3[] boxTopWorldSpaceArr = new Point3[4];
        boxTopWorldSpaceArr[0] = new Point3(-7.313, 0, -3);
        boxTopWorldSpaceArr[1] = new Point3(-7.313, -5.826, -3);
        boxTopWorldSpaceArr[2] = new Point3(7.313, -5.826, -3);
        boxTopWorldSpaceArr[3] = new Point3(7.313, 0, -3);
        MatOfPoint3f boxTopWorldSpacePts = new MatOfPoint3f(boxTopWorldSpaceArr);
        MatOfPoint2f boxTopImgPts = new MatOfPoint2f();
        Calib3d.projectPoints(boxTopWorldSpacePts, rvec, tvec, camIntrinsics, new MatOfDouble(), boxTopImgPts);

        drawBox(boxBottomImgPts, boxTopImgPts);
    }

    private void offsetAdjustment() {
        // Account for X and Z axis camera rotation offset, assuming the Y axis rotation
        // of the camera is
        // lined up correctly with the robot, Z axis rotation offset should be near-zero
        // too
        // We will use the Y angle to line up with the wall later
        Mat rotationMat = new Mat();
        rvec.copyTo(rotationMat);
        rotationMat.put(1, 0, 0);
        Calib3d.Rodrigues(rotationMat, rotationMat);
        Core.transpose(rotationMat, rotationMat);
        Core.gemm(rotationMat, tvec, 1, new Mat(), 0, tvec);

        Core.add(tvec, camOffset, tvec);

        Core.multiply(rvec, new Scalar(180 / Math.PI), rvec);
    }

    private void setHeading() {
        double x = tvec.get(0, 0)[0];
        double z = tvec.get(2, 0)[0];

        angleToTarget = Math.atan(x / z) * 180 / Math.PI;
        distanceToTarget = Math.hypot(x, z);
        distanceToTarget /= 12;
        // Angle to wall is the Y rotation of the camera to the target
        angleToWall = rvec.get(1, 0)[0];
    }

    private void drawBox(MatOfPoint2f imagePoints, MatOfPoint2f shiftedImagePoints) {
        Imgproc.drawContours(dst, Arrays.asList(new MatOfPoint(imagePoints.toArray())), -1, GREEN, 2);

        for (int i = 0; i < imagePoints.rows(); i++) {
            Imgproc.line(dst, new Point(imagePoints.get(i, 0)), new Point(shiftedImagePoints.get(i, 0)), BLUE, 2);
        }

        Imgproc.drawContours(dst, Arrays.asList(new MatOfPoint(shiftedImagePoints.toArray())), -1, MAGENTA, 2);

    }

    public void setOffset(double x, double y, double z) {
        camOffset = Mat.zeros(3, 1, CvType.CV_64FC1);
        camOffset.put(0, 0, x);
        camOffset.put(1, 0, y);
        camOffset.put(2, 0, z - DISTANCE_BUFFER);
    }

    public Mat getDst() {
        return dst;
    }

    public String getRotationVector() {
        String s = "<";
        if (rvec != null) {
            for (int i = 0; i < rvec.rows(); i++) {
                if (i != 0) {
                    s += ", ";
                }
                s += rvec.get(i, 0)[0];
            }
        }
        s += ">";

        return s;
    }

    public String getTranslationVector() {
        String s = "<";
        if (tvec != null) {
            for (int i = 0; i < tvec.rows(); i++) {
                if (i != 0) {
                    s += ", ";
                }
                s += tvec.get(i, 0)[0];
            }
        }
        s += ">";

        return s;
    }

    public double getAngleToTarget() {
        return angleToTarget;
    }

    public double getDistanceToTarget() {
        return distanceToTarget;
    }

    public double getAngleToWall() {
        return angleToWall;
    }
}
