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

    private static final double FOCAL_LENGTH = 400; // In pixels, needs tuning if res is changed

    public double cameraAngleOffset;
    public double cameraXOffset;
    public double cameraYOffset;
    public double cameraZOffset;

    public Mat dst;
    public double angleToLine;
    public double distanceToLine;
    public double angleToWall;

    public double lineX;
    public double lineY;
    public double lineZ;

    @Override
    public void process(Mat src) {
        if (src.empty()) {
            return;
        }

        ArrayList<MatOfPoint> contours = getContours(src);

        MatOfPoint2f line = getLine(contours);

        if (line == null) {
            return;
        }

        Mat[] vecs = getTranslation(line);

        Mat[] pos = camPosToRobotPos(vecs);
        setHeading(pos);
    }

    // Get the contours of bright objects
    private ArrayList<MatOfPoint> getContours(Mat src) {
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

        return contours;
    }

    private MatOfPoint2f getLine(ArrayList<MatOfPoint> contours) {
        MatOfPoint2f line = null;
        double closest = 1;

        // Approximate contours with polygons
        for (MatOfPoint contour : contours) {
            // Only include contours larger than 1/300 of the screen
            if (Imgproc.contourArea(contour) > dst.width() * dst.height() / 300) {
                // Convert format
                MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());

                // Epsilon will be 2% of the perimeter, a lower epsilon will result in more
                // vertices
                double epsilon = 0.015 * Imgproc.arcLength(contour2f, true);

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

        return line;

    }

    /*
    @Deprecated
    private Mat[] getVectors(MatOfPoint2f line) {
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
        Point3[] botWorldSpaceArr = new Point3[4];
        botWorldSpaceArr[0] = new Point3(-1, 0, 0);
        botWorldSpaceArr[1] = new Point3(1, 0, 0);
        botWorldSpaceArr[2] = new Point3(1, 0, 18);
        botWorldSpaceArr[3] = new Point3(-1, 0, 18);
        MatOfPoint3f botWorldSpacePts = new MatOfPoint3f(botWorldSpaceArr);

        Mat botRotationVector = new Mat();
        Mat botTranslationVector = new Mat();
        Calib3d.solvePnP(botWorldSpacePts, line, camIntrinsics, new MatOfDouble(), botRotationVector,
                botTranslationVector);

        // Shift the points up 2 inches to draw the 3D box
        Point3[] shiftedBotWorldSpaceArr = new Point3[4];
        shiftedBotWorldSpaceArr[0] = new Point3(-1, -2, 0);
        shiftedBotWorldSpaceArr[1] = new Point3(1, -2, 0);
        shiftedBotWorldSpaceArr[2] = new Point3(1, -2, 18);
        shiftedBotWorldSpaceArr[3] = new Point3(-1, -2, 18);
        MatOfPoint3f shiftedBotWorldSpacePts = new MatOfPoint3f(shiftedBotWorldSpaceArr);
        MatOfPoint2f shiftedImgPts = new MatOfPoint2f();
        Calib3d.projectPoints(shiftedBotWorldSpacePts, botRotationVector, botTranslationVector, camIntrinsics,
                new MatOfDouble(), shiftedImgPts);

        drawBox(line, shiftedImgPts);

        // Top of the line as (0, 0, 0)
        Point3[] topWorldSpaceArr = new Point3[4];
        topWorldSpaceArr[0] = new Point3(-1, 0, -18);
        topWorldSpaceArr[1] = new Point3(1, 0, -18);
        topWorldSpaceArr[2] = new Point3(1, 0, 0);
        topWorldSpaceArr[3] = new Point3(-1, 0, 0);
        MatOfPoint3f topWorldSpacePts = new MatOfPoint3f(topWorldSpaceArr);

        Mat topRotationVector = new Mat();
        Mat topTranslationVector = new Mat();
        Calib3d.solvePnP(topWorldSpacePts, line, camIntrinsics, new MatOfDouble(), topRotationVector,
                topTranslationVector);

        return new Mat[] { botRotationVector, botTranslationVector, topRotationVector, topTranslationVector };
    }
    */

    private Mat[] getTranslation(MatOfPoint2f line) {
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
        Point3[] botWorldSpaceArr = new Point3[4];
        botWorldSpaceArr[0] = new Point3(-1, 0, 0);
        botWorldSpaceArr[1] = new Point3(1, 0, 0);
        botWorldSpaceArr[2] = new Point3(1, 0, 18);
        botWorldSpaceArr[3] = new Point3(-1, 0, 18);
        MatOfPoint3f botWorldSpacePts = new MatOfPoint3f(botWorldSpaceArr);

        Mat botRotationVector = new Mat();
        Mat botTranslationVector = new Mat();
        Calib3d.solvePnP(botWorldSpacePts, line, camIntrinsics, new MatOfDouble(), botRotationVector,
                botTranslationVector);

        // Shift the points up 2 inches to draw the 3D box
        Point3[] shiftedBotWorldSpaceArr = new Point3[4];
        shiftedBotWorldSpaceArr[0] = new Point3(-1, -2, 0);
        shiftedBotWorldSpaceArr[1] = new Point3(1, -2, 0);
        shiftedBotWorldSpaceArr[2] = new Point3(1, -2, 18);
        shiftedBotWorldSpaceArr[3] = new Point3(-1, -2, 18);
        MatOfPoint3f shiftedBotWorldSpacePts = new MatOfPoint3f(shiftedBotWorldSpaceArr);
        MatOfPoint2f shiftedImgPts = new MatOfPoint2f();
        Calib3d.projectPoints(shiftedBotWorldSpacePts, botRotationVector, botTranslationVector, camIntrinsics,
                new MatOfDouble(), shiftedImgPts);

        drawBox(line, shiftedImgPts);

        // Top of the line as (0, 0, 0)
        Point3[] topWorldSpaceArr = new Point3[4];
        topWorldSpaceArr[0] = new Point3(-1, 0, -18);
        topWorldSpaceArr[1] = new Point3(1, 0, -18);
        topWorldSpaceArr[2] = new Point3(1, 0, 0);
        topWorldSpaceArr[3] = new Point3(-1, 0, 0);
        MatOfPoint3f topWorldSpacePts = new MatOfPoint3f(topWorldSpaceArr);

        Mat topRotationVector = new Mat();
        Mat topTranslationVector = new Mat();
        Calib3d.solvePnP(topWorldSpacePts, line, camIntrinsics, new MatOfDouble(), topRotationVector,
                topTranslationVector);

        return new Mat[] { botTranslationVector, topTranslationVector };
    }

    /*
    // Coordinates of the line relative to camera
    @Deprecated
    private Mat[] getRelativePos(Mat[] vecs) {
        Mat camOffset = Mat.zeros(3, 1, CvType.CV_64FC1);
        camOffset.put(0, 0, cameraXOffset);
        camOffset.put(1, 0, cameraYOffset);
        camOffset.put(2, 0, cameraZOffset);

        Mat lineBottomPos = vectorsToPos(vecs[0], vecs[1], cameraAngleOffset, camOffset);
        Mat lineTopPos = vectorsToPos(vecs[2], vecs[3], cameraAngleOffset, camOffset);

        return new Mat[] { lineBottomPos, lineTopPos };
    }
    */

    private Mat[] camPosToRobotPos(Mat[] pos) {
        Mat camOffset = Mat.zeros(3, 1, CvType.CV_64FC1);
        camOffset.put(0, 0, cameraXOffset);
        camOffset.put(1, 0, cameraYOffset);
        camOffset.put(2, 0, cameraZOffset);

        Mat lineBottomPos = camPosToRobotPos(pos[0], cameraAngleOffset, camOffset);
        Mat lineTopPos = camPosToRobotPos(pos[1], cameraAngleOffset, camOffset);

        return new Mat[] { lineBottomPos, lineTopPos };
    }

    private void setHeading(Mat[] pos) {
        double botX = pos[0].get(0, 0)[0];
        double botZ = pos[0].get(2, 0)[0];
        double topX = pos[1].get(0, 0)[0];
        double topZ = pos[1].get(2, 0)[0];

        angleToLine = Math.atan(botX / botZ) * 180 / Math.PI;
        distanceToLine = Math.sqrt(botX * botX + botZ * botZ);
        distanceToLine /= 12;
        angleToWall = Math.atan((topX - botX) / (topZ - botZ)) * 180 / Math.PI;

        lineX = botX;
        lineY = pos[0].get(1, 0)[0];
        lineZ = botZ;
    }

    private void drawBox(MatOfPoint2f imagePoints, MatOfPoint2f shiftedImagePoints) {
        Imgproc.drawContours(dst, Arrays.asList(new MatOfPoint(imagePoints.toArray())), -1, GREEN, 2);

        for (int i = 0; i < imagePoints.rows(); i++) {
            Imgproc.line(dst, new Point(imagePoints.get(i, 0)), new Point(shiftedImagePoints.get(i, 0)), BLUE, 2);
        }

        Imgproc.drawContours(dst, Arrays.asList(new MatOfPoint(shiftedImagePoints.toArray())), -1, MAGENTA, 2);

    }

    /*
    // Convert rotation and translation vectors to relative x, y, z position
    @Deprecated
    private Mat vectorsToPos(Mat rvec, Mat tvec, double camRot, Mat camOffset) {
        Mat rmat = new Mat();
        // Convert 3x1 rotation vector to 3x3 rotation matrix
        Calib3d.Rodrigues(rvec, rmat);
        // Transpose = inverse for rotation matrices
        Core.transpose(rmat, rmat);

        // Get line position relative to the camera by reversing transformation
        Mat lineRelativePos = new Mat();
        Core.multiply(rmat, new Scalar(-1), rmat);
        // Use gemm() instead of multiply() for matrices of different dimensions
        Core.gemm(rmat, tvec, 1, new Mat(), 0, lineRelativePos);
        Core.multiply(lineRelativePos, new Scalar(-1), lineRelativePos);

        // Account for camera rotation, reverse rotate about x axis with left-hand rule
        Mat camRotMat = Mat.zeros(3, 3, CvType.CV_64FC1);
        camRotMat.put(0, 0, 1);
        camRotMat.put(1, 1, Math.cos(camRot));
        camRotMat.put(1, 2, Math.sin(camRot));
        camRotMat.put(2, 1, -Math.sin(camRot));
        camRotMat.put(2, 2, Math.cos(camRot));

        // Line position relative to center of robot
        Mat lineWorldPos = new Mat();
        Core.gemm(camRotMat, lineRelativePos, 1, new Mat(), 0, lineWorldPos);
        Core.add(lineWorldPos, camOffset, lineWorldPos);

        return lineWorldPos;
    }
    */

    private Mat camPosToRobotPos(Mat pos, double camRot, Mat camOffset) {
        // Account for camera rotation, reverse rotate about x axis with left-hand rule
        Mat camRotMat = Mat.zeros(3, 3, CvType.CV_64FC1);
        camRotMat.put(0, 0, 1);
        camRotMat.put(1, 1, Math.cos(camRot));
        camRotMat.put(1, 2, Math.sin(camRot));
        camRotMat.put(2, 1, -Math.sin(camRot));
        camRotMat.put(2, 2, Math.cos(camRot));

        // Line position relative to center of robot
        Mat lineWorldPos = new Mat();
        Core.gemm(camRotMat, pos, 1, new Mat(), 0, lineWorldPos);
        Core.add(lineWorldPos, camOffset, lineWorldPos);

        return lineWorldPos;
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
