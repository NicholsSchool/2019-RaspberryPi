import org.opencv.core.Mat;

import edu.wpi.first.vision.VisionPipeline;

/**
 * EmtpyPiple copies the src to dst without any modification.
 */
public class EmptyPipeline implements VisionPipeline {

    public Mat dst;

    @Override
    public void process(Mat src) {
        if (src.empty()) {
            return;
        }

        dst = src;
    }

}