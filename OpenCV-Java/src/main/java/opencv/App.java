package opencv;

import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

import java.util.List;

/**
 * Hello world!
 *
 */
class DetectFaceDemo {
    public void run() {
        System.out.println("\nRunning DetectFaceDemo");
        // Create a face detector from the cascade file in the resources
        // directory.
        CascadeClassifier faceDetector = new CascadeClassifier(getClass().getResource("/lbpcascade_frontalface.xml").getPath());
        CascadeClassifier eyeDetector = new CascadeClassifier(getClass().getResource("/haarcascade_eye.xml").getPath());
        CascadeClassifier smileDetector = new CascadeClassifier(getClass().getResource("/haarcascade_smile.xml").getPath());
        Mat image = Imgcodecs.imread(getClass().getResource("/lena.png").getPath());
        // -- Detect faces
        MatOfRect faces = new MatOfRect();
        faceDetector.detectMultiScale(image, faces);
        List<Rect> listOfFaces = faces.toList();
        for (Rect face : listOfFaces) {
            Point center = new Point(face.x + face.width / 2, face.y + face.height / 2);
            Imgproc.ellipse(image, center, new Size(face.width / 2, face.height / 2), 0, 0, 360,
                    new Scalar(255, 0, 255));
            Mat faceROI = image.submat(face);
            // -- In each face, detect eyes
            MatOfRect eyes = new MatOfRect();
            eyeDetector.detectMultiScale(faceROI, eyes);
            List<Rect> listOfEyes = eyes.toList();
            for (Rect eye : listOfEyes) {
                Point eyeCenter = new Point(face.x + eye.x + eye.width / 2, face.y + eye.y + eye.height / 2);
                int radius = (int) Math.round((eye.width + eye.height) * 0.25);
                Imgproc.circle(image, eyeCenter, radius, new Scalar(255, 0, 0), 4);
            }

            // -- In each face, detect smile
            MatOfRect smile = new MatOfRect();
            smileDetector.detectMultiScale(faceROI, smile);
            List<Rect> listOfSmile = smile.toList();
            for (Rect rect : listOfSmile) {
                Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));
			}
        }
        // Save the visualized detection.
        String filename = "faceDetection.png";
        System.out.println(String.format("Writing %s", filename));
        Imgcodecs.imwrite(filename, image);
    }
}

public class App {
    public static void main(String[] args) {
        System.out.println("Hello, OpenCV");
        // Load the native library.
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new DetectFaceDemo().run();
    }
}