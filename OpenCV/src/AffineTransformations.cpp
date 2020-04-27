// https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/warp_affine/warp_affine.html
#include <cstdio>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

char *source_window = "Source image";
char *warp_window = "Warp";
char *warp_rotate_window = "Warp + Roatate";

int main(int argc, char **argv)
{
  Point2f srcTri[3];
  Point2f dstTri[3];

  Mat rot_mat(2, 3, CV_32FC1);
  Mat warp_mat(2, 3, CV_32FC1);
  Mat src, warp_dst, warp_rotate_dst;

  src = imread(argv[1], 1);

  // Set the dst image the same type and size as src
  warp_dst = Mat::zeros(src.rows, src.cols, src.type());

  // Set your 3 points to calculate the Affine Transform
  srcTri[0] = Point2f(0, 0);
  srcTri[1] = Point2f(src.cols - 1, 0);
  srcTri[2] = Point2f(0, src.rows - 1);

  // Set your 3 points to calculate the Affine Transform
  dstTri[0] = Point2f(src.cols * 0.0, src.rows * 0.33);
  dstTri[1] = Point2f(src.cols * 0.85, src.rows * 0.25);
  dstTri[2] = Point2f(src.cols * 0.15, src.rows * 0.7);

  warp_mat = getAffineTransform(srcTri, dstTri);

  cout << "Num rows: " << src.rows << '\n';
  cout << "Num cols: " << src.cols << '\n';
  cout << warp_mat << '\n';

  // Apply the Affine Transform just found the src image
  warpAffine(src, warp_dst, warp_mat, warp_dst.size());

  // Compute a rotation matrix with respect to the center of the image
  Point center = Point(warp_dst.cols / 2, warp_dst.rows / 2);
  double angle = -50.0;
  double scale = 0.6;

  rot_mat = getRotationMatrix2D(center, angle, scale);

  warpAffine(warp_dst, warp_rotate_dst, rot_mat, warp_dst.size());


  namedWindow(source_window, WINDOW_AUTOSIZE);
  imshow(source_window, src);

  namedWindow(warp_window, WINDOW_AUTOSIZE);
  imshow(warp_window, warp_dst);

  namedWindow(warp_rotate_window, WINDOW_AUTOSIZE);
  imshow(warp_rotate_window, warp_rotate_dst);

  waitKey(0);
  return 0;
}