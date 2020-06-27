#include <iostream>
#include <opencv2/core/core.hpp>           
#include <opencv2/highgui/highgui.hpp>     
#include <opencv2/imgproc/imgproc.hpp> 
#include<random>
#include <algorithm>

using namespace cv;
using namespace std;

void salt(Mat img, float n)
{
	int x, y;

	random_device rand{};//亂數種子
	default_random_engine gen{rand()};//亂數產生器
	uniform_int_distribution<int> row(0, img.rows - 1);//row亂數分布
	uniform_int_distribution<int> col(0, img.cols - 1);//col亂數分布

	n = n * img.rows * img.cols;//雜訊數量

	for (int i = 0; i < n; i++)
	{
		x = row(gen);//產生雜訊座標
		y = col(gen);
		img.at<Vec3b>(x, y) = Vec3b(255, 255, 255);//將雜訊座標改成白色
	}
	
}

uchar sortandmid(uchar a, uchar b, uchar c, uchar d, uchar e, uchar f, uchar g, uchar h, uchar i) {

	uchar mask[9] = { a,b,c,d,e,f,g,h,i };
	sort(mask, mask + 9);

	return mask[4];
}

Mat midfilter(Mat src) {

	Mat img(src.rows + 2, src.cols + 2, CV_8UC3, Scalar(0));//產生比原圖多一圈邊緣的圖
	Mat temp(size(img), CV_8UC3);//儲存中值濾波後的圖
	Mat dst;//儲存中值濾波後去邊緣的圖
	Mat imgROI = img(Rect(1, 1, src.cols, src.rows));
	addWeighted(src, 1, imgROI, 0, 0, imgROI);//將原圖貼到img中央空出邊緣

	img.at<Vec3b>(0, 0) = img.at<Vec3b>(1, 1);//將補的邊緣做鏡射
	img.at<Vec3b>(0, img.cols - 1) = img.at<Vec3b>(1, img.cols - 2);
	img.at<Vec3b>(img.rows - 1, 0) = img.at<Vec3b>(img.rows - 2, 1);
	img.at<Vec3b>(img.rows - 1, img.cols - 1) = img.at<Vec3b>(img.rows - 2, img.cols - 2);

	for (int i = 1; i < img.cols - 1; i++) img.at<Vec3b>(0, i) = img.at<Vec3b>(1, i);
	for (int i = 1; i < img.rows - 1; i++) img.at<Vec3b>(i, img.cols - 1) = img.at<Vec3b>(i, img.cols - 2);
	for (int i = 1; i < img.cols - 1; i++) img.at<Vec3b>(img.rows - 1, i) = img.at<Vec3b>(img.rows - 2, i);
	for (int i = 1; i < img.rows - 1; i++) img.at<Vec3b>(i, 0) = img.at<Vec3b>(i, 1);

	for (int i = 0; i < img.rows; i++) {//對img做中值濾波，存在temp
		for (int j = 0; j < img.cols; j++) {
			if (i == 0 || j == 0 || i == img.rows - 1 || j == img.cols - 1)//補的邊不動
				temp.at<Vec3b>(i, j) = img.at<Vec3b>(i, j);
			else {//原圖範圍取中值
				temp.at<Vec3b>(i, j)[0] = sortandmid(img.at<Vec3b>(i - 1, j - 1)[0], img.at<Vec3b>(i - 1, j)[0], img.at<Vec3b>(i - 1, j + 1)[0], img.at<Vec3b>(i, j - 1)[0], img.at<Vec3b>(i, j)[0], img.at<Vec3b>(i, j + 1)[0], img.at<Vec3b>(i + 1, j - 1)[0], img.at<Vec3b>(i + 1, j)[0], img.at<Vec3b>(i + 1, j + 1)[0]);
				temp.at<Vec3b>(i, j)[1] = sortandmid(img.at<Vec3b>(i - 1, j - 1)[1], img.at<Vec3b>(i - 1, j)[1], img.at<Vec3b>(i - 1, j + 1)[1], img.at<Vec3b>(i, j - 1)[1], img.at<Vec3b>(i, j)[1], img.at<Vec3b>(i, j + 1)[1], img.at<Vec3b>(i + 1, j - 1)[1], img.at<Vec3b>(i + 1, j)[1], img.at<Vec3b>(i + 1, j + 1)[1]);
				temp.at<Vec3b>(i, j)[2] = sortandmid(img.at<Vec3b>(i - 1, j - 1)[2], img.at<Vec3b>(i - 1, j)[2], img.at<Vec3b>(i - 1, j + 1)[2], img.at<Vec3b>(i, j - 1)[2], img.at<Vec3b>(i, j)[2], img.at<Vec3b>(i, j + 1)[2], img.at<Vec3b>(i + 1, j - 1)[2], img.at<Vec3b>(i + 1, j)[2], img.at<Vec3b>(i + 1, j + 1)[2]);
			}
		}
	}

	dst = temp(Rect(1, 1, src.cols, src.rows));//去除之前補的邊緣
	return dst;
}

double psnr(Mat img1, Mat img2) {//MSE=(sigma(|(img1-img2)|^2))/row*col, PSNR=10*log(maxI^2/MSE)

	Mat temp;
	absdiff(img1, img2, temp);//|(img1-img2)|
	temp.convertTo(temp, CV_32F);//轉成32bit避免平方溢位
	temp = temp.mul(temp);//平方
	Scalar s = sum(temp);//加總
	double MSE = (s.val[0] + s.val[1] + s.val[2]) / (double)(temp.rows * temp.cols * 3);
	double psnr = 10 * log((255 * 255) / MSE);

	return  psnr;
}

int main()
{
	Mat ori = imread("tree.jpg", 1);
	Mat img10, img20, img30;

	ori.copyTo(img10);
	ori.copyTo(img20);
	ori.copyTo(img30);

	namedWindow("original");
	imshow("original", ori);
	

	salt(img10, 0.1);//加入雜訊
	salt(img20, 0.2);
	salt(img30, 0.3);

	img10 = midfilter(img10);//中值濾波
	img20 = midfilter(img20);
	img30 = midfilter(img30);

	namedWindow("result10%");
	imshow("result10%", img10);

	namedWindow("result20%");
	imshow("result20%", img20);

	namedWindow("result30%");
	imshow("result30%", img30);

	cout << "original and salt10% PSNR=" << psnr(ori, img10) << endl;
	cout << "original and salt20% PSNR=" << psnr(ori, img20) << endl;
	cout << "original and salt30% PSNR=" << psnr(ori, img30) << endl;

	waitKey(0);
	destroyAllWindows();

	return 0;
}