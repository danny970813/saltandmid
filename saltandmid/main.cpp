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

	random_device rand;
	default_random_engine gen;//亂數產生器
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
	Mat img = imread("sun.jpg", 1);
	Mat ori;//儲存原圖用
	Mat result;//儲存結果用
	float percent;
	img.copyTo(ori);

	namedWindow("original");
	imshow("original", ori);

	cout << "輸入雜訊百分比(0~100):";
	cin >> percent;

	if (percent > 100 || percent < 0) {
		cout << "輸入錯誤，輸入範圍為0~100" << endl;
		return 0;
	}

	percent = percent / 100;

	salt(img, percent);//加入雜訊
	
	namedWindow("salt");
	imshow("salt", img);
	cout << "original and salt PSNR=" << psnr(ori, img) << endl;

	result = midfilter(img);//中值濾波

	namedWindow("result");
	imshow("result", result);
	cout << "original and result PSNR=" << psnr(ori, result) << endl;
	cout << "salt and result PSNR=" << psnr(img, result) << endl;

	waitKey(0);
	destroyAllWindows();

	return 0;
}