#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

#ifdef _DEBUG
#pragma comment(lib, "opencv_world4110d")
#else
#pragma comment(lib, "opencv_world4110")
#endif

// 잡음 판단
bool IsNoise(uchar val, int tolerance = 2) 
{
    return val <= tolerance || val >= 255 - tolerance;
}

// 거리 기반 공간 가중치 함수
// - 중심 픽셀에서의 거리(dx, dy)가 멀수록 가중치가 작아짐
// - alpha: 감쇠율
// - w(x,y) = exp(-α*(x² + y²))
float SpatialWeight(int dx, int dy, float alpha = 1.0f) 
{
    return expf(-alpha * (dx * dx + dy * dy));
}

// PFA 기반 노이즈 제거 필터
Mat PfaDenoise(const Mat& input, int tolerance = 2, int iterations = 2) 
{
    // 가장자리 처리를 위한 복제 패딩 추가
    Mat padded;
    copyMakeBorder(input, padded, 1, 1, 1, 1, BORDER_REPLICATE);

    // float 형태로 변환
    Mat src;
    padded.convertTo(src, CV_32F);
    Mat result = src.clone();

    int h = src.rows;
    int w = src.cols;

    // 지정된 횟수만큼 반복 적용
    for (int iter = 0; iter < iterations; ++iter) 
    {
        Mat temp = result.clone();

        for (int y = 0; y < h; ++y) 
        {
            for (int x = 0; x < w; ++x) 
            {
                float center = temp.at<float>(y, x);

                // 잡음이 아니면 그대로 유지
                if (!IsNoise((uchar)center, tolerance)) 
                {
                    result.at<float>(y, x) = center;
                    continue;
                }

                // 4방향 선형 보간
                vector<pair<Point, Point>> directions = 
                {
                    {Point(0, -1), Point(0, 1)},
                    {Point(-1, 0), Point(1, 0)},
                    {Point(-1, -1), Point(1, 1)},
                    {Point(-1, 1), Point(1, -1)}
                };

                float bestEstimate = 128.0f; // 두 픽셀의 평균값
                float minDiff = FLT_MAX;
                bool found = false;

                // 선형 보간 수행: diff가 가장 적은 방향 선택
                for (auto& dir : directions) 
                {
                    Point p1 = dir.first, p2 = dir.second;
                    int y1 = y + p1.y, x1 = x + p1.x;
                    int y2 = y + p2.y, x2 = x + p2.x;

                    if (y1 >= 0 && y1 < h && x1 >= 0 && x1 < w && y2 >= 0 && y2 < h && x2 >= 0 && x2 < w) 
                    {

                        float v1 = temp.at<float>(y1, x1);
                        float v2 = temp.at<float>(y2, x2);

                        if (!IsNoise((uchar)v1, tolerance) && !IsNoise((uchar)v2, tolerance)) 
                        {
                            float diff = fabs(v1 - v2); // diff가 작을 수록 평탄함

                            if (diff < minDiff) 
                            {
                                minDiff = diff;
                                bestEstimate = (v1 + v2) / 2.0f;
                                found = true;
                            }
                        }
                    }
                }

                // 공간 기반 가중 평균 수행
                float weightedSum = 0.0f, totalWeight = 0.0f;

                for (int dy = -1; dy <= 1; ++dy) 
                {
                    for (int dx = -1; dx <= 1; ++dx) 
                    {
                        int ny = y + dy;
                        int nx = x + dx;

                        if (ny >= 0 && ny < h && nx >= 0 && nx < w) 
                        {
                            float neighbor = temp.at<float>(ny, nx);
                            if (!IsNoise((uchar)neighbor, tolerance)) 
                            {
                                float w = SpatialWeight(dx, dy, 0.75f);
                                weightedSum += neighbor * w;
                                totalWeight += w;
                            }
                        }
                    }
                }

                // 보간값 대입
                if (totalWeight > 0.0f) 
                {
                    result.at<float>(y, x) = weightedSum / totalWeight;
                }
                else if (found) 
                {
                    result.at<float>(y, x) = bestEstimate;
                }
                else 
                {
                    // 모든 주변값이 잡음일 경우: 평균값으로 대체
                    float sum = 0.0f;
                    int count = 0;
                    for (int dy = -1; dy <= 1; ++dy) 
                    {
                        for (int dx = -1; dx <= 1; ++dx) 
                        {
                            int ny = y + dy, nx = x + dx;

                            if (ny >= 0 && ny < h && nx >= 0 && nx < w) 
                            {
                                float neighbor = result.at<float>(ny, nx);

                                if (!IsNoise((uchar)neighbor, tolerance)) 
                                {
                                    sum += neighbor;
                                    ++count;
                                }
                            }
                        }
                    }
                    result.at<float>(y, x) = (count > 0) ? sum / count : 128.0f;
                }
            }
        }
    }

    Mat finalResult = result(Rect(1, 1, input.cols, input.rows)).clone();
    finalResult.convertTo(finalResult, CV_8U);

    return finalResult;
}

int main(int argc, char* argv[])
{
	Mat src					= imread("C:/Users/dhgus/Downloads/src.png", 0);
	Mat gaussianNoiseImg    = imread("C:/Users/dhgus/Downloads/gns.png", 0);
	Mat pepperSaltNoiseImg  = imread("C:/Users/dhgus/Downloads/pns.png", 0);

	Mat newGns, newPsn;

	GaussianBlur(gaussianNoiseImg, newGns, Size(3, 3), 0.75);

    newPsn = PfaDenoise(pepperSaltNoiseImg);

	cout << "Gaussian PSNR : " << PSNR(src, newGns) << endl;
	cout << "Salt and Pepper PSNR : " << PSNR(src, newPsn) << endl;

	imshow("New Gaussian", newGns);
	imshow("New Salt and Pepper", newPsn);

	waitKey();

	return 0;
}

