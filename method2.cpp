#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/opencv.hpp>
#include <iostream> //ifstream
#include <fstream>  //ofstream
#include <string.h>
#include <sstream>  //文字ストリーム

using namespace cv;
using namespace std;


//-----画像を合成する関数-----
void paste(Mat dst, Mat src, int x, int y) {

  if (x >= dst.cols || y >= dst.rows) {
    cout << "cannot" << endl;
    return;
  }
  int w = (x >= 0) ? min(dst.cols - x, src.cols) : min(max(src.cols + x, 0), dst.cols);
  int h = (y >= 0) ? min(dst.rows - y, src.rows) : min(max(src.rows + y, 0), dst.rows);
  int u = (x >= 0) ? 0 : min(-x, src.cols - 1);
  int v = (y >= 0) ? 0 : min(-y, src.rows - 1);
  int px = std::max(x, 0);
  int py = std::max(y, 0);
  
  cv::Mat roi_dst = dst(cv::Rect(px, py, w, h));
  cv::Mat roi_resized = src(cv::Rect(u, v, w, h));
  roi_resized.copyTo(roi_dst);
}

///-----センサ情報格納用構造体-----
struct sensor_info
{
  float yaw;
  float pitch;
  float roll;
};
//-----センサ情報の読み込み関数-----
const sensor_info sensor_stream(string sensor) {
  ifstream ifs(sensor);
  if (!ifs) {
    cout << "input error!" << endl;
  }
  
  string str;
  vector<vector<string>> values;   //string型配列
  sensor_info data;    //float型の構造体
  
  int i;
  while (getline(ifs, str)) {     //first画像のセンサ情報読み込み
    vector<string> inner;
    while ((i = str.find(",")) != str.npos) {
      inner.push_back(str.substr(0, i));
      str = str.substr(i + 1);   //', 'を読み飛ばす
    }
    inner.push_back(str);
    values.push_back(inner);
  }
  
  //cout << "sensor info" << endl;
  for (unsigned int i = 0; i < values.size(); ++i) {    //読み込んだ値をdouble型にキャスト
    data.yaw = stod(values[i][0]);
    data.pitch = stod(values[i][1]);
    data.roll = stod(values[i][2]);
    
  }
  
  return data;
}

//-----homography行列を読み取る関数-----
const Mat data_stream(string path) {

  ifstream ifs(path);
  if (!ifs) {
    cout << "input error!" << endl;
  }
  string str;
  vector<vector<string>> values;   //string型配列

  int i;
  while (getline(ifs, str)) {
    vector<string> inner;
    while ((i = str.find(",")) != str.npos) {
      inner.push_back(str.substr(0, i));
      str = str.substr(i + 1);   //', 'を読み飛ばす
    }
    inner.push_back(str);
    values.push_back(inner);
  }
  Mat homo = (Mat_<double>(3, 3) << stod(values[0][0]), stod(values[0][1]), stod(values[0][2]),
	      stod(values[1][0]), stod(values[1][1]), stod(values[1][2]),
	      stod(values[2][0]), stod(values[2][1]), stod(values[2][2]));

  return homo;
}

//-------------------------------------------------main---------------------------------------------------//
int main(int argc, char* argv[])
{

  //-----使うフレームの設定-----
  int a, b;
  a = 1;		//一枚目のフレーム
  b = 255;              //最後のフレーム
  static Mat dst;
  string foldername = "sensor";
  
  //----1枚目の画像とHbaseを指定----
  string first = "./../../GC-A-01/png/frame-00001.png";       //1枚目の画像のパス
  Mat Hbase = data_stream("input/1homography_1base.csv");
  Mat Hall = Hbase;

  //-----1枚目の画像読み込み-----
  Mat src1 = imread(first);
  Mat brack1(Size(src1.cols * 4, src1.rows * 2), CV_8UC3, Scalar(0, 0, 0));
  paste(brack1, src1, src1.cols * 0.5, src1.rows * 0.5);
  imwrite("output/" + foldername + "/result/frame1.png", brack1);
  //-----姿勢情報読み込み-----
  ifstream ifs("input/ypr_daiji/sensor_ypr.csv");
  //ifstream ifs("input/tech2_after.csv");
  if (!ifs) {
    cout << "input error!" << endl;
  }

  string str;
  vector<vector<string>> values;   //string型配列
  sensor_info data[300];    //float型の構造体
  int i;
  while (getline(ifs, str)) {     //first画像のセンサ情報読み込み
    vector<string> inner;
    while ((i = str.find(",")) != str.npos) {
      inner.push_back(str.substr(0, i));
      str = str.substr(i + 1);   //', 'を読み飛ばす
    }
    inner.push_back(str);
    values.push_back(inner);
  }

  for (unsigned int i = 0; i < values.size(); ++i) {    //読み込んだ値をdouble型にキャスト
    data[i].yaw = stod(values[i][0]);
    data[i].pitch = stod(values[i][1]);
    data[i].roll = stod(values[i][2]);

  }

  //-----パノラマ平面画像の回転行列を設定----
  double Yaw1, Pitch1, Roll1;

  Yaw1 = -data[0].yaw * (3.14 / 180);
  Pitch1 = data[0].pitch * (3.14 / 180);
  Roll1 = data[0].roll * (3.14 / 180);

  Mat R1_y = (Mat_<double>(3, 3) << cos(Yaw1), 0, sin(Yaw1),			//Y軸 横
	      0, 1, 0,
	      -sin(Yaw1), 0, cos(Yaw1));

  Mat R1_p = (Mat_<double>(3, 3) << 1, 0, 0,							//X軸　縦
	      0, cos(Pitch1), sin(Pitch1),
	      0, -sin(Pitch1), cos(Pitch1));

  Mat R1_r = (Mat_<double>(3, 3) << cos(Roll1), -sin(Roll1), 0,		//Z軸　奥
	      sin(Roll1), cos(Roll1), 0,
	      0, 0, 1);

  Mat R1 = R1_y * R1_p * R1_r;			//roll,pitch,yawの順に変換

  //-----2枚目以降の画像のパス用の関数を用意-----
  string second;      //2枚目の画像のパス用変数

  //-----マスク画像の生成-----
  Mat white(Size(src1.cols, src1.rows), CV_8UC3, Scalar(255, 255, 255));
  Mat mask(Size(src1.cols * 4, src1.rows * 2), CV_8UC3, Scalar(0, 0, 0));
  paste(mask, white, src1.cols * 0.5, src1.rows * 0.5);
  imwrite("output/" + foldername + "/masks/mask1.png", mask);


  char cstr5f[32];
  char cstr1f[32];
  char cstr5s[32];
  char cstr1s[32];
  //-----指定した枚数分、ファイル内の画像を合成-----
  for (int i = 1; i < b; i++) {
    //2枚目以降の画像のパスを指定
    sprintf(cstr5f, "%05d", i);
    sprintf(cstr1f, "%d", i);
    sprintf(cstr5s, "%05d", i+1);
    sprintf(cstr1s, "%d", i+1);
    
    string frame5f = string(cstr5f);
    string frame1f = string(cstr1f);
    string frame5s = string(cstr5s);
    string frame1s = string(cstr1s);
    
    second = "./../../GC-A-01/png/frame-" + frame5s + ".png";

    //-----画像読み込み-----
    Mat src2 = imread(second);
    Mat brack2(Size(src1.cols * 4, src1.rows * 2), CV_8UC3, Scalar(0, 0, 0));
    paste(brack2, src2, src1.cols * 0.5, src1.rows * 0.5);
		
    //-----姿勢情報をラジアン表記に変換-----
    double Yaw2, Pitch2, Roll2;

    Yaw2 = -data[i-1].yaw * (3.14 / 180);
    Pitch2 = data[i-1].pitch * (3.14 / 180);
    Roll2 = data[i-1].roll * (3.14 / 180);

    //-----フレームごとの回転行列の作成-----

    Mat R2_y = (Mat_<double>(3, 3) << cos(Yaw2), 0, sin(Yaw2),			//Y軸
		0, 1, 0,
		-sin(Yaw2), 0, cos(Yaw2));

    Mat R2_p = (Mat_<double>(3, 3) << 1, 0, 0,                           	//X軸
		0, cos(Pitch2), sin(Pitch2),
		0, -sin(Pitch2), cos(Pitch2));

    Mat R2_r = (Mat_<double>(3, 3) << cos(Roll2), -sin(Roll2), 0,		//Z軸
		sin(Roll2), cos(Roll2), 0,
		0, 0, 1);

    Mat R2 = R2_y * R2_p * R2_r;		//roll,pitch,yawの順に回転

    Mat R = R1 * R2.inv();			//フレーム画像の平面からパノラマ画像平面へ

    //-----カメラ内部パラメータ-----       
    //パノラマ平面画像：内部パラメータは任意に設定
    Mat A = (Mat_<double>(3, 3) << 6.5446779777710044e+03, 0, 9.8739807499423318e+02,
	     0, 6.5453150788456724e+03, 8.5046671750973985e+02,
	     0, 0, 1);

    //-----ホモグラフィ行列の導出-----
    Mat homography, homography_relate;
    homography = A * R2.inv() * A.inv();

    cout << "homography" << endl;
    cout << homography << endl;
    //----ホモグラフィー行列を表示し、csvファイルに書き込む----
    string csv_name = "output/" + foldername + "/homography/frame_homography/frame_homography_" + frame1f + "to" + frame1s + ".csv"; 
    ofstream ofs3(csv_name);
    ofs3 << format(homography, Formatter::FMT_CSV) << endl;
		
    //-----ホモグラフィ行列を掛け合わせる-----
    Hall = Hall * homography;
    string allss = "output/" + foldername + "/homography/all_homography/all_homography_" + frame1s + ".csv";
    ofstream ofs_all(allss);
    ofs_all << format(Hall, Formatter::FMT_CSV) << endl;
    //----画像変換を行う----
    warpPerspective(brack2, dst, Hall, Size(src2.cols * 4, src2.rows * 2));		//縦横2倍のパノラマ平面に射影
    string tt = "output/" + foldername + "/result/frame" + frame1s + ".png";
    imwrite(tt, dst);      //射影変換された画像を保存

    //-----マスクも作る-----
    warpPerspective(mask, dst, Hall, Size(src2.cols * 4, src2.rows * 2));
    imwrite("output/" + foldername + "/masks/mask" + frame1s + ".png", dst);
  }

  waitKey(0);
  return 0;
}
