//１．マウスで各フレームのフィールド上の点と仮想フィールド上の点をを選び、対応点としてホモグラフィ変換を推定、変換
//２．変換後、ズレがあったらマウスで任意の点をインタラクティブに動かしズレを補正する

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace cv;
using namespace std;

string folder_path = "GC-A-01";
string file_path = "00140";

//縮尺 
static int u = 12;
 
static Point2d points[80]; //仮想フィールド上の各点格納する配列
static Point2d ptd[80];
vector<Point2f> pt1, pt2, pt3, pt4; //フィールド上の点、画像上の点、フィールドのコーナー、パノラマ上のコーナー
int counter1, counter2;
int c; //マウスで射影変換するときの点の数
int c_ex;
Mat f_img(Size(106*u, 68*u), CV_8UC3, Scalar(0,0,0));
Mat past_f_img;
Mat src;
Mat past_src;
string image_name;
Mat homo_mat;


//パノラマ画像の雛形
Mat ppp(Size(src.cols * 4, src.rows * 2) , CV_8UC3, Scalar(0,0,0));

struct mouseparam{
  int x;
  int y;
  int event;
  int flags;
};

// 射影変換のための各情報を保存する構造体
struct ImageInfo {
	Mat src, dst;		// 入力画像と出力画像
	Mat matrix;			// 射影変換行列
	Point2f srcPt[100];	// 変換前の座標
	Point2f dstPt[100];	// 変換後の座標
	vector<Point2f> VsrcPt;	// 変換前の座標(vector)
	vector<Point2f> VdstPt;	// 変換後の座標(vector)
	string winName;		// 出力ウインドウの名前
};
 

void format_field_points(Point2d* p){

  //センターサークル
  p[0] = Point2d(53*u,34*u); //circle_center
  p[1] = Point2d(53*u,24.85*u); //circle_top
  p[2] = Point2d(53*u,43.15*u); //circle_bottom

  //センターライン
  p[3] = Point2d(53*u,0*u); //centerline_top
  p[4] = Point2d(53*u,68*u); //centerline_bottom
  
  //コーナー
  p[5] = Point2d(0,0); //l_corner_top
  p[6] = Point2d(0*u,68*u); //l_corner_bottom
  p[7] = Point2d(106*u,0*u); //r_corner_top
  p[8] = Point2d(106*u,68*u); //r_corner_bottom

  //左
  p[9] = Point2d(0*u,13.84*u); //l_parea_gline_top
  p[10] = Point2d(0*u,54.16*u); //l_parea_gline_bottom
  p[11] = Point2d(16.5*u,13.84*u); //l_parea_top
  p[12] = Point2d(16.5*u,54.16*u); //l_parea_bottom

  p[13] = Point2d(0,24.84*u); //l_garea_gline_top
  p[14] = Point2d(0,43.16*u); //l_garea_gline_bottom
  p[15] = Point2d(5.5*u,24.84*u); //l_garea_top
  p[16] = Point2d(5.5*u,43.16*u); //l_garea_bottom

  p[17] = Point2d(16.5*u,26.69*u); //l_parc_top
  p[18] = Point2d(16.5*u,41.31*u); //l_parc_bottom
  p[19] = Point2d(11*u,34*u); //l_pmark
  
  //右
  p[20] = Point2d(106*u,13.84*u); //r_parea_gline_top
  p[21] = Point2d(106*u,54.16*u); //r_parea_gline_bottom
  p[22] = Point2d(89.5*u,13.84*u); //r_parea_top 
  p[23] = Point2d(89.5*u,54.16*u); //r_parea_bottom
  
  p[24] = Point2d(106*u,24.84*u); //r_garea_gline_top
  p[25] = Point2d(106*u,43.16*u); //r_garea_gline_bottom
  p[26] = Point2d(100.5*u,24.84*u); //r_garea_top
  p[27] = Point2d(100.5*u,43.16*u); //r_garea_bottom

  p[28] = Point2d(89.5*u,26.69*u); //r_parc_top
  p[29] = Point2d(89.5*u,41.31*u); //r_parc_bottom
  p[30] = Point2d(95*u,34*u); //r_pmark
  
  //芝目
  p[31] = Point2d(5.5*u, 0); //glass_top_1
  p[32] = Point2d(11*u,0); //2
  p[33] = Point2d(16.5*u,0); //3
  p[34] = Point2d(21.714*u,0); //4
  p[35] = Point2d(26.928*u,0); //5
  p[36] = Point2d(32.142*u,0); //6
  p[37] = Point2d(37.356*u,0); //7
  p[38] = Point2d(42.57*u,0); //8
  p[39] = Point2d(47.784*u,0); //9

  p[40] = Point2d(58.214*u,0); //10
  p[41] = Point2d(63.428*u,0); //11
  p[42] = Point2d(68.682*u,0); //12
  p[43] = Point2d(73.856*u,0); //13
  p[44] = Point2d(79.07*u,0); //14
  p[45] = Point2d(84.284*u,0); //15
  p[46] = Point2d(89.5*u,0); //16
  p[47] = Point2d(95*u,0); //17
  p[48] = Point2d(100.5*u,0); //18
  
  p[49] = Point2d(5.5*u,13.84*u); //glass_parea_top_1
  p[50] = Point2d(11*u,13.84*u); //2
  p[51] = Point2d(95*u,13.84*u); //3
  p[52] = Point2d(100.5*u,13.84*u); //4
  
  p[53] = Point2d(5.5*u,54.16*u); //glass_parea_bottom_1
  p[54] = Point2d(11*u,54.16*u); //2
  p[55] = Point2d(95*u,54.16*u); //3
  p[56] = Point2d(100.5*u,54.16*u); //4
  
  p[57] = Point2d(5.5*u,68*u); //glass_bottom_1
  p[58] = Point2d(11*u,68*u); //2
  p[59] = Point2d(16.5*u,68*u); //3
  p[60] = Point2d(21.714*u,68*u); //4
  p[61] = Point2d(26.928*u,68*u); //5
  p[62] = Point2d(32.142*u,68*u); //6
  p[63] = Point2d(37.356*u,68*u); //7
  p[64] = Point2d(42.57*u,68*u); //8
  p[65] = Point2d(47.784*u,68*u); //9
  
  p[66] = Point2d(58.214*u,68*u); //10
  p[67] = Point2d(63.428*u,68*u); //11
  p[68] = Point2d(68.682*u,68*u); //12
  p[69] = Point2d(73.856*u,68*u); //13
  p[70] = Point2d(79.07*u,68*u); //14
  p[71] = Point2d(84.284*u,68*u); //15
  p[72] = Point2d(89.5*u,68*u); //16
  p[73] = Point2d(95*u,68*u); //17
  p[74] = Point2d(100.5*u,68*u); //18

  p[75] = Point2d(103.25*u,34*u); //ゴールエリアの重心点
}

//サッカーフィールドを作成
void field_maker(Point2d* p, Mat img){
  double angle = 90; 
  //芝目
  line(img, p[31], p[57], Scalar(0,200,0), 2, 4);
  line(img, p[32], p[58], Scalar(0,200,0), 2, 4);
  line(img, p[33], p[59], Scalar(0,200,0), 2, 4);
  line(img, p[34], p[60], Scalar(0,200,0), 2, 4);
  line(img, p[35], p[61], Scalar(0,200,0), 2, 4);
  line(img, p[36], p[62], Scalar(0,200,0), 2, 4);
  line(img, p[37], p[63], Scalar(0,200,0), 2, 4);
  line(img, p[38], p[64], Scalar(0,200,0), 2, 4);
  line(img, p[39], p[65], Scalar(0,200,0), 2, 4);
  line(img, p[40], p[66], Scalar(0,200,0), 2, 4);
  line(img, p[41], p[67], Scalar(0,200,0), 2, 4);
  line(img, p[42], p[68], Scalar(0,200,0), 2, 4);
  line(img, p[43], p[69], Scalar(0,200,0), 2, 4);
  line(img, p[44], p[70], Scalar(0,200,0), 2, 4);
  line(img, p[45], p[71], Scalar(0,200,0), 2, 4);
  line(img, p[46], p[72], Scalar(0,200,0), 2, 4);
  line(img, p[47], p[73], Scalar(0,200,0), 2, 4);
  line(img, p[48], p[74], Scalar(0,200,0), 2, 4);
  //センターサークル
  line(img, p[1], p[2], Scalar(0,0,200), 2, 4);
  circle(img, p[0], 9.15*u, Scalar(0,0,200), 2, 4);
  line(img, p[0], p[0], Scalar(0,0,200), 5,4);
  
  //センターライン
  line(img, p[3], p[4], Scalar(0,0,200), 2, 4);
  
  //コーナー
  line(img, p[5], p[6], Scalar(0,0,200), 2, 4);
  line(img, p[5], p[7], Scalar(0,0,200), 2, 4);
  line(img, p[6], p[8], Scalar(0,0,200), 2, 4);
  line(img, p[7], p[8], Scalar(0,0,200), 2, 4);

  //左
  line(img, p[9], p[11], Scalar(0,0,200), 2, 4);
  line(img, p[11], p[12], Scalar(0,0,200), 2, 4);
  line(img, p[10], p[12], Scalar(0,0,200), 2, 4);

  line(img, p[13], p[15], Scalar(0,0,200), 2, 4);
  line(img, p[15], p[16], Scalar(0,0,200), 2, 4);  
  line(img, p[14], p[16], Scalar(0,0,200), 2, 4);

  line(img, p[17], p[17], Scalar(0,200,0), 3, 4);
  line(img, p[18], p[18], Scalar(0,200,0), 3, 4);
  line(img, p[19], p[19], Scalar(0,0,200), 5, 4);
  ellipse(img, p[19], Size(9.15*u, 9.15*u), -53, 0, 106, Scalar(0,0,200), 2, 4);

  //右
  line(img, p[20], p[22], Scalar(0,0,200), 2, 4);
  line(img, p[22], p[23], Scalar(0,0,200), 2, 4);
  line(img, p[21], p[23], Scalar(0,0,200), 2, 4);

  line(img, p[24], p[26], Scalar(0,0,200), 2, 4);
  line(img, p[26], p[27], Scalar(0,0,200), 2, 4);  
  line(img, p[25], p[27], Scalar(0,0,200), 2, 4);
  
  line(img, p[28], p[28], Scalar(0,200,0), 3, 4);
  line(img, p[29], p[29], Scalar(0,200,0), 3, 4);
  line(img, p[30], p[30], Scalar(0,0,200), 5, 4);
  ellipse(img, p[30], Size(9.15*u, 9.15*u), 127, 0, 106, Scalar(0,0,200), 2, 4);


  line(img, p[75], p[75], Scalar(0,0,200), 3, 4);
}
using namespace cv;
using namespace std;

//クリックした位置に一番近い点の番号を返す変数
int dist(int x, int y){
  double ans = 1000000;
  double dis;
  int i, j = 0;
  
  for(i = 0 ; i <= 76 ; i++){
    dis = sqrt(pow((double)x - points[i].x,2) + pow((double)y - points[i].y,2));
        
    if(dis < ans){
      ans = dis;
      j = i;
    }
    //cout << j << endl;
  }

  if(ans <= 1){
    return j;
  }else{
    return -1;
  }
}

//コールバック関数(画像用)
void CallBackFunc(int event, int x, int y, int flags, void* param)
{
    switch(event){
    case EVENT_LBUTTONDOWN:
      //一つ前の状態を保存
      src.copyTo(past_src);
      //クリックした地点に点をうつ
      line(src, Point(x,y), Point(x,y), Scalar(0,0,255), 3, 7);
      imshow(image_name, src);
      
      ptd[counter1] = Point(x,y);
      //cout << Point(x,y) << endl;
      
      counter1++;
      counter2++;

      break;

    case EVENT_RBUTTONDOWN:
      //一つ前の状態に戻す(２つ前の状態には戻せません。実装するのめんどい)
      if(counter1 > 0 && counter2 > 0){
	counter1--;
	counter2--;
	//cout << "undo" << endl;
	imshow(image_name, past_src);
	past_src.copyTo(src);
      }

      break;
    }
}

//コールバック関数(仮想フィールド用)
void SecondCallBackFunc(int event, int x, int y, int flags, void* param)
{
  //int f = 0;
  int z;
  //左クリックがあったら表示
    switch(event){
    case EVENT_LBUTTONDOWN:

    f_img.copyTo(past_f_img); //点を打つ前の状態を保存
      //それぞれの点の距離を算出して一番近い点を採用する
      z = dist(x, y);
      if(z != -1){
	ptd[counter1] = points[z];
	//クリックした地点に白点をうつ
	line(f_img, points[z], points[z], Scalar(255,255,255), 3, 6);
      } else{
	ptd[counter1] = Point(x,y);
	//クリックした地点に白点をうつ
	line(f_img, Point(x,y), Point(x,y), Scalar(255,255,255), 3, 6);
      }

      imshow(image_name, f_img);
      //cout << ptd[counter1] << endl;
      
      //クリック回数
      counter1++; 

      break;

    case EVENT_RBUTTONDOWN:
      //一つ前の状態に戻す(２つ前の状態には戻せません。実装するのめんどい)
      if(counter1 > counter2){
	counter1--;
	//cout << "undo" << endl;
	imshow(image_name, past_f_img);
	past_f_img.copyTo(f_img);
      }

      break;
    }
    

}

//maskでラインのみを画像上に重複表示
void mask_maker(Mat mask, Mat dst_r, Mat src){

  for(int y = 0; y < mask.rows; y++){
    Vec3b* ptr_mask = mask.ptr<Vec3b>( y );
    Vec3b* ptr_dst_r = dst_r.ptr<Vec3b>( y );
    for(int x = 0; x < mask.cols; x++){
      if(ptr_dst_r[x] == Vec3b(0,0,0)){
	  ptr_mask[x] = Vec3b(0,0,0);
	}
      else{
	ptr_mask[x] = Vec3b(255,255,255);
      }
    }
  }
  
  for(int y = 0; y < mask.rows; y++){
    Vec3b* ptr_src = src.ptr<Vec3b>( y );
    Vec3b* ptr_mask = mask.ptr<Vec3b>( y );
    Vec3b* ptr_dst_r = dst_r.ptr<Vec3b>( y );
    for(int x = 0; x < mask.cols; x++){
      if(ptr_mask[x] == Vec3b(255,255,255)){
	ptr_src[x] = ptr_dst_r[x];
      }
    }
  }
}


// コールバック関数(homography)
void mouseCallback(int event, int x, int y, int flags, void *data)
{
	static int select = -1;		// マウスで選択された頂点番号（-1:選択無し）
	cv::Point2f p(x, y);		// マウスの座標
	double dis = 1e10;
 	Mat aa, bb;
 	double zz;
	ImageInfo &info = *(ImageInfo *)data;
 	Point2f P;
 	
 	
	switch (event) {
	case cv::EVENT_LBUTTONDOWN:
		// 左ボタンを押したとき、指定した点のうち一番近い点を探す
		for (int i = 0; i < c_ex; i++) {
			double d = cv::norm(p - info.dstPt[i]);	// 頂点iとマウス座標との距離を計算
			if (d < 20 && d < dis) {
				select = i;
				dis = d;
			}
		}
		/*
		//左ボタンを押したときその点を新しい点としてsrcPtに登録する
		aa = (Mat_<float> (3,1) << p.x, p.y, 1);
		bb = homo_mat.inv() * aa;
		zz = bb.at<float>(2, 0);
		bb = bb / zz;
		P = Point2f(bb.at<float>(0, 0), bb.at<float>(1, 0));
		info.srcPt[c] = P;
		info.VsrcPt.at(c) = P;
		select = c;
		*/
		break;
 
	case cv::EVENT_RBUTTONDOWN:
		// 右ボタンを押したとき、ホモグラフィ行列を出力する
		std::cout << info.matrix << std::endl;
		
		break;
 
	case cv::EVENT_LBUTTONUP:
		// 左ボタンを離したとき、画像を射影変換してウインドウに表示する
 
		// 変換前後の座標からホモグラフィ行列を求める
		info.matrix = findHomography(info.VsrcPt, info.VdstPt);
		//info.matrix = getPerspectiveTransform(info.srcPt, info.dstPt);
		// 射影変換をする
		cv::warpPerspective(info.src, info.dst, info.matrix, info.dst.size(), cv::INTER_LINEAR);
	      field_maker(points, info.dst);
		// ウインドウに表示する
		cv::imshow(info.winName, info.dst);
 
		select = -1;	
		break;
	
	case EVENT_MBUTTONDOWN: //マウスホイール押し込みで保存
	        // 変換前後の座標からホモグラフィ行列を求める
		info.matrix = findHomography(info.VsrcPt, info.VdstPt);
		//info.matrix = getPerspectiveTransform(info.srcPt, info.dstPt);
	    
		// 射影変換をして画像を保存する
		warpPerspective(info.src, info.dst, info.matrix, Size(108*u, 68*u), cv::INTER_LINEAR);
		field_maker(points, info.dst);
		imwrite("./output/bird_view/" + folder_path + "/bird_view_" + file_path + ".png", info.dst);
		
		break;
	}
	
	
	if (flags & cv::EVENT_FLAG_LBUTTON && select > -1) {
		// マウスの左ボタンが押されている、かつ、頂点が選択されているとき、
		// 選択されている頂点の座標を現在のマウスの位置にする
		info.dstPt[select] = p;
		info.VdstPt.at(select) = p;
		// 変換前後の座標からホモグラフィ行列を求める
		info.matrix = findHomography(info.VsrcPt, info.VdstPt);
		//info.matrix = getPerspectiveTransform(info.srcPt, info.dstPt);
		// 射影変換をする
		cv::warpPerspective(info.src, info.dst, info.matrix, info.dst.size(), cv::INTER_LINEAR);
		field_maker(points, info.dst);
		// ウインドウに表示する
		cv::imshow(info.winName, info.dst);
		

	}
	
}
 
/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////
//-----------------------------------main------------------------------------//
/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]){
  //フィールド上の点を初期化
  format_field_points(points);
  //field_maker呼び出し(仮想フィールド作成)
  field_maker(points, f_img);
  ///*

  //imwrite("./field_image_point.png", f_img);


  
  image_name = "image";
  src = imread("./../../../" + folder_path + "/png/frame-" + file_path + ".png");
  //src = imread("./../../../panoramic_image.png");
  //src = imread("./../testimage.JPG"); //テスト用画像
   ///*
   //ウィンドウサイズ
  imshow(image_name, src);


  //コールバックの設定
  mouseparam mouseEvent;
  //実際の写真に対する点を打つ処理
  cout << "image points" << endl;
  setMouseCallback(image_name , CallBackFunc, &mouseEvent);
  waitKey(0);

  int i;
  for(i=0 ; i<counter1 ; i++){
    cout << ptd[i] << endl;
  }
  cout << counter2 << "points" << endl;
  
  //用意した仮想フィールドに対する処理
  cout << endl << "field points" << endl;
  setMouseCallback(image_name, SecondCallBackFunc, &mouseEvent);
  imshow(image_name, f_img);
  waitKey(0);

  for(i=counter2 ; i < counter1 ; i++){
    cout << ptd[i] << endl;
  }
  cout << counter1 - counter2 << "points" << endl;
  
  
  //ptdを前半後半の点で分ける
  Point2f p;
  for(i = 0; i < counter1 ; i++){
    p = ptd[i];
    if( i >= 0 && i < counter2){
      pt1.push_back(p);
      
    }
    else if (i >= counter2 && i < counter1){
      pt2.push_back(p);
    }
    }
  //*/
  /*
	counter1 = 12;
	counter2 = 6;

	pt1.push_back(Point2f(1414, 484));
	pt1.push_back(Point2f(1819, 523));
	pt1.push_back(Point2f(1075, 559));
	pt1.push_back(Point2f(1479, 616));
	pt1.push_back(Point2f(454, 520));
	pt1.push_back(Point2f(208, 527));
	
	pt2.push_back(Point2f(1271, 1));
	pt2.push_back(Point2f(1271, 165));
	pt2.push_back(Point2f(1074, 166.08));
	pt2.push_back(Point2f(1073, 319));
	pt2.push_back(Point2f(1011, 2));
	pt2.push_back(Point2f(950, 1));
	*/
	
  string ofs_file;

  
  ///////////////////ホモグラフィで変換した画像を仮想フィールド上に貼り付ける/////////////////////////
  //////////////////////////////////   bird_view   //////////////////////////////////////////
  ///*
  homo_mat = findHomography(pt1, pt2, 0); //pt1をpt2に合わせに行く射影変換行列
  /*Mat homo_mat = (Mat_<double>(3, 3) <<
		  -3.404276787023332, -16.42787835543752, -8701.656978406716,
		  -2.426019632926914, -51.34036200636319, 26363.22557196613,
		  6.115313982712375e-05, -0.04029668227602698, 1);
  */
  Mat dst = f_img.clone();
  warpPerspective(src, dst, homo_mat, dst.size()); //透視投影変換
  //射影変換後の画像保存
  Mat dst1;
  dst.copyTo(dst1);
  //imwrite("./output/bird_view/" + folder_path + "/" + file_path + ".png", dst1);
  cout << homo_mat << endl;

  //変換した画像において再度仮想フィールドのラインを作成
  field_maker(points, dst);
  imshow("result", dst);
  imwrite("./output/bird_view/" + folder_path + "/bird_view_" + file_path + ".png", dst);
  //imwrite("./output/bird_view/panoramic_image.png", dst);
  ofs_file = "./output/bird_view/" + folder_path + "/homo/H_" + file_path + ".csv";
  //ofs_file = "./output/bird_view/panoramic_image_H.csv";
  ofstream ofs(ofs_file);
  ofs << format(homo_mat, Formatter::FMT_CSV) << endl;
  //*/
  ///////////////////////////////////////////////////////////////////////////////////////////

  //マウスイベントで射影変換を行い微調整により補正
	
	//指定した点の座標をもとにマウスイベント呼び出しを行い、画像変形を行う
	Size window(108*u, 68*u);
	ImageInfo info;
	info.src = src;
	//info.src = dst1;
	info.dst = Mat(info.src.size(), CV_8UC3);
	
	int count;
	cout << "counter2 = " << counter2 << endl;
	// 原画像の4頂点座標 → 原画像上の任意の固定する点 + 変更する点（左下、右下の点？）
	for(count = 0; count < counter2 ; count++){
		info.srcPt[count] = pt1[count];
		info.VsrcPt.push_back(pt1.at(count));
	}
	/*
	for(count = 0; count < 3 ; count++){
		info.srcPt[count] = pt2[count];
		info.VsrcPt.push_back(pt2.at(count));
	}
	*/
		
	c = count; //マウスで射影変換するときの点の数
	
	info.srcPt[count] = Point2f(0, src.rows);
	info.VsrcPt.push_back(Point2f(0, src.rows));
	count++;
	c_ex = count;
	
	Mat a,b;
	double z;
	b = (Mat_<double> (3,1) << 0, src.rows, 1); //左下
	a = homo_mat * b;
	z = a.at<double>(2,0);
	a = a / z;
	//*/
	
	//スケール(縮小率)を計算
	double scale;
	if (info.src.cols > info.src.rows) {
		scale = double(window.width) / info.src.cols;
	} else {
		scale = double(window.height) / info.src.rows;
	}
	//コールバック関数
	info.winName = "マウスで射影変換(微調整用)";
	namedWindow(info.winName);
	setMouseCallback(info.winName, mouseCallback, (void *)&info);
	
	int key = 'o';
	do {
	  if(key == 'o'){
	    // 変換後の固定する座標  
	    
	    for(count = 0 ; count < c; count++){
		   info.dstPt[count] = pt2[count];
	     	   info.VdstPt.push_back(pt2.at(count));
	     }
	     info.dstPt[count] = Point2f(a.at<double>(0,0), a.at<double>(1,0));
	     info.VdstPt.push_back(Point2f(a.at<double>(0,0), a.at<double>(1,0)));
	     count++;
	     c_ex = count;
	     /*
	    for(count = 0 ; count < c; count++){
		   info.dstPt[count] = info.srcPt[count];
	     	   info.VdstPt.push_back(info.VsrcPt.at(count));
	     }
	     //*/  
	    // 対応点から射影変換行列を求める（srcPoint → dstPoint1）
	    info.matrix = findHomography(info.VsrcPt, info.VdstPt);
	    //info.matrix = getPerspectiveTransform(info.srcPt, info.dstPt);
	    // 射影変換をして画像を表示する
	    warpPerspective(info.src, info.dst, info.matrix, window, cv::INTER_LINEAR);
	    field_maker(points, info.dst);
	    imshow(info.winName, info.dst);
	  }
	  
	  if (key == 'r') { //rキーで初期状態に戻す
	  
	    for(count = 0 ; count < c_ex; count++){
	     	   info.VdstPt.pop_back();
	     }
	    // 変換後の固定する座標
	    
	    for(count = 0 ; count < c; count++){
		   info.dstPt[count] = pt2[count];
	     	   info.VdstPt.push_back(pt2.at(count));
	     }
	     info.dstPt[count] = Point2f(a.at<double>(0,0), a.at<double>(1,0));
	     info.VdstPt.push_back(Point2f(a.at<double>(0,0), a.at<double>(1,0)));
	     count++;
	     c_ex = count;
	     /*
	    for(count = 0 ; count < c; count++){
		   info.dstPt[count] = info.srcPt[count];
	     	   info.VdstPt.push_back(info.VsrcPt.at(count));
	     }
	    */
	    // 対応点から射影変換行列を求める（srcPoint → dstPoint1）
	    info.matrix = findHomography(info.VsrcPt, info.VdstPt);
	    //info.matrix = getPerspectiveTransform(info.srcPt, info.dstPt);
	    // 射影変換をして画像を表示する
	    warpPerspective(info.src, info.dst, info.matrix, window, cv::INTER_LINEAR);
	    field_maker(points, info.dst);
	    imshow(info.winName, info.dst);
	  }
	  key = cv::waitKey();
	} while (key != 0x1b); //Escキーで終了
	
	
	    
  /////////////////ホモグラフィで変換した仮想フィールドをパノラマ平面上に貼り付ける/////////////////////
  /////////////////////////////////    panorama_line    /////////////////////////////////////
  /*
  Mat homo_mat_pl = findHomography(pt3, pt4, 0);
  Mat homo_mat_fpl = homo_mat_pl * homo_mat;
  
  Mat dst_pl, dst_fpl;
  warpPerspective(src, dst_pl, homo_mat_fpl, ppp.size());
  warpPerspective(f_img, dst_fpl, homo_mat_pl, ppp.size());
  Mat mask_pl(Size(ppp.cols, ppp.rows), CV_8UC3, Scalar(0,0,0));

  mask_maker(mask_pl, dst_fpl, dst_pl);
  
  //imshow("result", ppp);
  imwrite("./output/panorama_l/" + folder_path + "/panorama_l_" + file_path + ".png", dst_pl);
  ofs_file = "./output/panorama_l/" + folder_path + "/homo/H_" + file_path + ".csv";
  ofstream ofs_pl(ofs_file);
  ofs_pl << format(homo_mat_fpl, Formatter::FMT_CSV) << endl;
  //*/
  ///////////////////////////////////////////////////////////////////////////////////////////
  
 /////////////////ホモグラフィで変換した仮想フィールドをパノラマ平面上に貼り付ける/////////////////////
  /////////////////////////////////    panorama   //////////////////////////////////////////
  /*
  Mat homo_mat_p = findHomography(pt3, pt4, 0);
  Mat homo_mat_fp = homo_mat_p * homo_mat;
  
  Mat dst_p;
  warpPerspective(src, dst_p, homo_mat_fp, ppp.size());
  
  imwrite("./output/panorama/" + folder_path + "/panorama_" + file_path + ".png", dst_p);
  ofs_file = "./output/panorama/" + folder_path + "/homo/H_" + file_path + ".csv";
  ofstream ofs_p(ofs_file);
  ofs_p << format(homo_mat_p, Formatter::FMT_CSV) << endl;
  //*/
  ///////////////////////////////////////////////////////////////////////////////////////////
  
  /////////////////ホモグラフィで変換した仮想フィールド上の線を画像に貼り付ける////////////////////////
  ////////////////////////////////   image_line   ///////////////////////////////////////////
  /*
  Mat homo_mat_r = findHomography(pt2, pt1, 0);

  Mat dst_r;
  warpPerspective(f_img, dst_r,homo_mat_r, src.size());

  Mat mask(Size(src.cols, src.rows), CV_8UC3, Scalar(0,0,0));

  mask_maker(mask, dst_r, src);
  //imshow("result", src);
  imwrite("./output/image_line/" + folder_path + "/image_l_" + file_path + ".png", src);
  ofs_file = "./output/image_line/" + folder_path + "/homo/H_" + file_path + ".csv";
  ofstream ofs_r(ofs_file);
  ofs_r << format(homo_mat_r, Formatter::FMT_CSV) << endl;
  //*/
  ///////////////////////////////////////////////////////////////////////////////////////////

  return 0;
}
