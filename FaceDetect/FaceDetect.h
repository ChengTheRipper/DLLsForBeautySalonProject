#pragma once
#ifdef FACEDETECT_EXPORTS
#define FDAPI __declspec(dllexport)
#else
#define FDAPI __declspec(dllimport)
#endif // FACEDETECT_EXPORTS

//header files
#include "src/FaceProc/opencv_includes.h"
#include "src/FaceProc/std_includes.h"
#include "src/FaceProc/torch_lib_includes.h"

const std::string haar_file_name("Resource_Depo/face/haarcascade_frontalface_alt2.xml");
const std::string torch_file_name("Resource_Depo/face/Face_sematic_seg_model.pt");
const cv::String model_bin = "Resource_Depo/face/opencv_face_detector_uint8.pb";
const cv::String config_text = "Resource_Depo/face/opencv_face_detector.pbtxt";
const cv::String genderProto = "Resource_Depo/gender/gender_deploy.prototxt";
const cv::String genderModel = "Resource_Depo/gender/gender_net.caffemodel";


//默认虚函数类
class IFaceDetect
{
public:
	virtual bool StartGrab() = 0;
	virtual bool GetFace() = 0;
	virtual void GetFrame() = 0;
	virtual void Write2Disk() = 0;
	virtual void ShowSrc() = 0;
	virtual void Release() = 0;
};

class FaceDetect : public IFaceDetect
{
public:
	FaceDetect();
	~FaceDetect();

	//继承的实现
	bool StartGrab();
	bool GetFace();
	void GetFrame();
	void Write2Disk();
	void ShowSrc();
	void Release();
private:
	//三个区域分别的索引
	enum TypeIndex
	{
		BACKGROUND = 0,
		FACE = 127,
		HAIR = 254
	};
	//////////////////////////////////////////方法//////////////////////////////////////////
	void GetGender(const cv::Mat& input);
	bool FaceDetectTorch(const cv::Mat& input);
	bool GetSegments();

	///////后处理////////
	//总美化函数
	void FaceBeautify(cv::Mat& input, cv::Mat& output);
	/*
	dx ,fc 磨皮程度与细节程度的确定 双边滤波参数
	transparency 透明度
	*/
	void FaceGrinding(cv::Mat& input, cv::Mat& output, int value1 = 3, int value2 = 1);//磨皮
	//saturation    max_increment
	void AdjustSaturation(cv::Mat& input, cv::Mat& output, int saturation = 0, const int max_increment = 200);
	//alpha 调整对比度				beta 调整亮度
	void AdjustBrightness(cv::Mat& input, cv::Mat& output, float alpha = 1.1, float beta = 40);

	//去除背景
	void RemoveBackground(cv::Mat& img);
	////////////////////////////////////////分类器///////////////////////////////////////////
	cv::CascadeClassifier face_cascade_;

	torch::jit::Module sematic_module_;

	cv::dnn::Net face_net_;
	cv::dnn::Net gender_net_;
	//视频控制器
	cv::VideoCapture cap_;

	//图片对象
	//Mat
	cv::Mat src_;
	cv::Mat dst_;
	cv::Mat dst_torch_;

	//face_beautified
	cv::Mat face_beautified_;
	//只有脸
	cv::Mat roi_face_only_;

	//未经过裁切
	cv::Mat roi_face_all_;
	//脸加头发
	cv::Mat roi_face_hair_;

	//杂项
	cv::Rect rect_face_;
	//person's gender
	std::string cur_gender_;
};


//工厂函数，产生一个facedetect实例
extern "C" FDAPI IFaceDetect * GetFD();
