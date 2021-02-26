#include "pch.h"
#include "FaceDetect.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;


FDAPI IFaceDetect* GetFD()
{
	IFaceDetect* p = new FaceDetect();
	if (!p)
	{
		throw bad_alloc();
	}
	return p;
}

FDAPI IFaceDetect* GetFD_Sharp()
{
	IFaceDetect* p = new FaceDetect();
	if (!p)
	{
		throw bad_alloc();
	}
	return p;
}

FDAPI bool StartGrab(IFaceDetect* fdpt)
{
	return fdpt->StartGrab();
}

FDAPI bool ProcessFace(IFaceDetect* fdpt)
{
	return fdpt->ProcessFace();
}

FDAPI void GetFrame(IFaceDetect* fdpt)
{
	fdpt->GetFrame();
}

FDAPI void Write2Disk(IFaceDetect* fdpt)
{
	fdpt->Write2Disk();
}

FDAPI void ShowSrc(IFaceDetect* fdpt)
{
	fdpt->ShowSrc();
	waitKey(1);
}

FDAPI void Release(IFaceDetect* fdpt)
{
	fdpt->Release();
}

FaceDetect::FaceDetect()
{
	//读入各个模型
	if (!haar_detector.load(haar_file_name))
	{
		std::cout << "error loading haar_file !" << std::endl;
	}
	//torch模型
	sematic_module_ = torch::jit::load(torch_file_name);
	//opencv脸模型
	face_net_ = cv::dnn::readNetFromTensorflow(model_bin, config_text);
	face_net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	face_net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	//性别模型
	gender_net_ = cv::dnn::readNet(genderModel, genderProto);
	//五官模型
	dlib::deserialize(DlibModel) >> pose_model;
	dlib_detector_ = dlib::get_frontal_face_detector();
}

FaceDetect::~FaceDetect()
{
}

bool FaceDetect::StartGrab()
{
	for (int i = 0; i < 2; ++i)
	{
		cap_.open(i);
		if (cap_.isOpened())
			return true;
	}
	cout << "no cam available" << endl;
	return false;
}

void FaceDetect::GetFrame()
{
	cap_ >> src_;
	cout << "get a frame" << endl;
}

bool FaceDetect::ProcessFace()
{

	if (!GetFace())
		return false;

	//识别性别
	GetGender(roi_face_all_);

	//美化图片
	FaceBeautify(roi_face_all_, roi_face_all_);

	//像素语义分割
	FaceDetectTorch(roi_face_all_);
	Mat output;
	//根据掩膜信息，得到各个部位图, 并对肤色进行赋值
	GetSegments();
	try
	{
		GetFacialFeatures(roi_face_hair_, output);
	}
	catch (...)
	{

	}

	//获取去除眼睛和嘴巴的图片
	Mat tmp = roi_face_only_.clone();

	//通过对标准值进行图层叠加完成实验
	Mat mask(roi_face_only_.rows, roi_face_only_.cols, CV_8UC3, BODY_COLOR);
	Mat dst;
	ApplyMask(MIX_TYPE::COLOR, tmp, mask, dst);

	GetBaldHead(dst, objects_eyes);
	return true;
}


bool FaceDetect::GetFace()
{
	//src判空
	bool face_dectected = false;
	if (src_.empty())
		return false;

	//整体像素值减去平均值（mean）通过缩放系数（scalefactor）对图片像素值进行缩放
	cv::Mat blob_image = blobFromImage(src_, 1.0,
		cv::Size(300, 300),
		cv::Scalar(104.0, 177.0, 123.0), false, false);

	face_net_.setInput(blob_image, "data");
	cv::Mat detection = face_net_.forward("detection_out");

	const int x_padding = 40;
	const int y_padding = 80;
	cv::Mat detection_mat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	//阈值为0.5 超过0.5才会显示
	float confidence_threshold = 0.5;
	for (int i = 0; i < detection_mat.rows; i++) {
		float confidence = detection_mat.at<float>(i, 2);
		if (confidence > confidence_threshold) {
			size_t objIndex = (size_t)(detection_mat.at<float>(i, 1));
			float tl_x = detection_mat.at<float>(i, 3) * src_.cols;
			float tl_y = detection_mat.at<float>(i, 4) * src_.rows;
			float br_x = detection_mat.at<float>(i, 5) * src_.cols;
			float br_y = detection_mat.at<float>(i, 6) * src_.rows;
			//原始ROI
			rect_face_ = cv::Rect((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int)(br_y - tl_y));
			if (rect_face_.area() < 50)
				return false;
			//由于有时候会产生十分奇怪的坐标，故对坐标进行规范化
			if (rect_face_.x > src_.cols || rect_face_.x < 0 || rect_face_.y > src_.rows || rect_face_.x < 0)
			{
				return false;
			}
			//放大后的ROI
			cv::Rect roi;
			roi.x = max(0, rect_face_.x - x_padding);
			roi.y = max(0, rect_face_.y - y_padding);

			roi.width = rect_face_.width + 2 * x_padding;
			if (roi.width + roi.x > src_.cols - 1)
				roi.width = src_.cols - 1 - roi.x;

			roi.height = rect_face_.height + 2 * y_padding;
			if (roi.height + roi.y > src_.rows - 1)
				roi.height = src_.rows - 1 - roi.y;

			roi_face_all_ = src_(roi);

			return true;
		}

	}
	return false;
}

const string path("F:/Beauty/Beauty/Assets/Resources/");
void FaceDetect::Write2Disk()
{
	RemoveBackground(roi_face_hair_);
	resize(roi_face_hair_, roi_face_hair_, Size(170, 267), 0.0, 0.0, 0);

	RemoveBackground(bald_head_);
	resize(bald_head_, bald_head_, Size(170, 267));

	resize(roi_face_all_, roi_face_all_, Size(400, 400), 0.0, 0.0, 0);
	Mat mask(roi_face_all_.size(), roi_face_all_.type(), Scalar(0, 0, 0, 0));
	circle(mask, Point(200, 200), 200, Scalar(255, 255, 255, 255), -1);
	bitwise_and(roi_face_all_, mask, roi_face_all_);

	if (!cur_gender_.empty())
	{
		const string suffix(".png");
		imwrite(path + string("head-2") + suffix, roi_face_hair_);
		imwrite(path + string("face-2") + suffix, bald_head_);
		imwrite(path + string("head-1") + suffix, roi_face_all_);
	}
}


void FaceDetect::ShowSrc()
{
	if (!bald_head_.empty())
	{
		imshow("test", bald_head_);
	}
	else
		cout << "no bald head" << endl;
}

void FaceDetect::Release()
{
	delete this;
}

void FaceDetect::GetGender(const cv::Mat& input)
{
	const cv::String gender_list[] = { "m", "f" };
	//整体像素值减去平均值（mean）通过缩放系数（scalefactor）对图片像素值进行缩放
	Mat face_blob = blobFromImage(input, 1.0, cv::Size(227, 227), cv::Scalar(78.4263377603, 87.7689143744, 114.895847746), false, false);
	gender_net_.setInput(face_blob);

	Mat gender_preds = gender_net_.forward();
	Mat prob_mat = gender_preds.reshape(1, 1);
	//复制
	Mat output = src_.clone();

	Point class_number;
	double class_prob;
	//寻找矩阵(一维数组当作向量, 用Mat定义) 中最小值和最大值的位置.单通道图像
	minMaxLoc(prob_mat, NULL, &class_prob, NULL, &class_number);

	int classidx = class_number.x;
	cv::String gender = gender_list[classidx];
	//在图像上绘制文字
	putText(output, cv::format("gender:%s", gender.c_str()), rect_face_.tl(),
		cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 1, 8);
	cur_gender_ = gender;
}

bool FaceDetect::FaceDetectTorch(const cv::Mat& input)
{
	//判空
	if (input.empty())
		return false;
	Mat image_transformed;
	const int set_size = 224;//网络需要的固定图片长宽大小
	const int multiple = 127; // 转换的倍数大小
	//重设尺寸
	resize(input, image_transformed, Size(set_size, set_size));
	cvtColor(image_transformed, image_transformed, COLOR_BGR2RGB);

	// 3.图像转换为Tensor
	torch::Tensor tensor_image = torch::from_blob(image_transformed.data, { image_transformed.rows, image_transformed.cols,3 }, torch::kByte);
	tensor_image = tensor_image.permute({ 2,0,1 });
	tensor_image = tensor_image.toType(torch::kFloat);
	tensor_image = tensor_image.div(255);
	tensor_image = tensor_image.unsqueeze(0);


	//网络前向计算
	torch::Tensor out_tensor_all = sematic_module_.forward({ tensor_image }).toTensor();
	torch::Tensor out_tensor = out_tensor_all.argmax(1);
	out_tensor = out_tensor.squeeze();

	//mul函数，表示张量中每个元素乘与一个数，clamp表示夹紧，限制在一个范围内输出
	//由于一共就三种标签0 1 2， 所以最终mat输出应该是 0 127 254
	out_tensor = out_tensor.mul(multiple).to(torch::kU8);
	out_tensor = out_tensor.to(torch::kCPU);

	dst_torch_.create(set_size, set_size, CV_8U);
	memcpy((void*)dst_torch_.data, out_tensor.data_ptr(), sizeof(torch::kU8) * out_tensor.numel());

	//resize回原来的大小
	resize(dst_torch_, dst_torch_, Size(input.cols, input.rows), 0.0, 0.0, INTER_NEAREST);

	return true;
}

bool FaceDetect::GetSegments()
{
	if (roi_face_all_.empty() || dst_torch_.empty())
		return false;
	//创建一个图像矩阵的矩阵体，之后该图像只有脸
	roi_face_only_.create(Size(roi_face_all_.cols, roi_face_all_.rows), CV_8UC3);
	//创建一个图像矩阵的矩阵体，之后该图像只有头发和脸
	roi_face_hair_.create(Size(roi_face_all_.cols, roi_face_all_.rows), CV_8UC3);
	//创建一个图像，之后该图像只有头发
	roi_hair_only_.create(Size(roi_face_all_.cols, roi_face_all_.rows), CV_8UC3);
	//设置背景为黑色
	const Vec3b background = { 0, 0, 0 };
	//循环 遍历每个像素
	for (int i = 0; i < dst_torch_.rows; ++i)
	{
		for (int j = 0; j < dst_torch_.cols; ++j)
		{
			auto cur_pixel = dst_torch_.at<uchar>(i, j);
			//如果监测到头发的颜色，有头及脸的图像不做改动，另一张去除头发保持只有脸
			if (cur_pixel == TypeIndex::HAIR)
			{
				roi_face_only_.at<Vec3b>(i, j) = background;
				roi_face_hair_.at<Vec3b>(i, j) = roi_face_all_.at<Vec3b>(i, j);
				roi_hair_only_.at<Vec3b>(i, j) = Vec3b(178, 178, 158);
			}
			//如果监测到脸的颜色，两张图像都保存脸的部分
			else if (cur_pixel == TypeIndex::FACE)
			{
				roi_face_only_.at<Vec3b>(i, j) = roi_face_all_.at<Vec3b>(i, j);
				roi_face_hair_.at<Vec3b>(i, j) = roi_face_all_.at<Vec3b>(i, j);
				roi_hair_only_.at<Vec3b>(i, j) = background;
			}
			//如果是其他地方，通通变为黑色背景
			else
			{
				roi_face_only_.at<Vec3b>(i, j) = background;
				roi_face_hair_.at<Vec3b>(i, j) = background;
				roi_hair_only_.at<Vec3b>(i, j) = background;
			}
		}
	}

	return true;
}

void FaceDetect::FaceBeautify(cv::Mat& input, cv::Mat& output)
{
	Mat dst_grinded(input.size(), input.type());
	FaceGrinding(input, dst_grinded);
	Mat dst_Saturated(input.size(), input.type());
	AdjustSaturation(dst_grinded, dst_Saturated);
	Mat dst_brighted(input.size(), input.type());
	AdjustBrightness(dst_Saturated, dst_brighted);
	output = dst_brighted.clone();
}

void FaceDetect::FaceGrinding(cv::Mat& input, cv::Mat& output, int value1, int value2)
{
	int dx = value1 * 5;    //双边滤波参数之一  
	double fc = value1 * 12.5; //双边滤波参数之一  
	int transparency = 50; //透明度  
	cv::Mat dst;
	//双边滤波  
	bilateralFilter(input, dst, dx, fc, fc);
	dst = (dst - input + 128);
	//高斯模糊  
	GaussianBlur(dst, dst, cv::Size(2 - 1, 2 - 1), 0, 0);
	dst = input + 2 * dst - 255;
	dst = (input * (100 - transparency) + dst * transparency) / 100;
	dst.copyTo(output);
}

void FaceDetect::AdjustSaturation(cv::Mat& input, cv::Mat& output, int saturation, const int max_increment)
{
	float increment = (saturation - 80) * 1.0 / max_increment;

	for (int col = 0; col < input.cols; col++)
	{
		for (int row = 0; row < input.rows; row++)
		{
			// R,G,B 分别对应数组中下标的 2,1,0
			uchar r = input.at<Vec3b>(row, col)[2];
			uchar g = input.at<Vec3b>(row, col)[1];
			uchar b = input.at<Vec3b>(row, col)[0];

			float maxn = max(r, max(g, b));
			float minn = min(r, min(g, b));

			float delta, value;
			delta = (maxn - minn) / 255;
			value = (maxn + minn) / 255;

			float new_r, new_g, new_b;

			if (delta == 0)		 // 差为 0 不做操作，保存原像素点
			{
				output.at<Vec3b>(row, col)[0] = b;
				output.at<Vec3b>(row, col)[1] = g;
				output.at<Vec3b>(row, col)[2] = r;
				continue;
			}

			float light, sat, alpha;
			light = value / 2;

			if (light < 0.5)
				sat = delta / value;
			else
				sat = delta / (2 - value);

			if (increment >= 0)
			{
				if ((increment + sat) >= 1)
					alpha = sat;
				else
				{
					alpha = 1 - increment;
				}
				alpha = 1 / alpha - 1;
				new_r = r + (r - light * 255) * alpha;
				new_g = g + (g - light * 255) * alpha;
				new_b = b + (b - light * 255) * alpha;
			}
			else
			{
				alpha = increment;
				new_r = light * 255 + (r - light * 255) * (1 + alpha);
				new_g = light * 255 + (g - light * 255) * (1 + alpha);
				new_b = light * 255 + (b - light * 255) * (1 + alpha);
			}
			output.at<Vec3b>(row, col)[0] = new_b;
			output.at<Vec3b>(row, col)[1] = new_g;
			output.at<Vec3b>(row, col)[2] = new_r;
		}
	}
}

void FaceDetect::AdjustBrightness(cv::Mat& input, cv::Mat& output, float alpha, float beta)
{
	int height = input.rows;//求出src的高
	int width = input.cols;//求出input的宽
	output = cv::Mat::zeros(input.size(), input.type());  //这句很重要，创建一个与原图一样大小的空白图片              
	//循环操作，遍历每一列，每一行的元素
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			if (input.channels() == 3)//判断是否为3通道图片
			{
				//将遍历得到的原图像素值，返回给变量b,g,r
				float b = input.at<Vec3b>(row, col)[0];//nlue
				float g = input.at<Vec3b>(row, col)[1];//green
				float r = input.at<Vec3b>(row, col)[2];//red
				//开始操作像素，对变量b,g,r做改变后再返回到新的图片。
				output.at<Vec3b>(row, col)[0] = cv::saturate_cast<uchar>(b * alpha + beta);
				output.at<Vec3b>(row, col)[1] = cv::saturate_cast<uchar>(g * alpha + beta);
				output.at<Vec3b>(row, col)[2] = cv::saturate_cast<uchar>(r * alpha + beta);
			}
			else if (input.channels() == 1)//判断是否为单通道的图片
			{

				float v = input.at<uchar>(row, col);
				output.at<uchar>(row, col) = cv::saturate_cast<uchar>(v * alpha + beta);
			}
		}
	}
}

void FaceDetect::ApplyMask(const std::string& mask_type, const cv::Mat& input, const cv::Mat& mask, cv::Mat& dst)
{
	MixerFactory m_factory;
	auto mixer = m_factory.GetMixer(mask_type);
	mixer->Mix(input, mask, dst);

	double alpha = 0.7;
	addWeighted(roi_face_only_, alpha, dst, 1 - alpha, 0, dst);

	cv::imshow("dst", dst);
}

void FaceDetect::GetBaldHead(cv::Mat& input, std::vector<cv::Rect>& eyes)
{
	try
	{

		if (input.empty() || eyes.empty())
		{
			cout << "error in GetBaldHead" << endl;
			return;
		}


		//确定左右眼的中心
		Point lefteye_center, righteye_center;
		if (eyes[0].x < eyes[1].x)
		{
			lefteye_center = SimpleMath::GetMidpt(eyes[0].tl(), eyes[0].br());
			righteye_center = SimpleMath::GetMidpt(eyes[1].tl(), eyes[1].br());
		}
		else
		{
			lefteye_center = SimpleMath::GetMidpt(eyes[1].tl(), eyes[1].br());
			righteye_center = SimpleMath::GetMidpt(eyes[0].tl(), eyes[0].br());
		}
		//生成一幅
		Mat tmp = input.clone();

		Point2d center = SimpleMath::GetMidpt(lefteye_center, righteye_center);
		Point2d virtual_top = SimpleMath::GetRotatedVecRad(center, Point2d((double)righteye_center.x, (double)righteye_center.y), -M_PI / 2, 1.2);
		Point2d virtual_bottom = SimpleMath::GetRotatedVecRad(center, Point2d((double)righteye_center.x, (double)righteye_center.y), M_PI / 2, 1.2);
		Point2d virtual_left = SimpleMath::GetRotatedVecRad(center, Point2d((double)lefteye_center.x, (double)lefteye_center.y), 0.01, 1.2);
		Point2d virtual_right = SimpleMath::GetRotatedVecRad(center, Point2d((double)righteye_center.x, (double)righteye_center.y), 0.01, 1.2);

		std::vector<Point2d> virtual_pts = { virtual_top, virtual_bottom, virtual_left, virtual_right };

		/*circle(tmp, center, 3, Scalar(255, 0, 0), -1);
		circle(tmp, virtual_top, 3, Scalar(0, 255, 255), -1);
		circle(tmp, lefteye_center, 3, Scalar(0, 255, 255), -1);
		circle(tmp, righteye_center, 3, Scalar(0, 255, 255), -1);*/


		double short_axis = SimpleMath::GetLineLen(center, virtual_top) * 2.0;
		double long_axis = SimpleMath::GetLineLen(center, virtual_left) * 2.2;

		std::vector<Point> ellipes_verti;
		ellipse2Poly((Point)center, Size(long_axis, short_axis), 0, 180 + 15, 180 + 165, 1, ellipes_verti);


		Mat mask(input.size(), CV_8UC1, Scalar(0));

		for (int i = 0; i < ellipes_verti.size() - 1; ++i)
		{
			//line(tmp, ellipes_verti[i], ellipes_verti[i + 1], Scalar(123, 45, 78), 2);
			cv::line(mask, ellipes_verti[i], ellipes_verti[i + 1], Scalar(255), 2);
		}

		Point2d br_padding = ellipes_verti.front(); br_padding.y += input.rows - br_padding.y - 1;
		Point2d bl_padding = ellipes_verti.back(); bl_padding.y += input.rows - bl_padding.y - 1;

		/*line(tmp, ellipes_verti.front(), br_padding, Scalar(123, 45, 78), 2);
		line(tmp, ellipes_verti.back(), bl_padding, Scalar(123, 45, 78), 2);
		line(tmp, bl_padding, br_padding, Scalar(123, 45, 78), 2);*/

		cv::line(mask, ellipes_verti.front(), br_padding, Scalar(255), 2);
		cv::line(mask, ellipes_verti.back(), bl_padding, Scalar(255), 2);
		cv::line(mask, bl_padding, br_padding, Scalar(255), 2);

		Mat tmp_hair = roi_hair_only_.clone();

		//染发
		for (int x = 0; x < tmp_hair.cols; ++x)
		{
			for (int y = 0; y < tmp_hair.rows; ++y)
			{
				Vec3b& pixel_color = tmp_hair.at<Vec3b>(y, x);
				if (pixel_color[0] != 0 && pixel_color[1] != 0 && pixel_color[2] != 0)
				{
					pixel_color[0] = skin_color_[0];
					pixel_color[1] = skin_color_[1];
					pixel_color[2] = skin_color_[2];
				}
			}
		}

		cout << "tmp row cols" << tmp.rows << " " << tmp.cols << endl;
		cout << "tmp_hair" << tmp_hair.rows << " " << tmp_hair.cols << endl;

		tmp += tmp_hair;
		cout << "beacon" << endl;

		FillContour(mask, mask);

		//将图片转换为三通道执行与运算
		cvtColor(mask, mask, COLOR_GRAY2BGR);
		bitwise_and(tmp, mask, bald_head_);

		//cv::imshow("mask", mask);
		//cv::imshow("bald", bald_head_);


	}
	catch (exception& e)
	{
		cout << e.what() << endl;
		//do nothing
	}

}

void FaceDetect::GetFacialFeatures(cv::Mat& input, cv::Mat& output)
{
	if (input.empty())
	{
		cout << "error in Facial Features, empty input" << endl;
		return;
	}
	// 转换输入数据到cimg

	Mat img = input.clone();
	dlib::cv_image<dlib::bgr_pixel> cimg(img);

	std::vector<dlib::rectangle> faces = dlib_detector_(cimg);

	if (faces.empty())
		return;
	// Find the pose of input
	dlib::full_object_detection current_pose = pose_model(cimg, faces[0]);

	mask_eye_ = cv::Mat::zeros(input.size(), CV_8UC1);
	mask_lip_ = cv::Mat::zeros(input.size(), CV_8UC1);

	if (current_pose.num_parts() == 68)
	{

		vector<Point> left_eye_points, right_eye_points;
		//将左眼区域进行连接
		for (int i = 36; i < 41; i++)
		{
			if (current_pose.part(i).size() == 0 || current_pose.part(i + 1).size() == 0)
			{
				continue;
			}
			cv::line(mask_eye_, cvPoint(current_pose.part(i).x(), current_pose.part(i).y()), cvPoint(current_pose.part(i + 1).x(), current_pose.part(i + 1).y()), Scalar(255), 1);
			left_eye_points.emplace_back(current_pose.part(i).x(), current_pose.part(i).y());
		}
		cv::line(mask_eye_, cvPoint(current_pose.part(36).x(), current_pose.part(36).y()), cvPoint(current_pose.part(41).x(), current_pose.part(41).y()), Scalar(255), 1);
		auto rrec1 = minAreaRect(left_eye_points);
		objects_eyes.push_back(rrec1.boundingRect());

		//将右眼区域进行连接
		for (int i = 42; i < 47; i++)
		{
			if (current_pose.part(i).size() == 0 || current_pose.part(i + 1).size() == 0)
			{
				continue;
			}
			cv::line(mask_eye_, cvPoint(current_pose.part(i).x(), current_pose.part(i).y()), cvPoint(current_pose.part(i + 1).x(), current_pose.part(i + 1).y()), Scalar(255), 1);
			right_eye_points.emplace_back(current_pose.part(i).x(), current_pose.part(i).y());
		}
		cv::line(mask_eye_, cvPoint(current_pose.part(42).x(), current_pose.part(42).y()), cvPoint(current_pose.part(47).x(), current_pose.part(47).y()), Scalar(255), 1);
		auto rrec2 = minAreaRect(right_eye_points);
		objects_eyes.push_back(rrec2.boundingRect());

		//将嘴巴区域进行连接
		for (int i = 48; i < 60; i++)
		{
			if (current_pose.part(i).size() == 0 || current_pose.part(i + 1).size() == 0)
			{
				continue;
			}
			cv::line(mask_lip_, cvPoint(current_pose.part(i).x(), current_pose.part(i).y()), cvPoint(current_pose.part(i + 1).x(), current_pose.part(i + 1).y()), Scalar(255), 1);
		}
		cv::line(mask_lip_, cvPoint(current_pose.part(48).x(), current_pose.part(48).y()), cvPoint(current_pose.part(60).x(), current_pose.part(60).y()), Scalar(255), 1);

		Point nose_tip(current_pose.part(30).x(), current_pose.part(30).y());
		skin_color_ = input.at<Vec3b>(nose_tip.y - 50, nose_tip.x);
	}
	FillContour(mask_eye_, mask_eye_, TypeIndex::EYES);
	FillContour(mask_lip_, mask_lip_, TypeIndex::LIPS);
}

void FaceDetect::FillContour(cv::Mat& input, cv::Mat& output, const uchar mask_val)
{
	if (input.empty())
	{
		return;
	}
	Mat img;

	//对轮廓图进行填充
	if (input.type() != CV_8UC1)
	{
		cvtColor(input, img, COLOR_BGR2GRAY);
	}
	else
	{
		img = input.clone();
	}
	output = Mat::zeros(img.size(), img.type());


	std::vector<std::vector<Point>> contours;
	findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	//对图中的所有轮廓进行填充
	for (const auto& contour : contours)
	{
		Mat tmp = Mat::zeros(img.size(), img.type());

		Rect b_rect = boundingRect(contour);
		b_rect.x = max(0, b_rect.x - 1);
		b_rect.y = max(0, b_rect.y - 1);
		b_rect.width += 2;
		if (b_rect.width + b_rect.x > img.cols)
			b_rect.width = img.cols - 1 - b_rect.x;
		if (b_rect.height + b_rect.y > img.rows)
			b_rect.height = img.rows - 1 - b_rect.y;


		auto fun_in_rect = [&b_rect](int x, int y)
		{
			return (x >= b_rect.x && x <= b_rect.x + b_rect.width && y >= b_rect.y && y <= b_rect.y + b_rect.height);
		};
		std::queue<Point> neighbor_queue;
		neighbor_queue.emplace(b_rect.x, b_rect.y);
		tmp.at<uchar>(b_rect.y, b_rect.x) = 128;

		while (!neighbor_queue.empty())
		{
			//从队列取出种子点，获取其4邻域坐标点
			auto seed = neighbor_queue.front();
			neighbor_queue.pop();

			std::vector<Point> pts;
			pts.emplace_back(seed.x, (seed.y - 1));
			pts.emplace_back(seed.x, (seed.y + 1));
			pts.emplace_back((seed.x - 1), seed.y);
			pts.emplace_back((seed.x + 1), seed.y);

			for (auto& pt : pts)
			{
				if (fun_in_rect(pt.x, pt.y) && tmp.at<uchar>(pt.y, pt.x) == 0 && img.at<uchar>(pt.y, pt.x) == 0)
				{
					//将矩形范围内且灰度值为0的可连通坐标点添加到队列
					neighbor_queue.push(pt);
					tmp.at<uchar>(pt.y, pt.x) = 128;
				}
			}

		}


		for (int i = b_rect.y; i < b_rect.y + b_rect.height; i++)
		{
			for (int j = b_rect.x; j < b_rect.x + b_rect.width; j++)
			{
				if (tmp.at<uchar>(i, j) == 0)
				{
					output.at<uchar>(i, j) = mask_val;
				}
			}
		}
	}
	return;
}

void FaceDetect::RemoveBackground(cv::Mat& img)
{
	if (img.channels() != 4)
	{
		cv::cvtColor(img, img, cv::COLOR_BGR2BGRA);


		for (int y = 0; y < img.rows; ++y)
		{
			for (int x = 0; x < img.cols; ++x)
			{
				cv::Vec4b& pixel = img.at<cv::Vec4b>(y, x);
				if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0)
				{
					pixel[0] = 255;
					pixel[1] = 255;
					pixel[2] = 255;
					pixel[3] = 0;
				}

			}
		}
	}
	else
	{
		img = img.clone();
		for (int y = 0; y < img.rows; ++y)
		{
			for (int x = 0; x < img.cols; ++x)
			{
				cv::Vec4b& pixel = img.at<cv::Vec4b>(y, x);
				if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0)
				{
					pixel[0] = 255;
					pixel[1] = 255;
					pixel[2] = 255;
					pixel[3] = 0;
				}

			}
		}
	}
}
