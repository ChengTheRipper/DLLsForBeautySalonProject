DLLs

# 1. 如何使用FaceDetect.dll

## 1.1 环境搭建

经过研究发现，C#调用dll只需要调用将DLL放在和文件执行文件相同的地方就可，不需要再配置头文件，静态库之类的，比较方便

![image-20210226105333620](D:\CPP_Projects\BeautySalon\DLLs\image-20210226105333620.png)

#### 模型文件和其他C++ DLL

此外，该dll需要大量的深度学习模型文件进行辅助，所有的文件都放在了ResourceTable文件夹中，其他的C++ dll 只需要放在和FaceDetect.dll相同的文件夹即可运行

## 1.2 例程讲解

例程在Sample文件夹中，其中由于我Unity 还不是很会，Unity例子我只放了一个标准的Unity调用c++ dll 例子，实际测试需要我在学习完Unity后才能够进一步使用，其中**Program.cs**包含了一个标准的C#例子

```c#
using System;
using System.Runtime.InteropServices;

namespace absff
{
    public class CallCpp
    {
        //引入DLL接口函数
        
        //获取一个图像处理类
        [DllImport("FaceDetect.dll")]
        public static extern IntPtr GetFD_Sharp();
        //开始采集
        [DllImport("FaceDetect.dll")]
        public static extern bool StartGrab(IntPtr fdpt);
        //处理脸部函数
        [DllImport("FaceDetect.dll")]
        public static extern bool ProcessFace(IntPtr fdpt);
        //采集当前帧
        [DllImport("FaceDetect.dll")]
        public static extern void GetFrame(IntPtr fdpt);
        //写入文件到磁盘
        [DllImport("FaceDetect.dll")]
        public static extern void Write2Disk(IntPtr fdpt);
        //显示秃头图片
        [DllImport("FaceDetect.dll")]
        public static extern void ShowSrc(IntPtr fdpt);
        //释放这个类
        [DllImport("FaceDetect.dll")]
        public static extern void Release(IntPtr fdpt);
		
        //构造函数
        public CallCpp()
        {
            _fdpt = GetFD_Sharp();
        }
		//测试函数
        public void test()
        {
            if (!StartGrab(_fdpt))
                return;
            while (true)
            {
                GetFrame(_fdpt);
                ProcessFace(_fdpt);
                ShowSrc(_fdpt);
            }
        }
		//析构函数
        ~CallCpp()
        {
            Release(_fdpt);
        }
        private IntPtr _fdpt;
    }
    class entry
    {
        //主函数
        static void Main()
        {
            CallCpp c1 = new CallCpp();
            c1.test();
        }
    }
}

```

Unity例子在DLL导入中似乎有变换，应该不用加文件名后缀

```C#
public class CallCpp : MonoBehaviour { //需要继承
    [DllImport("UnityCall")]//尾部无后缀
	public static extern IntPtr GenerateEcho(float x);
	
	[DllImport("UnityCall")]
	public static extern void setX(IntPtr echo,float x);
	
	[DllImport("UnityCall")]
	public static extern float getX(IntPtr echo);
	
	[DllImport("UnityCall")]
	public static extern void ReleaseEcho(IntPtr echo);
    
    private IntPtr echo;
    
    // Use this for initialization
	void Start () {
		this.echo = GenerateEcho(3.0f);
		Debug.Log(getX(echo));
		setX(echo,2.5f);
		Debug.Log(getX(echo));
		ReleaseEcho(echo);
	}
}

```

# 2. 编译好的C#文件

百度网盘地址： 链接：https://pan.baidu.com/s/18PtKp1MAUqdVSAZPaWIJRQ 
提取码：m0rn 
