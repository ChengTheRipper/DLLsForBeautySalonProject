using System;
using System.Runtime.InteropServices;

namespace absff
{
    public class CallCpp
    {
        [DllImport("FaceDetect.dll")]
        public static extern IntPtr GetFD_Sharp();
        [DllImport("FaceDetect.dll")]
        public static extern bool StartGrab(IntPtr fdpt);
        [DllImport("FaceDetect.dll")]
        public static extern bool ProcessFace(IntPtr fdpt);
        [DllImport("FaceDetect.dll")]
        public static extern void GetFrame(IntPtr fdpt);
        [DllImport("FaceDetect.dll")]
        public static extern void Write2Disk(IntPtr fdpt);
        [DllImport("FaceDetect.dll")]
        public static extern void ShowSrc(IntPtr fdpt);
        [DllImport("FaceDetect.dll")]
        public static extern void Release(IntPtr fdpt);

        public CallCpp()
        {
            _fdpt = GetFD_Sharp();
        }

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

        ~CallCpp()
        {
            Release(_fdpt);
        }
        private IntPtr _fdpt;
    }
    class entry
    {
        static void Main()
        {
            CallCpp c1 = new CallCpp();
            c1.test();
        }
    }
}
