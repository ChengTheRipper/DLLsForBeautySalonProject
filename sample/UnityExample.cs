public class CallCpp : MonoBehaviour
{

    [DllImport("UnityCall", EntryPoint = "Add")]
    public static extern int Add(int x, int y);

    void Start()
    {
        Debug.Log(Add(1, 2));
    }
}
