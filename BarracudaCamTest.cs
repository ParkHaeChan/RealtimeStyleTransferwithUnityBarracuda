using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using UnityEngine.UI;
//using UnityEngine.Events;       // UnityAction
//using UnityEngine.Rendering;    // AsyncGPUReadback

public class BarracudaCamTest : MonoBehaviour
{
    public NNModel modelAsset;
    private Model m_RuntimeModel;
    private IWorker m_Worker;

    RenderTexture m_InputTexture; //get webcam input as texture
    public Texture2D m_StyleTexture;

    public RenderTexture m_ResultTexture;

    // Webcam component
    AspectRatioFitter fitter;
    WebCamTexture webcamTexture;
    bool ratioSet;

    public RawImage originalImage;

    const int IMAGE_SIZE = 300;

    // Start is called before the first frame update
    void Start()
    {
        m_RuntimeModel = ModelLoader.Load(modelAsset);
        m_Worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, m_RuntimeModel);
        fitter = GetComponent<AspectRatioFitter>();
        InitWebCam();
    }

    // Update is called once per frame
    void Update()
    {
        raw_output();
    }

    void LateUpdate()
    {
        originalImage.texture = webcamTexture;
    }

    void raw_output()
    {
        var channelCount = 3;
        var inputs = new Dictionary<string, Tensor>();
        
        Converter(webcamTexture);
        inputs["input"] = new Tensor(m_InputTexture, channelCount);
        inputs["input.1"] = new Tensor(m_StyleTexture, channelCount);
        m_Worker.Execute(inputs);
        Tensor output = m_Worker.PeekOutput();
        output.ToRenderTexture(m_ResultTexture, 0, 0, new Vector4(1f, 1f, 1f, 1f),
        new Vector4(0f, 0f, 0f, 0f), null);

        output.Dispose();
    }

    void SetAspectRatio() {
        fitter.aspectRatio = (float)webcamTexture.width / (float)webcamTexture.height;
    }

    void InitWebCam() {
        string camName = WebCamTexture.devices[0].name;
        webcamTexture = new WebCamTexture(camName, Screen.width, Screen.height, 30);
        Debug.Log("width: " + Screen.width + "height: " + Screen.height);
        originalImage.texture = webcamTexture;
        webcamTexture.Play();
    }

    void Converter(WebCamTexture webCamTexture)
    {
        m_InputTexture = new RenderTexture(IMAGE_SIZE, IMAGE_SIZE, 0, RenderTextureFormat.ARGB32);
        Vector2 scale = new Vector2(1, 1);
        Vector2 offset = Vector2.zero;

        scale.x = (float)webCamTexture.height / (float)webCamTexture.width;
        offset.x = (1 - scale.x) / 2f;
        Graphics.Blit(webCamTexture, m_InputTexture, scale, offset);
    }
}