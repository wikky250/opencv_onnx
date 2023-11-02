#include <assert.h>
#include <vector>
#include<ctime>
#include <onnxruntime_cxx_api.h>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
class U2NetModel
{
private:
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<const char*>input_node_names;
    std::vector<const char*>output_node_names;
    std::vector<int64_t> input_node_dims;
    std::vector<int64_t> output_node_dims;
public:
    U2NetModel(const wchar_t* onnx_model_path);
    float* predict(std::vector<float>input_data, int batch_size = 1);
    std::vector<float> predict(std::vector<float>& input_data, int batch_size = 1, int index = 0);
    cv::Mat predict(cv::Mat& input_tensor, int batch_size = 1, int index = 0);
};
U2NetModel::U2NetModel(const wchar_t* onnx_model_path) :session(nullptr), env(nullptr)
{
    //初始化环境，每个进程一个环境,环境保留了线程池和其他状态信息
    this->env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "u2net");
    //初始化Session选项
    Ort::SessionOptions session_options;
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // 创建Session并把模型加载到内存中
    this->session = Ort::Session(env, onnx_model_path, session_options);
    //输入输出节点数量和名称
}
float* U2NetModel::predict(std::vector<float>input_tensor_values, int batch_size)
{
    this->input_node_dims[0] = batch_size;
    auto input_tensor_size = input_tensor_values.size();
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), & input_tensor, 1, output_node_names.data(), 1);
    assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    return floatarr;
}

std::vector<float> U2NetModel::predict(std::vector<float>& input_tensor_values, int batch_size, int index)
{
    //this->input_node_dims[0] = batch_size;
    //this->output_node_dims[0] = batch_size;
    float* floatarr = nullptr;
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();
    for (int i = 0; i < num_input_nodes; i++)
    {
        Ort::AllocatedStringPtr input_name_Ptr = session.GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name_Ptr.get());
        printf("Input %d : name=%s\n", i, input_node_names[i]);
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        this->input_node_dims = tensor_info.GetShape();

    }
    for (int i = 0; i < num_output_nodes; i++)
    {
        Ort::AllocatedStringPtr output_name_Ptr = session.GetOutputNameAllocated(i, allocator);
        this->output_node_names.push_back(output_name_Ptr.get());
        printf("Output %d : name=%s\n", i, output_node_names[i]);
        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        this->output_node_dims = tensor_info.GetShape();
    }
    //try
    {
        if (index != -1)
        {
            output_node_names = { this->output_node_names[index] };
        }
        else
        {
            output_node_names = this->output_node_names;
        }
        //this->input_node_dims[0] = batch_size;
        auto input_tensor_size = input_tensor_values.size();

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);

        printf("Input %d : name=%s\n", 0, input_node_names[0]);
        auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), & input_tensor, 1, output_node_names.data(), 1);
        assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
        floatarr = output_tensors.front().GetTensorMutableData<float>();
    }
   
    int64_t output_tensor_size = 1;
    for (auto& it : this->output_node_dims)
    {
        output_tensor_size *= it;
    }
    std::vector<float>results(output_tensor_size);
    for (unsigned i = 0; i < output_tensor_size; i++)
    {
        results[i] = floatarr[i];
    }
    return results;
}
cv::Mat U2NetModel::predict(cv::Mat& input_tensor, int batch_size, int index)
{
    int input_tensor_size = input_tensor.cols * input_tensor.rows * 3;
    std::size_t counter = 0;//std::vector空间一次性分配完成，避免过多的数据copy
    std::vector<float>input_data(input_tensor_size);
    std::vector<float>output_data;
    try
    {
        for (unsigned k = 0; k < 3; k++)
        {
            for (unsigned i = 0; i < input_tensor.rows; i++)
            {
                for (unsigned j = 0; j < input_tensor.cols; j++)
                {
                    input_data[counter++] = static_cast<float>(input_tensor.at<cv::Vec3b>(i, j)[k]) / 255.0;
                }
            }
        }
    }
    catch (cv::Exception& e)
    {
        printf(e.what());
    }
    try
    {
        output_data = this->predict(input_data,1,0);
    }
    catch (Ort::Exception& e)
    {
        std::cout << e.what()<<std::endl;
    }

    cv::Mat out0 = cv::Mat::zeros(512,512, CV_8UC1);
    cv::Mat out1 = cv::Mat::zeros(512,512, CV_8UC1);

    for (int i = 0; i < 512; i++)
    {
        for (int j = 0; j < 512; j++)
        {
            float a0 = output_data[0 * 512 * 512 + i * 512 + j];
            float a1 = output_data[1 * 512 * 512 + i * 512 + j];
            if (std::max(a0, a1) == a0) out0.at<uchar>(i, j) = 255;
            if (std::max(a0, a1) == a1) out1.at<uchar>(i, j) = 255;
            //if (max(max(max(max(max(max(max(max(a0, a1), a2), a3), a4), a5), a6), a7), a8) == a0) out0.at<uchar>(i, j) = 255;
            //if (max(max(max(max(max(max(max(max(a0, a1), a2), a3), a4), a5), a6), a7), a8) == a1) out1.at<uchar>(i, j) = 255;
            //if (max(max(max(max(max(max(max(max(a0, a1), a2), a3), a4), a5), a6), a7), a8) == a2) out2.at<uchar>(i, j) = 255;

        }
    }
    cv::Mat output_tensor(output_data);
    output_tensor = output_tensor.reshape(1, { 512,512 }) ;
    std::cout << output_tensor.rows << " " << output_tensor.cols << "fuck" << std::endl;
    return output_tensor;
}
int main(int argc, char* argv[])
{
    U2NetModel model(L"D:/test.onnx");
    cv::Mat image = cv::imread("E:/image/2023_10_09_13_04_38_413/DA1346115/ori/images/2_0_ori.tiff");
    cv::resize(image, image, { 512,512 }, 0.0, 0.0, cv::INTER_CUBIC);//调整大小到320*320

    cv::imshow("image", image);                                     //打印原图片
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);                  //BRG格式转化为RGB格式
    auto result = model.predict(image);                               //模型预测
    cv::imshow("result", result);                                   //打印结果
    cv::waitKey(0);
    ////记录程序运行时间
    //auto start_time = clock();
    ////初始化环境，每个进程一个环境
    ////环境保留了线程池和其他状态信息
    //Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    ////初始化Session选项
    //Ort::SessionOptions session_options;
    //session_options.SetIntraOpNumThreads(1);
    //// Available levels are
    //// ORT_DISABLE_ALL -> To disable all optimizations
    //// ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
    //// ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
    //// ORT_ENABLE_ALL -> To Enable All possible opitmizations
    //session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    ////*************************************************************************
    //// 创建Session并把模型加载到内存中
    //const wchar_t* model_path = L"D:/test.onnx";

    //printf("Using Onnxruntime C++ API\n");
    //Ort::Session session(env, model_path, session_options);

    ////*************************************************************************
    ////打印模型的输入层(node names, types, shape etc.)
    //Ort::AllocatorWithDefaultOptions allocator;

    ////输出模型输入节点的数量
    //size_t num_input_nodes = session.GetInputCount();
    //size_t num_output_nodes = session.GetOutputCount();
    //std::vector<const char*> input_node_names;
    //std::vector<const char*> output_node_names;
    //std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
    //// Otherwise need vector<vector<>>

    //printf("Number of inputs = %zu\n", num_input_nodes);
    ////迭代所有的输入节点
    //for (int i = 0; i < num_input_nodes; i++) {
    //    //输出输入节点的名称
    //    Ort::AllocatedStringPtr input_name_Ptr = session.GetInputNameAllocated(i, allocator);
    //    input_node_names.push_back(input_name_Ptr.get());
    //    printf("Input %d : name=%s\n", i, input_node_names[i]);

    //    // 输出输入节点的类型
    //    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    //    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    //    ONNXTensorElementDataType type = tensor_info.GetElementType();
    //    printf("Input %d : type=%d\n", i, type);

    //    input_node_dims = tensor_info.GetShape();
    //    //输入节点的打印维度
    //    printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
    //    //打印各个维度的大小
    //    for (int j = 0; j < input_node_dims.size(); j++)
    //        printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
    //    //batch_size=1
    //    input_node_dims[0] = 1;
    //}
    ////打印输出节点信息，方法类似
    //for (int i = 0; i < num_output_nodes; i++)
    //{

    //    Ort::AllocatedStringPtr output_name_Ptr = session.GetOutputNameAllocated(i, allocator);
    //    output_node_names.push_back(output_name_Ptr.get());
    //    printf("Input %d : name=%s\n", i, output_node_names[i]);

    //    Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
    //    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    //    ONNXTensorElementDataType type = tensor_info.GetElementType();
    //    printf("Output %d : type=%d\n", i, type);
    //    auto output_node_dims = tensor_info.GetShape();
    //    printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
    //    for (int j = 0; j < input_node_dims.size(); j++)
    //        printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
    //}

    ////*************************************************************************
    //// 使用样本数据对模型进行评分，并检验出入值的合法性
    //size_t input_tensor_size = 3 * 512 * 512;  // simplify ... using known dim values to calculate size
    //// use OrtGetTensorShapeElementCount() to get official size!

    //std::vector<float> input_tensor_values(input_tensor_size);

    //// 初始化一个数据（演示用,这里实际应该传入归一化的数据）
    //for (unsigned int i = 0; i < input_tensor_size; i++)
    //    input_tensor_values[i] = (float)i / (input_tensor_size + 1);

    //// 为输入数据创建一个Tensor对象
    //try
    //{
    //    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    //    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
    //    //assert(input_tensor.IsTensor());

    //    // 推理得到结果
    //    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), & input_tensor, 1, output_node_names.data(), 1);
    //    assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

    //    // Get pointer to output tensor float values
    //    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    //    printf("Number of outputs = %d\n", output_tensors.size());
    //}
    //catch (Ort::Exception& e)
    //{
    //    printf(e.what());
    //}
    //auto end_time = clock();
    //printf("Proceed exit after %.2f seconds\n", static_cast<float>(end_time - start_time) / CLOCKS_PER_SEC);
    //printf("Done!\n");
    return 0;
}