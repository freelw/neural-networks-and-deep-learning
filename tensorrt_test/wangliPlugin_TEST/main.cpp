#include <unistd.h>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvUtils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <assert.h>
#include <string.h>
#include <cmath>

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cout << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

using namespace nvuffparser;
using namespace nvinfer1;

void sleepForever() {
    while (true) {
        usleep(50000000);
    }
}

class Logger : public ILogger {
    void log(Severity severity, const char* msg) override {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} _gLogger;
class MyPlugin: public IPluginExt
{
public:
    MyPlugin()
    {   
        std::cout << "construct " << std::endl;
    }


    ~MyPlugin()
    {
    }

    int getNbOutputs() const override
    {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        std::cout << "index : " << index << std::endl;
        std::cout << "nbInputDims : " << nbInputDims << std::endl;
        for (auto i = 0; i < nbInputDims; ++ i) {
            std::cout << "[" << i << "] ";
            for (auto j = 0; j < inputs[i].nbDims; ++ j) {
                std::cout << inputs[i].d[j] << " ";
            }
            std::cout << std::endl;
        }

        return Dims3(1, 1, 10);
    }

    bool supportsFormat(DataType type, PluginFormat format) const override { return true; };

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override
    {
        std::cout << "configureWithFormat " << std::endl;
    }

    int initialize() override
    {
        std::cout << "init " << std::endl;
        return 0;
    }

    virtual void terminate() override
    {
    }

    virtual size_t getWorkspaceSize(int maxBatchSize) const override
    {
        return 0;
    }

    virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override
    {
        std::cout << "batchSize : " << batchSize << std::endl;
        float tmp[10];
        float res[10];
        CHECK(cudaMemcpy(tmp, (char*)inputs[0], 10*sizeof(float), cudaMemcpyDeviceToHost));
        float sum = 0;
        for (auto i  = 0; i < 10; ++ i) {
            std::cout << tmp[i] << " ";
            sum += exp(tmp[i]);
        }
        for (auto i  = 0; i < 10; ++ i) {
            res[i] = exp(tmp[i]) / sum;
        }
        std::cout << std::endl;
        CHECK(cudaMemcpy(outputs[0], res, 10*sizeof(float), cudaMemcpyHostToDevice));
        return 0;
    }

    virtual size_t getSerializationSize() override
    {
        return 0;
    }

    virtual void serialize(void* buffer) override
    {
    }
};

class PluginFactory : public nvuffparser::IPluginFactoryExt
{
public:
    bool isPlugin(const char* name) override
    {
        return isPluginExt(name);
    }

    bool isPluginExt(const char* name) override
    {
        return !strcmp(name, "_my_soft_max");
    }

    virtual nvinfer1::IPlugin* createPlugin(const char *layerName, const nvinfer1::Weights *weights, int nbWeights, const FieldCollection fc) override       
    {
        return new MyPlugin();
    }
};

ICudaEngine* initEngine() {
    PluginFactory pluginFactory;
    IBuilder* builder = createInferBuilder(_gLogger);
    assert(builder != nullptr);
    INetworkDefinition* network = builder->createNetwork();
    auto parser = createUffParser();
    parser->registerInput("input_x", Dims3(1, 1, 784), UffInputOrder::kNCHW);
    parser->registerOutput("output_y");
    parser->setPluginFactoryExt(&pluginFactory);
    if (!parser->parse("../restore_test.uff", *network, nvinfer1::DataType::kFLOAT)) {
        std::cerr << "Failure while parsing UFF file" << std::endl;
        exit(-1);
    } else {
        std::cout << "load uff succ" << std::endl;
    }
    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(1 << 10);
    builder->setFp16Mode(0);
    builder->setInt8Mode(0);
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine) {
        std::cerr << "Unable to create engine" << std::endl;
        exit(-1);
    } else {
        std::cout << "create engine done." << std::endl;
    }
    network->destroy();
    builder->destroy();
    parser->destroy();
    return engine;
}

inline int64_t volume(const Dims& d)
{
    int64_t v = 1;
    for (int64_t i = 0; i < d.nbDims; i++)
        v *= d.d[i];
    return v;
}

inline unsigned int elementSize(DataType t)
{
    switch (t)
    {
    case DataType::kINT32:
        // Fallthrough, same as kFLOAT
    case DataType::kFLOAT: return 4;
    case DataType::kHALF: return 2;
    case DataType::kINT8: return 1;
    }
    assert(0);
    return 0;
}

std::vector<std::pair<int64_t, DataType>>
calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<int64_t, DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }

    return sizes;
}

void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}


inline void readPGMFile(const std::string& fileName, uint8_t* buffer, int inH, int inW)
{
    std::ifstream infile(fileName, std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    std::string magic, h, w, max;
    infile >> magic >> h >> w >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(buffer), inH * inW);
}

void* createMnistCudaBuffer(int64_t eltCount, DataType dtype, int num)
{
    /* in that specific case, eltCount == INPUT_H * INPUT_W */
    assert(eltCount == 784);
    assert(elementSize(dtype) == sizeof(float));

    size_t memSize = eltCount * elementSize(dtype);
    float* inputs = new float[eltCount];

    /* read PGM file */
    uint8_t fileData[784];
    readPGMFile("../pgm_data/" + std::to_string(num) + ".pgm", fileData, 28, 28);

    /* display the number in an ascii representation */
    std::cout << "Input:\n";
    for (int i = 0; i < eltCount; i++)
        std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % 28) ? "" : "\n");
    std::cout << std::endl;

    /* initialize the inputs buffer */
    for (int i = 0; i < eltCount; i++)
        inputs[i] = 1.0 - float(fileData[i]) / 255.0;

    void* deviceMem = safeCudaMalloc(memSize);
    CHECK(cudaMemcpy(deviceMem, inputs, memSize, cudaMemcpyHostToDevice));

    delete[] inputs;
    return deviceMem;
}

int execute(ICudaEngine &engine)
{
    IExecutionContext* context = engine.createExecutionContext();
    int batchSize = 1;
    int nbBindings = engine.getNbBindings();
    assert(nbBindings == 2);
    std::vector<void*> buffers(nbBindings);
    auto buffersSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);

    int bindingIdxInput = 0;
    for (int i = 0; i < nbBindings; ++i)
    {
        if (engine.bindingIsInput(i))
            bindingIdxInput = i;
        else
        {
            auto bufferSizesOutput = buffersSizes[i];
            buffers[i] = safeCudaMalloc(bufferSizesOutput.first * elementSize(bufferSizesOutput.second));
        }
    }
    auto bufferSizesInput = buffersSizes[bindingIdxInput];

    int numberRun = 10;
    for (int num = 0; num < numberRun; num++)
    {
        buffers[bindingIdxInput] = createMnistCudaBuffer(bufferSizesInput.first,
                                                         bufferSizesInput.second, num);

        int outputIndex = engine.getBindingIndex("output_y");
        std::cout << "outputIndex : " << outputIndex << std::endl;
        context->execute(batchSize, &buffers[0]);
        float* outputs = new float[10];
        CHECK(cudaMemcpy(outputs, buffers[outputIndex], 10*sizeof(float), cudaMemcpyDeviceToHost));
        int maxIndex = 0;
        for (int i = 0; i < 10; ++ i) {
            printf("%.3f ", outputs[i]);
            if (outputs[maxIndex] < outputs[i]) {
                maxIndex = i;
            }
        }
        std::cout << std::endl;
        std::cout << "maxIndex : " << maxIndex << std::endl;
        delete[] outputs;
        CHECK(cudaFree(buffers[bindingIdxInput]));
    }

    sleepForever();
    context->destroy();
    std::cout << "execute done." << std::endl;
    return 0;
}

int main() {
    ICudaEngine *engine = initEngine();
    execute(*engine);
    engine->destroy();
    return 0;
}
