#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
#include "simple_simd.h"
// 可以自行添加需要的头文件

using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}


int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path2 = "/anndata/"; 
    // std::string data_path = "/files/"; 
    auto test_query = LoadData<float>(data_path2 + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path2 + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path2 + "DEEP100K.base.100k.fbin", base_number, vecdim);
    // auto test_query = LoadData<uint8_t>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    // auto base = LoadData<uint8_t>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    std::vector<std::vector<int>> labels;
    std::vector<std::vector<std::vector<float>>> centroids;
    kmeans(base,base_number,vecdim,centroids,labels);
    // quantize_and_write_to_file(test_query,test_number,vecdim,"DEEP100K.query.fbin");
    // quantize_and_write_to_file(base,base_number,vecdim,"DEEP100K.base.100k.fbin");
    // 只测试前2000条查询
    test_number = 2000;

    const size_t k = 10;

    std::vector<SearchResult> results;
    results.resize(test_number);
    // auto LUT=precompute_lut(test_query,test_number,vecdim,centroids);
    // // 如果你需要保存索引，可以在这里添加你需要的函数，你可以将下面的注释删除来查看pbs是否将build.index返回到你的files目录中
    // // 要保存的目录必须是files/*
    // // 每个人的目录空间有限，不需要的索引请及时删除，避免占空间太大
    // // 不建议在正式测试查询时同时构建索引，否则性能波动会较大
    // // 下面是一个构建hnsw索引的示例
    // // build_index(base, base_number, vecdim);

    size_t rerank=400;
    // 查询测试代码
    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        
        int ret = gettimeofday(&val, NULL);
        auto LUT=precompute_lut_one(test_query+i*vecdim,vecdim,centroids);
        // 该文件已有代码中你只能修改该函数的调用方式
        // 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。
        // auto res = flat_search_improve(base, test_query + i*vecdim, base_number, vecdim, k);
        // auto res = pq_search_one(base,test_query + i*vecdim,base_number,vecdim,i,k,labels,LUT,rerank);
       auto res = pq_search_simd(base,test_query + i*vecdim,base_number,vecdim,i,k,labels,LUT,rerank);

        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while (res.size()) {   
            int x = res.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc/k;

        results[i] = {recall, diff};
    }

    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    // 浮点误差可能导致一些精确算法平均recall不是1
    std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";
    return 0;
}

