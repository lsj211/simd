#include<queue>
#include "simd.h"


void quantize_and_write_to_file(const float* data, size_t n, size_t d, const std::string& filename) {
    std::string filepath = "files/" + filename;

    std::ofstream output_file(filepath, std::ios::binary);
    if (!output_file) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    // 写入 header：n 和 d
    uint32_t n32 = static_cast<uint32_t>(n);
    uint32_t d32 = static_cast<uint32_t>(d);
    output_file.write(reinterpret_cast<const char*>(&n32), sizeof(uint32_t));
    output_file.write(reinterpret_cast<const char*>(&d32), sizeof(uint32_t));

    float scale = 255.0f / 2.0f;
    for (size_t i = 0; i < n * d; ++i) {
        int quantized_value = std::round((data[i] + 1.0f) * scale);
        quantized_value = std::max(0, std::min(255, quantized_value));
        uint8_t quantized_data = static_cast<uint8_t>(quantized_value);
        output_file.write(reinterpret_cast<const char*>(&quantized_data), sizeof(quantized_data));
    }

    output_file.close();

    if (output_file.good()) {
        std::cout << "Successfully wrote quantized data to: " << filepath << std::endl;
    } else {
        std::cerr << "Failed to properly close file: " << filepath << std::endl;
    }
}

// void quantize_and_write_to_file(const float* data, size_t size, const std::string& filename) {
//     std::string filepath = "files/" + filename;

//     std::ofstream output_file(filepath, std::ios::binary);
//     if (!output_file) {
//         std::cerr << "Failed to open file for writing: " << filename << std::endl;
//         return;
//     }

//     float scale = 255.0f / 2.0f;
//     for (size_t i = 0; i < size; ++i) {
//         int quantized_value = std::round((data[i] + 1.0f) * scale);
//         quantized_value = std::max(0, std::min(255, quantized_value));
//         uint8_t quantized_data = static_cast<uint8_t>(quantized_value);
//         output_file.write(reinterpret_cast<const char*>(&quantized_data), sizeof(quantized_data));
//     }

//     output_file.close();

//     if (output_file) {
//         std::cout << "success" << std::endl;
//     }
// }



float innerProductSimd(uint8_t *target,uint8_t *test,size_t vecdim){
    assert(vecdim%32==0);//假设维度能被8整除
    
    simd32uint8 sum1(uint8_t(0));//8xfloat32全部初始化为0
    simd32uint8 sum2(uint8_t(0));
    for(int i =0; i<vecdim; i+=32){
    simd32uint8 s1(target+i), s2(test+i);
    simd32uint8 m=s1*s2;
    sum1=sum1+m;
    sum2=sum2+s1+s2;
    }
    uint8_t tmp1[32];
    uint8_t tmp2[32];
    float result1=0.0f;
    float result2=0.0f;
    sum1.storeu(tmp1); 
    sum2.storeu(tmp2); 
    for (int i = 0; i < 32; i++) {
        result1 += float(tmp1[i]);
    }
    for (int i = 0; i < 32; i++) {
        result2 += float(tmp2[i]);
    }
    float scale=255/2;
    float dis=(result1-scale*result2)/(scale*scale)+vecdim*1;
    // float tmp[8];
    // sum.storeu(tmp);
    // float dis=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    return 1-dis;
}
    





std::priority_queue<std::pair<float, uint32_t> > flat_search_sq(uint8_t* base, uint8_t* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t> > q;

    for(int i = 0; i < base_number; ++i) {
        float dis = 0;

        // DEEP100K数据集使用ip距离
        // for(int d = 0; d < vecdim; ++d) {
        //     dis += base[d + i*vecdim]*query[d];
        // }
        // dis = 1 - dis;
        dis=innerProductSimd(&(base[i*vecdim]),query,vecdim);

        if(q.size() < k) {
            q.push({dis, i});
        } else {
            if(dis < q.top().first) {
                q.push({dis, i});
                q.pop();
            }
        }
    }
    return q;
}
