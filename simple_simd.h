#include<queue>
#include "simd.h"

float innerProductSimd(float *target,float *test,size_t vecdim){
    assert(vecdim%8==0);//假设维度能被8整除
    
    simd8float32 sum(0.0);//8xfloat32全部初始化为0
    for(int i =0; i<vecdim; i+=8){
    simd8float32 s1(target+i), s2(test+i);
    simd8float32 m=s1*s2;
    sum+=m;
    }
   
    float tmp[8];
    sum.storeu(tmp);
    float dis=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
    return 1-dis;
}
    





std::priority_queue<std::pair<float, uint32_t> > flat_search_improve(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
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
