#include "simd.h"
float compute_distance(const float *v1, const float * v2,size_t vecdim) {
    float dist = 0.0;
    for (int i = 0; i < vecdim; ++i) {
        dist += (v1[i] - v2[i]) * (v1[i] - v2[i]);
        // simd8float32 s1(v1+i),s2(v2+i);

    }
    return sqrt(dist);
}

void kmeans(const float *data,size_t data_num,size_t vecdim){
    const int division=4;
    const size_t div_dim=vecdim/division;
    const size_t K = 256;
    const size_t vec_num=data_num/vecdim;
    std::vector<std::vector<std::vector<float>>> centroids(division, std::vector<std::vector<float>>(K, std::vector<float>(div_dim)));//这里保存256个簇中心
    std::vector<std::vector<int>> labels(division, std::vector<int>(vec_num));//每个子段对应的编号
    for(int i=0;i<division;i++)
    {
        float *begin=data+i*div_dim;
        std::vector<float> assemble(vec_num*div_dim)//储存子段中的所有向量
        for(int j=0;j<vec_num;j++)
        {
            const float* src = data + j * vecdim + i * div_dim;
            float* dst = assemble.data() + j * div_dim;
            std::copy(src, src + div_dim, dst);
        }
        
        std::vector<std::vector<float>> new_centroids(K, std::vector<float>(div_dim, 0.0f));
        std::vector<int> counts(K, 0);

       
        std::srand(42);
        for (size_t j = 0; j < K; ++j) {
            size_t r = std::rand() % vec_num;
            const float* src = assemble.data() + r * div_dim;
            std::copy(src, src + div_dim, centroids[i][j].begin());
        }
        
        bool stop=false;
        int time=0;
        while(!stop||time<20)
        {
            for(int t=0;t<vec_num;t++)
            {
                float *src=assemble.data()+t*div_dim;
                float best_d = std::numeric_limits<float>::max();
                int best_k = 0;
                for(int k=0;k<K;k++)
                {
                    float dis=compute_distance(src,centroids[i][k].begin(),div_dim);
                    if(dis<best_d)
                    {
                        best_d=dis;
                        best_k=k;
                    }
                }
                if(labels[i][t]!=best_k)
                {
                    stop=false;
                    labels[i][t]=best_k//记录所属的簇
                }
              
            }
            for (int k = 0; k < K; ++k) {
                std::fill(new_centroids[k].begin(), new_centroids[k].end(), 0.0f);
                counts[k] = 0;
            }

            for (int j = 0; j < vec_num; ++j) {
                int label = labels[i][j];
                float* pt = assemble.data() + j * div_dim;
                for (size_t d = 0; d < div_dim; ++d) {
                    new_centroids[label][d] += pt[d];
                }
                counts[label]++;
            }

            
            for (int k = 0; k < K; ++k) {
                if (counts[k] > 0) {
                    for (int d = 0; d < div_dim; ++d) {
                        new_centroids[k][d] /= counts[k];
                    }
                    centroids[i][k].swap(new_centroids[k]);
                }
            }
            time++;
        }
    
    }
    

}



std::priority_queue<std::pair<float, uint32_t> > pq_search_improve(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
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