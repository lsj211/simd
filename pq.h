#include "simd.h"
float compute_distance(const float *v1, const float * v2,size_t vecdim) {
    float dist = 0.0;
    for (int i = 0; i < vecdim; ++i) {
        dist += (v1[i] - v2[i]) * (v1[i] - v2[i]);
        // simd8float32 s1(v1+i),s2(v2+i);

    }
    return sqrt(dist);
}

void kmeans(const float* data, size_t base_number, size_t vecdim, std::vector<std::vector<std::vector<float>>>  &centroids, std::vector<std::vector<int>> &labels) {
    const int division = 4;
    const size_t div_dim = vecdim / division;
    const size_t K = 256;
    const size_t vec_num = base_number;
    centroids.resize(division, std::vector<std::vector<float>>(K, std::vector<float>(div_dim)));//这里保存256个簇中心
    labels.resize(division, std::vector<int>(vec_num));//每个子段对应的编号
    for (int i = 0; i < division; i++)
    {
        const float* begin = data + i * div_dim;
        std::vector<float> assemble(vec_num * div_dim);//储存子段中的所有向量
            for (int j = 0; j < vec_num; j++)
            {
                const float* src = data + j * vecdim + i * div_dim;
                float* dst = assemble.data() + j * div_dim;
                std::copy(src, src + div_dim, dst);
            }

        std::vector<std::vector<float>> new_centroids(K, std::vector<float>(div_dim, 0.0f));
        std::vector<int> counts(K, 0);


        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        for (size_t j = 0; j < K; ++j) {
            size_t r = std::rand() % vec_num;
            const float* src = assemble.data() + r * div_dim;
            std::copy(src, src + div_dim, centroids[i][j].begin());
        }

        bool stop = false;
        int time = 0;
        while (!stop &&time < 20)
        {
            for (int t = 0; t < vec_num; t++)
            {
                float* src = assemble.data() + t * div_dim;
                float best_d = std::numeric_limits<float>::max();
                int best_k = 0;
                for (int k = 0; k < K; k++)
                {
                    float dis = compute_distance(src, centroids[i][k].data(), div_dim);
                    if (dis < best_d)
                    {
                        best_d = dis;
                        best_k = k;
                    }
                }
                if (labels[i][t] != best_k)
                {
                    stop = false;
                    labels[i][t] = best_k;//记录所属的簇
                }

            }
            for (int k = 0; k < K; ++k) {
                std::fill(new_centroids[k].begin(), new_centroids[k].end(), 0.0f);
                counts[k] = 0;
            }
            //计算该聚类平均值并重新设置中心点
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




std::vector<std::vector<float>> precompute_lut(const float *queries, size_t num_queries, size_t vecdim,const std::vector<std::vector<std::vector<float>>>& centroids) 
{
    const int division = 4;
    const size_t div_dim = vecdim / division;
    const size_t K = 256;
    std::vector<std::vector<float>> lut(num_queries, std::vector<float>(division * K));//对每一个查询向量都进行预处理

    for (size_t q = 0; q < num_queries; ++q) {
        const float* query = queries + q * vecdim;
        for (int k_sub = 0; k_sub < division; ++k_sub) {
            const float* query_sub = query + k_sub * div_dim;
            for (int center_id = 0; center_id < K; ++center_id) {
                const float* centroid = centroids[k_sub][center_id].data();
                float sum = 0.0;
                for(int t=0;t<div_dim;t++)
                {
                    sum+=centroid[t]*query_sub[t];
                }
                lut[q][k_sub * K + center_id] = sum;
            }
        }
    }
    return lut;
}



std::priority_queue<std::pair<float, uint32_t>> pq_search(
    float *base, float *test,size_t base_number,size_t vecdim,
    size_t test_number, size_t k,
    const std::vector<std::vector<int>>& labels,std::vector<std::vector<float>> LUT,size_t rerank) {
    std::priority_queue<std::pair<float, uint32_t>> q;
    const int division = 4;
 
    const size_t K = 256;
    for (size_t i = 0; i < base_number; ++i) {
        float total_inner = 0.0;
        for (int k_sub = 0; k_sub < division; ++k_sub) {
            int cluster_id = labels[k_sub][i];
            total_inner += LUT[test_number][k_sub * K + cluster_id];
        }
        float distance = 1.0 - total_inner;

        if (q.size() < rerank) {
            q.push({distance, i});
        } else if (distance < q.top().first) {
          
            q.push({distance, i}); 
            q.pop();
        }
    }
    // return q;
    std::priority_queue<std::pair<float, uint32_t>> result;
    for(size_t i=0;i<rerank;i++)
    {
       size_t id=q.top().second;
       float dis=innerProductSimd(base+id*vecdim,test,vecdim);
       q.pop();
       if (result.size() < k) 
       {
         result.push({dis, id});
       } 
       else if (dis < result.top().first)
        {
      
            result.push({dis, id}); 
            result.pop();
       }
    }
    return result;
}
