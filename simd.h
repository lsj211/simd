#include <arm_neon.h>


struct simd8float32{
    float32x4x2_t data;
    
    simd8float32()=default;
    
    explicit simd8float32(const float*x)
     :data{vld1q_f32(x),vld1q_f32(x+4)}{}
    simd8float32 operator*(const simd8float32 &other)const
    {
        simd8float32 t;
        t.data.val[0]=vmulq_f32(this->data.val[0],other.data.val[0]);
        t.data.val[1]=vmulq_f32(this->data.val[1],other.data.val[1]);
        return t;
    }
    simd8float32 operator+(const simd8float32 &other)const
    {
        simd8float32 t;
        t.data.val[0]=vmulq_f32(this->data.val[0],other.data.val[0]);
        t.data.val[1]=vmulq_f32(this->data.val[1],other.data.val[1]);
        return t;
    }
        
};