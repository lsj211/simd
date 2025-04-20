#include <arm_neon.h>
#program once

struct simd8float32{
    float32x4x2_t data;
    
    simd8float32()=default;
    
    explicit simd8float32(float value) {
        data.val[0] = vdupq_n_f32(value);
        data.val[1] = vdupq_n_f32(value);
    }

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
        t.data.val[0]=vaddq_f32(this->data.val[0],other.data.val[0]);
        t.data.val[1]=vaddq_f32(this->data.val[1],other.data.val[1]);
        return t;
    }
    
    void storeu(float *target) const
    {
        vst1q_f32(target, this->data.val[0]);     
        vst1q_f32(target + 4, this->data.val[1]); 
    }
};



struct simd32uint8 {
    uint8x16x2_t data;  // 使用两个 128 位的寄存器（uint8x16_t）表示 256 位的数据

    // 默认构造函数
    simd32uint8() = default;

   
    explicit simd32uint8(uint8_t value) {
        data.val[0] = vdupq_n_u8(value);  
        data.val[1] = vdupq_n_u8(value);
    }

    explicit simd32uint8(const uint8_t *x) {
        data.val[0] = vld1q_u8(x);      // 加载前 16 个 uint8_t 到 val[0]
        data.val[1] = vld1q_u8(x + 16); // 加载后 16 个 uint8_t 到 val[1]
    }

    // 按元素相乘
    simd32uint8 operator*(const simd32uint8 &other) const {
        simd32uint8 t;
        t.data.val[0] = vmulq_u8(this->data.val[0], other.data.val[0]);  // 按元素相乘：val[0] 与 val[0]
        t.data.val[1] = vmulq_u8(this->data.val[1], other.data.val[1]);  // 按元素相乘：val[1] 与 val[1]
        return t;
    }

    // 按元素相加
    simd32uint8 operator+(const simd32uint8 &other) const {
        simd32uint8 t;
        t.data.val[0] = vaddq_u8(this->data.val[0], other.data.val[0]);  // 按元素相加：val[0] 与 val[0]
        t.data.val[1] = vaddq_u8(this->data.val[1], other.data.val[1]);  // 按元素相加：val[1] 与 val[1]
        return t;
    }

    // 存储到内存
    void storeu(uint8_t *target) const {
        vst1q_u8(target, this->data.val[0]);      // 存储 val[0] 到内存
        vst1q_u8(target + 16, this->data.val[1]); // 存储 val[1] 到内存
    }


};
