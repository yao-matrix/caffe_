#include "gtest/gtest.h"

#include <iostream>
#include <numeric>
#include "mkldnn.hpp"
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/conv_layer.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_conv_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"


using namespace mkldnn;

// initialize 3D conv parameters, these are used as global variables
int dims[] = {2, 32, 64, 128, 128};        // input dims
int out_dims[] = {2, 32, 64, 128, 128};    // output dims
int kernel_dims[] = {32, 32, 5, 5, 5};     // kernel_dims
int strides[] = {1, 1, 1};                 // strides
int paddings[] = {2, 2, 2};                // paddings
std::vector<int> blob_dims(dims, dims + 5);

int compute_input_index(int out_index ,int kernel_index, int pad_D, int stride_D){
  return out_index * stride_D - pad_D + kernel_index;
}


namespace caffe {

template<typename Dtype>
void init_data(float *output, Blob<Dtype>* data_blob, int bias_divide = -1, bool useAVX512 = true, bool reverse = false, bool isdiff = false) {
  std::vector<int> dims = data_blob->shape();
  Dtype *input = isdiff ? data_blob->mutable_cpu_diff() : data_blob->mutable_cpu_data();
  int cblk = useAVX512? 16 : 8;

  if(bias_divide == -1){  // reorder src data
    int nblk_size_i = dims[1]*dims[2]*dims[3]*dims[4];
    int cblk_size_i = dims[2]*dims[3]*dims[4];
    int Cblk_size_i = cblk*cblk_size_i;
    int dblk_size_i = dims[3]*dims[4];
    int hblk_size_i = dims[4];

    int nblk_size_o = dims[1]*dims[3]*dims[4];
    int Cblk_size_o = cblk*dims[3]*dims[4];
    int dblk_size_o = dims[0]*dims[1]*dims[3]*dims[4];
    int hblk_size_o = cblk*dims[4];

#pragma omp parallel for collapse(6) schedule(static)
    for(int d = 0; d < dims[2]; d++){
      for (int n = 0; n < dims[0]; ++n){
        for (int C = 0; C < dims[1]/cblk; ++C){
          for (int h = 0; h < dims[3]; ++h){
            //int blk_off_i = n*nblk_size_i + C*Cblk_size_i + d*dblk_size_i + h*hblk_size_i;
            //int blk_off_o = d*dblk_size_o + n*nblk_size_o + C*Cblk_size_o + h*hblk_size_o;
            for (int w = 0; w < dims[4]; ++w) {
              for (int c = 0; c < cblk; ++c) {
                //int off_i = blk_off_i + c*cblk_size_i + w;
                //int off_o = blk_off_o + w*cblk + c;
                int off_i = n*nblk_size_i + C*Cblk_size_i + d*dblk_size_i + h*hblk_size_i + c*cblk_size_i + w; 
                int off_o = d*dblk_size_o + n*nblk_size_o + C*Cblk_size_o + h*hblk_size_o + w*cblk + c;
                if (!reverse) output[off_o] = (float)input[off_i];
                else input[off_i] = (float)output[off_o];
              }
            }
          }
        }
      }
    }
  } else if(bias_divide == 0){  // reorder weight
    int oblk_size_i = dims[1]*dims[2]*dims[3]*dims[4];
    int Oblk_size_i = cblk*oblk_size_i;
    int iblk_size_i = dims[2]*dims[3]*dims[4];
    int Iblk_size_i = cblk*iblk_size_i;
    int hblk_size_i = dims[4];
    int dblk_size_i = dims[3]*dims[4];

    int wblk_size_o = cblk*cblk;
    int hblk_size_o = dims[4]*wblk_size_o;
    int Iblk_size_o = dims[3] * hblk_size_o;
    int Oblk_size_o = dims[1]/cblk *  Iblk_size_o; //dims[1]*dims[3]*dims[4];
    int dblk_size_o = dims[0]*dims[1]*dims[3]*dims[4];

#pragma omp parallel for collapse(7) schedule(static)
    for (int d = 0; d < dims[2]; ++d){
      for (int O = 0; O < dims[0]/cblk; ++O){
        for (int I = 0; I < dims[1]/cblk; ++I){
          for (int h = 0; h < dims[3]; ++h){
            for (int w  = 0; w < dims[4]; ++w){
              //int blk_off_i = O*Oblk_size_i + I*Iblk_size_i + d*dblk_size_i + h*hblk_size_i;
              //int blk_off_o = d*dblk_size_o + O*Oblk_size_o + I*Iblk_size_o + h*hblk_size_o;
              for (int ic = 0; ic < cblk; ++ic) {
                for (int oc = 0; oc < cblk; ++oc) {
                  //int off_i = blk_off_i + ic*iblk_size_i + oc*oblk_size_i + w;
                  //int off_o = blk_off_o + w*wblk_size_o + ic*cblk + oc;
                  int off_i = O*Oblk_size_i + I*Iblk_size_i + d*dblk_size_i + h*hblk_size_i + ic*iblk_size_i + oc*oblk_size_i + w;
                  int off_o = d*dblk_size_o + O*Oblk_size_o + I*Iblk_size_o + h*hblk_size_o + w*wblk_size_o + ic*cblk + oc;
                  if (!reverse) output[off_o] = (float)input[off_i];
                  else input[off_i] = (float)output[off_o];
                }
              }
            }
          }
        }
      }
    }
  } else if(bias_divide == -2) {
    int oblk_size_i = dims[1]*dims[2]*dims[3]*dims[4];
    int Oblk_size_i = cblk*oblk_size_i;
    int iblk_size_i = dims[2]*dims[3]*dims[4];
    int Iblk_size_i = cblk*iblk_size_i;
    int hblk_size_i = dims[4];
    int dblk_size_i = dims[3]*dims[4];

    int wblk_size_o = cblk*cblk;
    int hblk_size_o = dims[4]*wblk_size_o;
    int Iblk_size_o = dims[3] * hblk_size_o;
    int Oblk_size_o = dims[1]/cblk *  Iblk_size_o; //dims[1]*dims[3]*dims[4];
    int dblk_size_o = dims[0]*dims[1]*dims[3]*dims[4];
#pragma omp parallel for collapse(7) schedule(static)
    for (int d = 0; d < dims[2]; ++d){
      for (int O = 0; O < dims[0]/cblk; ++O){
        for (int I = 0; I < dims[1]/cblk; ++I){
          for (int h = 0; h < dims[3]; ++h){
            for (int w  = 0; w < dims[4]; ++w){
              for (int oc = 0; oc < cblk; ++oc) {
                for (int ic = 0; ic < cblk; ++ic) {
                int off_i = O*Oblk_size_i + I*Iblk_size_i + d*dblk_size_i + h*hblk_size_i + ic*iblk_size_i + oc*oblk_size_i + w;
                int off_o = d*dblk_size_o + O*Oblk_size_o + I*Iblk_size_o + h*hblk_size_o + w*wblk_size_o + oc*cblk + ic;
                if (!reverse) output[off_o] = (float)input[off_i];
                else input[off_i] = (float)output[off_o];
                }
              }
            }
          }
        }
      }
    }
  } else if (bias_divide >= 1){  // set bias data
    for(int oc = 0; oc < dims[0]; oc++){
      if(!reverse) output[oc] = (float)input[oc] / (float)bias_divide;
      else input[oc] = output[oc];
    }
  }
  else {
    std::cout << "Not implemented" << std::endl;
  }

}

template void init_data(float *output, Blob<float>* data_blob, int bias_divide, bool useAVX512, bool reverse, bool isdiff);
template void init_data(float *output, Blob<double>* data_blob, int bias_divide, bool useAVX512, bool reverse, bool isdiff);

template<typename Dtype>
void output_data(std::vector<float*> input, Blob<Dtype>* top_blob, bool useAVX512 = true){
  std::vector<int> dims = top_blob->shape();
  Dtype *output = top_blob->mutable_cpu_data();
  int cblk = useAVX512? 16 : 8;

  int nblk_size_o = dims[1]*dims[2]*dims[3]*dims[4];
  int cblk_size_o = dims[2]*dims[3]*dims[4];
  int Cblk_size_o = cblk * cblk_size_o;
  int dblk_size_o = dims[3]*dims[4];
  int hblk_size_o = dims[4];

  int nblk_size_i = dims[1]*dims[3]*dims[4];
  int Cblk_size_i = cblk*dims[3]*dims[4];
  //int dblk_size_i = dims[0]*dims[1]*dims[3]*dims[4];
  int hblk_size_i = cblk*dims[4];

  #pragma omp parallel for collapse(4) schedule(static)
  for(int d = 0; d < dims[2]; d++){
    for (int n = 0; n < dims[0]; ++n){
      for (int C = 0; C < dims[1]/cblk; ++C){
        for (int h = 0; h < dims[3]; ++h){
          int blk_off_i = n*nblk_size_i + C*Cblk_size_i + h*hblk_size_i;
          int blk_off_o = d*dblk_size_o + n*nblk_size_o + C*Cblk_size_o + h*hblk_size_o;
          for (int w = 0; w < dims[4]; ++w) {
            for (int c = 0; c < cblk; ++c) {
              int off_i = blk_off_i + w*cblk + c;
              int off_o = blk_off_o + c*cblk_size_o + w;
                output[off_o] = (Dtype)input[d][off_i];
            }
          }
        }
      }
    }
  }
}
template void output_data(std::vector<float*> input, Blob<float>* top_blob, bool useAVX512);
template void output_data(std::vector<float*> input, Blob<double>* top_blob, bool useAVX512);





template<typename Dtype>
void test_net(Blob<Dtype>* bottom_blob, Blob<Dtype>* weights_blob, Blob<Dtype>* bias_blob, Blob<Dtype>* top_blob){
  // get engine
  auto cpu_engine = engine(engine::cpu, 0);

  // init src, weights, bias data
  //float *data_src;
  std::vector<float> data_src(bottom_blob->count());
  //float *data_weights;
  std::vector<float> data_weights(weights_blob->count());
  std::vector<float> data_weights_bwddata(weights_blob->count());
  //float *data_bias;
  std::vector<float> data_bias(bias_blob->count());
  //float *data_dst;

  // init src_diff, weights_diff,
  std::vector<float> diff_src(bottom_blob->count()); // will be calculated ?
  std::vector<float> diff_weights(weights_blob->count()); // will be calculated ?
  std::vector<float> diff_top(top_blob->count());  // will be initialized

  std::vector<float> data_src_zero(dims[0] * dims[1] * dims[3] * dims[4], 0.);
  std::vector<float> data_weights_zero(kernel_dims[0] * kernel_dims[1] * kernel_dims[3] * kernel_dims[4], 0.);

  init_data(data_src.data(), bottom_blob, -1);
  init_data(data_weights.data(), weights_blob, 0);
  init_data(data_bias.data(), bias_blob, kernel_dims[2]);

  // initialize dat
  timeval gem_start, gem_end; // for timing test
  gettimeofday(&gem_start, NULL);   // for timing test 
  
  init_data(diff_top.data(), top_blob, -1, true, false, true);
  init_data(data_weights_bwddata.data(), weights_blob, -2);

  gettimeofday(&gem_end,NULL);
  int gem_timeuse1 = 1000000 *  ( gem_end.tv_sec - gem_start.tv_sec ) + gem_end.tv_usec - gem_start.tv_usec;
  printf("init elapsed  time: %d us\n", gem_timeuse1);


  // set 2d memory dims
  memory::dims conv_src_nchw = {dims[0], dims[1], dims[3], dims[4]}; //src shape
  memory::dims conv_weights_oihw = {out_dims[1], dims[1], kernel_dims[3], kernel_dims[4]}; // weights shape
  memory::dims conv_bias_x = {out_dims[1]};  // bias shape
  memory::dims conv_dst_nchw = {out_dims[0], out_dims[1], out_dims[3], out_dims[4]};   // dst shape

  // set 2d convolution parameters
  memory::dims conv_strides = {strides[1], strides[2]};  // strides shape
  memory::dims conv_padding = {paddings[1], paddings[2]};   // padding shape

  // create convolution usr memory desc
  auto conv_usr_src_md = memory::desc({conv_src_nchw}, memory::data_type::f32, memory::format::nChw16c);
  auto conv_usr_weights_md = memory::desc({conv_weights_oihw}, memory::data_type::f32, memory::format::OIhw16i16o);
  auto conv_usr_weights_bwddata_md = memory::desc({conv_weights_oihw}, memory::data_type::f32, memory::format::OIhw16o16i);
  //auto conv_usr_bias_md = memory::desc({conv_bias_x}, memory::data_type::f32, memory::format::x);

  //create convolution usr diff memory desc
  //auto conv_usr_diff_src_md = memory::desc({conv_src_nchw}, memory::data_type::f32, memory::format::nChw16c);
  //auto conv_usr_diff_weights_md = memory::desc({conv_weights_oihw}, memory::data_type::f32, memory::format::OIhw16i16o);
  auto conv_usr_diff_dst_md = memory::desc({conv_dst_nchw}, memory::data_type::f32, memory::format::nChw16c);


  // create convolution private memory desc
  auto conv_src_md = memory::desc({conv_src_nchw}, memory::data_type::f32, memory::format::any);
  auto conv_dst_md = memory::desc({conv_dst_nchw}, memory::data_type::f32,memory::format::any);
  auto conv_bias_md = memory::desc({conv_bias_x}, memory::data_type::f32, memory::format::x);
  auto conv_weights_md = memory::desc({conv_weights_oihw}, memory::data_type::f32, memory::format::any);
  auto conv_weights_bwddata_md = memory::desc({conv_weights_oihw}, memory::data_type::f32, memory::format::any);

  // create convolution private diff memory desc
  auto conv_diff_src_md = memory::desc({conv_src_nchw}, memory::data_type::f32, memory::format::any); // to be deleted
  auto conv_diff_dst_md = memory::desc({conv_dst_nchw}, memory::data_type::f32,memory::format::any);  // to be deleted
  auto conv_diff_weights_md = memory::desc({conv_weights_oihw}, memory::data_type::f32, memory::format::any); // to be deleted
  auto conv_diff_bias_md = memory::desc({conv_bias_x}, memory::data_type::f32, memory::format::any);  // to be deleted

  // create conv desc
  auto conv_desc = convolution_forward::desc(prop_kind::forward,
         convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
         conv_dst_md, conv_strides, conv_padding, conv_padding, padding_kind::zero);

  // create conv primitive desc
  auto conv_pd = convolution_forward::primitive_desc(conv_desc, cpu_engine);
  // create conv backward data desc
  auto conv_bwd_data_desc = convolution_backward_data::desc(convolution_direct, conv_src_md, conv_weights_bwddata_md,
                                  conv_dst_md, conv_strides, conv_padding, conv_padding, padding_kind::zero);
  auto conv_bwd_data_pd = convolution_backward_data::primitive_desc(conv_bwd_data_desc, cpu_engine, conv_pd);

  // create conv backward weights desc
  auto conv_bwd_weights_desc = convolution_backward_weights::desc(convolution_direct, conv_src_md, conv_weights_md, 
                                  conv_bias_md, conv_dst_md, conv_strides, conv_padding, conv_padding, padding_kind::zero);
  auto conv_bwd_weights_pd = convolution_backward_weights::primitive_desc(conv_bwd_weights_desc, cpu_engine, conv_pd);


  // create conv diff memory vector
  std::vector<memory> conv_diff_src_memory(kernel_dims[2]*dims[2],memory({conv_usr_src_md, cpu_engine}, data_src_zero.data()));
  std::vector<memory> conv_diff_weights_memory(out_dims[2]*kernel_dims[2], memory({conv_usr_weights_md, cpu_engine}, data_weights_zero.data()));
  std::vector<memory> conv_diff_bias_memory;  // can be deleted  

  std::vector<memory> conv_usr_src_memory;
  std::vector<memory> conv_usr_weights_memory;
  std::vector<memory> conv_usr_weights_memory_bwddata;
  std::vector<memory> conv_usr_diff_dst_memory;
  // create conv backward nets for both data and weights
  std::vector<std::vector<primitive> > backward_data_nets;
  std::vector<std::vector<primitive> > backward_weights_nets;
  // create a set of 2d conv backward layers
  int src_slice_num = dims[0] * dims[1] * dims[3] * dims[4];
  int weights_slice_num = kernel_dims[0] * kernel_dims[1] * kernel_dims[3] * kernel_dims[4];
  int dst_slice_num = out_dims[0] * out_dims[1] * out_dims[3] * out_dims[4];

  for(int od = 0; od < out_dims[2]; od++){
    for(int kd = 0; kd < kernel_dims[2]; kd++){
      // create two nets to include reorder and conv_bwd_data, and, reorder and conv_bwd_weights
      std::vector<primitive> net_bwd_data;
      std::vector<primitive> net_bwd_weights;

      int input_index = compute_input_index(od, kd, paddings[0], strides[0]);

      //set conv_usr_diff_dst_memory
      auto conv_usr_diff_dst_memory_tmp = memory({conv_usr_diff_dst_md, cpu_engine}, diff_top.data() + od * dst_slice_num);
      conv_usr_diff_dst_memory.push_back(conv_usr_diff_dst_memory_tmp);
      //set conv_usr_src_memory
      auto conv_usr_src_memory_tmp = memory({conv_usr_src_md, cpu_engine});
      if(input_index < 0 || input_index >= dims[2]){
        conv_usr_src_memory_tmp = memory({conv_usr_src_md, cpu_engine}, data_src_zero.data());
      } else {
        conv_usr_src_memory_tmp = memory({conv_usr_src_md, cpu_engine}, data_src.data() + input_index * src_slice_num);
      }
      conv_usr_src_memory.push_back(conv_usr_src_memory_tmp);

      //set conv_usr_weights_memory
      auto conv_usr_weights_memory_tmp = memory({conv_usr_weights_md, cpu_engine},data_weights.data() + kd * weights_slice_num);
      conv_usr_weights_memory.push_back(conv_usr_weights_memory_tmp);

      //set conv_usr_weights_bwddata_memory
      auto conv_usr_weights_memory_bwddata_tmp = memory({conv_usr_weights_bwddata_md, cpu_engine},data_weights_bwddata.data() + kd * weights_slice_num);
      conv_usr_weights_memory_bwddata.push_back(conv_usr_weights_memory_bwddata_tmp);

      //set conv_usr_bias_memory
      //auto conv_usr_bias_memory_tmp = memory({conv_usr_bias_md, cpu_engine}, data_bias.data());
      //conv_usr_bias_memory.push_back(conv_usr_bias_memory_tmp);

      // set conv_diff_src_memory
      auto conv_diff_src_memory_tmp = memory(conv_bwd_data_pd.diff_src_primitive_desc());
      // create conv_bwd_data
      auto conv_bwd_data = convolution_backward_data(conv_bwd_data_pd, conv_usr_diff_dst_memory_tmp, 
                               conv_usr_weights_memory_bwddata_tmp, conv_diff_src_memory_tmp);      
      if(input_index >= 0 && input_index < dims[2]){
        conv_diff_src_memory[input_index * kernel_dims[2] + kd] = conv_diff_src_memory_tmp;
        conv_bwd_data = convolution_backward_data(conv_bwd_data_pd, conv_usr_diff_dst_memory_tmp,
                               conv_usr_weights_memory_bwddata_tmp, conv_diff_src_memory[input_index * kernel_dims[2] + kd]);
      }
      net_bwd_data.push_back(conv_bwd_data);

      // set conv_diff_weigts_memory
      auto conv_diff_weights_memory_tmp = memory(conv_bwd_weights_pd.diff_weights_primitive_desc());
      // set conv_diff_bias_memory
      auto conv_diff_bias_memory_tmp = memory(conv_bwd_weights_pd.diff_bias_primitive_desc());
      conv_diff_bias_memory.push_back(conv_diff_bias_memory_tmp);

      conv_diff_weights_memory[od * kernel_dims[2] + kd] = conv_diff_weights_memory_tmp;
      // create_conv_bwd_weights
      auto conv_bwd_weights = convolution_backward_weights(conv_bwd_weights_pd, conv_usr_src_memory_tmp,
                                conv_usr_diff_dst_memory_tmp, conv_diff_weights_memory[od * kernel_dims[2] + kd], 
                                  conv_diff_bias_memory[od * kernel_dims[2] + kd]);
      net_bwd_weights.push_back(conv_bwd_weights);

      // put the two net into net lists.
      backward_data_nets.push_back(net_bwd_data);
      backward_weights_nets.push_back(net_bwd_weights);
    }
  }
  
  //create elementwise sum layer for both diff_src and diff_weights
  std::vector<double> diffsrc_scale(kernel_dims[2], 1.0);
  std::vector<double> diffweights_scale(out_dims[2], 1.0);
  auto diffsrc_sum_memory_pd = conv_bwd_data_pd.diff_src_primitive_desc();
  auto diffweights_sum_memory_pd = conv_bwd_weights_pd.diff_weights_primitive_desc();
  std::vector<memory::primitive_desc> diffsrc_srcs_pd(kernel_dims[2], diffsrc_sum_memory_pd);
  std::vector<memory::primitive_desc> diffweights_srcs_pd(out_dims[2], diffweights_sum_memory_pd);

  const auto diffsrc_sum_dst_memroy_desc = diffsrc_sum_memory_pd.desc();     // create the dst meomry descriptor
  const auto diffweights_sum_dst_memory_desc = diffweights_sum_memory_pd.desc();
  auto diffsrc_sum_pd = sum::primitive_desc(diffsrc_sum_dst_memroy_desc, diffsrc_scale, diffsrc_srcs_pd);  // create sum primitive descriptor
  auto diffweights_sum_pd = sum::primitive_desc(diffweights_sum_dst_memory_desc, diffweights_scale, diffweights_srcs_pd);

  std::vector<double> diffbias_scale(kernel_dims[2]*out_dims[2], 1.0/float(kernel_dims[2]));
  auto diffbias_sum_memory_pd = conv_bwd_weights_pd.diff_bias_primitive_desc();
  std::vector<memory::primitive_desc> diffbias_srcs_pd(kernel_dims[2]*out_dims[2], diffbias_sum_memory_pd);
  const auto diffbias_sum_dst_memroy_desc = diffbias_sum_memory_pd.desc();     // create the dst meomry descriptor
  auto diffbias_sum_pd = sum::primitive_desc(diffbias_sum_dst_memroy_desc, diffbias_scale, diffbias_srcs_pd);  // create sum primitive descriptor



  std::vector<float> diffsrc_sum_out(bottom_blob->count(), 0.);
  std::vector<float> diffweights_sum_out(weights_blob->count(), 0.);
  std::vector<float> diffbias_sum_out(bias_blob->count(), 0.);

  timeval start, end;    // for timing test
  gettimeofday(&start,NULL);  // for timing test

  // backward data diff
  timeval bwd_data_start, bwd_data_end;    // for timing test
  gettimeofday(&bwd_data_start,NULL);  // for timing test

  for(int od = 0; od < out_dims[2]; od++) {
    for(int kd = 0; kd < kernel_dims[2]; kd++) {
      int cal_input_index = compute_input_index(od, kd, paddings[0], strides[0]);
      if(cal_input_index >= 0 && cal_input_index < dims[2]) {
        stream(stream::kind::eager).submit(backward_data_nets[od * kernel_dims[2] + kd]).wait();
        stream(stream::kind::eager).submit(backward_weights_nets[od * kernel_dims[2] + kd]).wait();
      } else {
        stream(stream::kind::eager).submit(backward_weights_nets[od * kernel_dims[2] + kd]).wait();
        //conv_diff_weights_memory[od * kernel_dims[2] + kd] = memory(conv_bwd_weights_pd.diff_weights_primitive_desc(), data_weights_zero.data());
      }
    }
  }

  gettimeofday(&bwd_data_end,NULL);
  int bwd_data_timeuse1 = 1000000 *  ( bwd_data_end.tv_sec - bwd_data_start.tv_sec ) + bwd_data_end.tv_usec - bwd_data_start.tv_usec;
  printf("bwd data elapsed  time: %d us\n", bwd_data_timeuse1);


  timeval sumdiff_data_start, sumdiff_data_end;    // for timing test
  gettimeofday(&sumdiff_data_start,NULL);  // for timing test

  // calculate src diff
  for(int id = 0; id < dims[2]; id++) {
    auto diffsrc_dst = memory(diffsrc_sum_pd.dst_primitive_desc(), diffsrc_sum_out.data() + id * src_slice_num);
    std::vector<primitive::at> diffsrc_inputs;
    for(int kd = 0; kd < kernel_dims[2]; kd++) {
      diffsrc_inputs.push_back(conv_diff_src_memory[id * kernel_dims[2] + kd]);
    }
    auto diffsrc_sum_inputs = sum(diffsrc_sum_pd, diffsrc_inputs, diffsrc_dst);
    std::vector<primitive> diffsrc_pipeline;
    diffsrc_pipeline.push_back(diffsrc_sum_inputs);
    stream(stream::kind::eager).submit(diffsrc_pipeline).wait();
  }

  gettimeofday(&sumdiff_data_end,NULL);
  int sumdiff_data_timeuse1 = 1000000 *  ( sumdiff_data_end.tv_sec - sumdiff_data_start.tv_sec ) + sumdiff_data_end.tv_usec - sumdiff_data_start.tv_usec;
  printf("sum diff data elapsed  time: %d us\n", sumdiff_data_timeuse1);

/*
  timeval bwd_weight_start, bwd_weight_end;    // for timing test
  gettimeofday(&bwd_weight_start,NULL);  // for timing test

  // backward weights diff

  for(int od = 0; od < out_dims[2]; od++) {
    for(int kd = 0; kd < kernel_dims[2]; kd++) {
      int cal_input_index = compute_input_index(od, kd, paddings[0], strides[0]);
      if(cal_input_index >= 0 && cal_input_index < dims[2]) {
        stream(stream::kind::eager).submit(backward_weights_nets[od * kernel_dims[2] + kd]).wait();
      } else {
        conv_diff_weights_memory[od * kernel_dims[2] + kd] = memory(conv_bwd_weights_pd.diff_weights_primitive_desc(), data_weights_zero.data());
      }
    }
  }



  gettimeofday(&bwd_weight_end,NULL);
  int bwd_weight_timeuse1 = 1000000 *  ( bwd_weight_end.tv_sec - bwd_weight_start.tv_sec ) + bwd_weight_end.tv_usec - bwd_weight_start.tv_usec;
  printf("bwd diff weight elapsed  time: %d us\n", bwd_weight_timeuse1);
**/
  
  timeval sumdiff_weight_start, sumdiff_weight_end;    // for timing test
  gettimeofday(&sumdiff_weight_start,NULL);  // for timing test
  // calculate weights diff
  for(int kd = 0; kd < kernel_dims[2]; kd++) {
    auto diffweights_dst = memory(diffweights_sum_pd.dst_primitive_desc(), diffweights_sum_out.data() + kd * weights_slice_num);
    std::vector<primitive::at> diffweights_inputs;
    for(int od = 0; od < out_dims[2]; od++) {
      diffweights_inputs.push_back(conv_diff_weights_memory[od * kernel_dims[2] + kd]);
    }
    auto diffweights_sum_inputs = sum(diffweights_sum_pd, diffweights_inputs, diffweights_dst);
    std::vector<primitive> diffweights_pipeline;
    diffweights_pipeline.push_back(diffweights_sum_inputs);
    stream(stream::kind::eager).submit(diffweights_pipeline).wait();
  }
  gettimeofday(&sumdiff_weight_end,NULL);
  int sumdiff_weight_timeuse1 = 1000000 *  ( sumdiff_weight_end.tv_sec - sumdiff_weight_start.tv_sec ) +sumdiff_weight_end.tv_usec - sumdiff_weight_start.tv_usec;
  printf("sum diff weight elapsed  time: %d us\n", sumdiff_weight_timeuse1); 


  timeval sumdiff_bias_start, sumdiff_bias_end;    // for timing test
  gettimeofday(&sumdiff_bias_start,NULL);  // for timing test
  auto diffbias_dst = memory(diffbias_sum_pd.dst_primitive_desc(), diffbias_sum_out.data());
  std::vector<primitive::at> diffbias_inputs;
  for(int n = 0; n < out_dims[2] * kernel_dims[2]; n++){
    diffbias_inputs.push_back(conv_diff_bias_memory[n]);
  }
  auto diffbias_sum_inputs = sum(diffbias_sum_pd, diffbias_inputs, diffbias_dst);
  std::vector<primitive> diffbias_pipeline;
  diffbias_pipeline.push_back(diffbias_sum_inputs);
  stream(stream::kind::eager).submit(diffbias_pipeline).wait();

  gettimeofday(&sumdiff_bias_end,NULL);
  int sumdiff_bias_timeuse1 = 1000000 *  ( sumdiff_bias_end.tv_sec - sumdiff_bias_start.tv_sec ) +sumdiff_bias_end.tv_usec - sumdiff_bias_start.tv_usec;
  printf("sum diff bias elapsed  time: %d us\n", sumdiff_bias_timeuse1);



  gettimeofday(&end,NULL);
  int timeuse1 = 1000000 *  ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
  printf("elapsed  time: %d us\n", timeuse1);
  
  timeval out_start, out_end;    // for timing test
  gettimeofday(&out_start,NULL);  // for timing test

  init_data(diffweights_sum_out.data(), weights_blob, 0, true, true, true);
  init_data(diffsrc_sum_out.data(), bottom_blob, -1, true, true, true);
  init_data(diffbias_sum_out.data(), bias_blob, kernel_dims[2], true, true, true);

  gettimeofday(&out_end,NULL);
  int out_timeuse1 = 1000000 *  ( out_end.tv_sec - out_start.tv_sec ) + out_end.tv_usec - out_start.tv_usec;
  printf("out put elapsed  time: %d us\n", out_timeuse1);

}

template void test_net(Blob<float>* bottom_blob, Blob<float>* weights_blob, Blob<float>* bias_blob, Blob<float>* top_blob);
template void test_net(Blob<double>* bottom_blob, Blob<double>* weights_blob, Blob<double>* bias_blob, Blob<double>* top_blob);


template<typename TypeParam>
class MKLConvolution3dLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

  protected:
    MKLConvolution3dLayerTest()
      : blob_bottom_(new Blob<Dtype>(blob_dims)),
        blob_top_(new Blob<Dtype>()) {}
    virtual void SetUp() {
      // fill the values
      FillerParameter filler_param;
      filler_param.set_value(1.);
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(this->blob_bottom_);
      this->blob_bottom_vec_.push_back(this->blob_bottom_);
      this->blob_top_vec_.push_back(this->blob_top_);
    }

    virtual ~MKLConvolution3dLayerTest(){
      delete blob_bottom_;
      delete blob_top_;      
    }

    virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
      this->ref_blob_top_.reset(new Blob<Dtype>());
      this->ref_blob_top_->ReshapeLike(*top);
      return this->ref_blob_top_.get();
    }

    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_;
    shared_ptr<Blob<Dtype> > ref_blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MKLConvolution3dLayerTest, TestDtypesAndDevices);

TYPED_TEST(MKLConvolution3dLayerTest, TestSimpleConvolution) {
  typedef typename TypeParam::Dtype Dtype;


  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(kernel_dims[2]);
  convolution_param->add_stride(strides[0]);
  convolution_param->add_pad(paddings[0]);
  convolution_param->set_num_output(out_dims[1]);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  //nvolution_param->mutable_weight_filler()->set_value(0.1);
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);

  shared_ptr<Layer<Dtype> > layer(
      new ConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  Blob<Dtype>* random_blob(new Blob<Dtype>(this->blob_top_->shape()));
  FillerParameter filler_param;
  filler_param.set_value(1.);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(random_blob);
  caffe_copy(this->blob_top_->count(), random_blob->cpu_data(), this->blob_top_->mutable_cpu_diff());
  

  Blob<Dtype>* blob_weight = layer->blobs()[0].get();
  Blob<Dtype>* blob_bias = layer->blobs()[1].get();

  Blob<Dtype>* blob_bottom2(new Blob<Dtype>(this->blob_bottom_->shape()));
  Blob<Dtype>* blob_weight2(new Blob<Dtype>(blob_weight->shape()));
  Blob<Dtype>* blob_bias2(new Blob<Dtype>(blob_bias->shape()));

  caffe_copy(this->blob_bottom_->count(), this->blob_bottom_->cpu_data(), blob_bottom2->mutable_cpu_data());
  caffe_copy(blob_weight->count(), blob_weight->cpu_data(), blob_weight2->mutable_cpu_data());
  caffe_copy(blob_bias->count(), blob_bias->cpu_data(), blob_bias2->mutable_cpu_data());

  caffe_set(blob_bottom2->count(), static_cast<Dtype>(0.), blob_bottom2->mutable_cpu_diff());
  caffe_set(blob_weight2->count(), static_cast<Dtype>(0.), blob_weight2->mutable_cpu_diff());
  caffe_set(blob_bias2->count(), static_cast<Dtype>(0.), blob_bias2->mutable_cpu_diff());


  test_net(blob_bottom2, blob_weight2, blob_bias2, this->blob_top_);


  std::vector<bool> propagate_down(1, true);
  timeval gem_start, gem_end; // for timing test
  gettimeofday(&gem_start, NULL);   // for timing test

  layer->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);

  gettimeofday(&gem_end,NULL);
  int gem_timeuse1 = 1000000 *  ( gem_end.tv_sec - gem_start.tv_sec ) + gem_end.tv_usec - gem_start.tv_usec;
  printf("gemm elapsed  time: %d us\n", gem_timeuse1);


  // Check against reference convolution.  
  const Dtype* bottom_diff_gemm = this->blob_bottom_->cpu_diff();
  const Dtype* bottom_diff_mkldnn = blob_bottom2->cpu_diff();
  for (int n = 0; n < this->blob_bottom_->count(); n++){
    EXPECT_NEAR(bottom_diff_gemm[n], bottom_diff_mkldnn[n], 3e-4);
  }


  const Dtype* weight_diff_gemm = blob_weight->cpu_diff();
  const Dtype* weight_diff_mkldnn = blob_weight2->cpu_diff();
  for (int n = 0; n < blob_weight->count(); n++){
    //EXPECT_NEAR(weight_diff_gemm[n] , weight_diff_mkldnn[n], 2e-4);
    //EXPECT_NEAR(fabs(weight_diff_mkldnn[n] / weight_diff_gemm[n]), 1.0, 1e-2);
    if(fabs(fabs(weight_diff_mkldnn[n] / weight_diff_gemm[n])-1.0) > 1e-2) 
       std::cout << "weight_diff_gemm[n] = " << weight_diff_gemm[n] << ", weight_diff_mkldnn[n] = " << weight_diff_mkldnn[n] 
                 << ", difference = " << weight_diff_gemm[n] - weight_diff_mkldnn[n] << ", n = " << n << std::endl;
  }

  const Dtype* bias_diff_gemm = blob_bias->cpu_diff();
  const Dtype* bias_diff_mkldnn = blob_bias2->cpu_diff();
  for(int n = 0; n < blob_bias->count(); n++) {
     EXPECT_NEAR(fabs(bias_diff_gemm[n] / bias_diff_mkldnn[n]), 1, 1e-4);
  }
}

}


