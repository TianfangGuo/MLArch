#ifndef NETWORK_H
#define NETWORK_H

#include <iostream>
#include <vector>
#include <map>

#include "file_utils.h"
#include "stream_utils.h"
#include "array4d.h"

template <class T>
class Network {
public:
    Network(std::string cfg_file_name);
    ~Network();

    int obtain_parameters();
    int conv_convert(int layer_id, int padding, int stride, Array3D<T>& initial_input, Array4D<T>& initial_kernel,
             Array2D<T>& input_matrix, Array2D<T>& kernel_matrix);
    int conv_convert_stream(int layer_id, int padding, int stride, Stream<T>& input, Stream<T>& output);

    void initialize();
    std::string get_parameters();

    const std::vector<int> &getInput_height() const;
    void setInput_height(const std::vector<int>& input_height);
    const std::vector<int> &getInput_width() const;
    void setInput_width(const std::vector<int>& input_width);
    const std::vector<int> &getInput_channel() const;
    void setInput_channel(const std::vector<int>& input_channel);

    const std::vector<int> &getKernel_dimension() const;
    void setKernel_dimension(const std::vector<int>& kernel_dimension);
    const std::vector<int> &getKernel_size() const;
    void setKernel_size(const std::vector<int> &kernel_size);
    const std::vector<int> &getKernel_channel() const;
    void setKernel_channel(const std::vector<int> &kernel_channel);

    const std::vector<int> &getOutput_height() const;
    void setOutput_height(const std::vector<int> &output_height);
    const std::vector<int> &getOutput_width() const;
    void setOutput_width(const std::vector<int> &output_width);
    const std::vector<int> &getOutput_channel() const;
    void setOutput_channel(const std::vector<int> &output_channel);

    int getLayer_number() const;
    void setLayer_number(int layer_number);

private:
    int layer_number;

    std::vector<int> input_height;
    std::vector<int> input_width;
    std::vector<int> input_channel;

    std::vector<int> kernel_dimension;
    std::vector<int> kernel_size;
    std::vector<int> kernel_channel;

    std::vector<int> output_height;
    std::vector<int> output_width;
    std::vector<int> output_channel;

    std::string cfg_file_name;
    File_utils<T> *cfg_util;
    std::vector<std::string> network_cfg_description;
};

template <class T>
Network<T>::Network(std::string cfg_file_name) {
    this->cfg_file_name = cfg_file_name;
}

template <class T>
Network<T>::~Network() {
    input_height.clear();
    input_width.clear();
    input_channel.clear();

    kernel_dimension.clear();
    kernel_size.clear();
    kernel_channel.clear();

    output_height.clear();
    output_width.clear();
    output_channel.clear();

    delete(cfg_util);
}

template <class T>
void Network<T>::initialize() {
    cfg_util = new File_utils<T>(cfg_file_name);
    cfg_util->parse_file();
}

template<class T>
std::string Network<T>::get_parameters() {
    std::string parameters = std::to_string(layer_number);
    parameters += "\n";
    for (int i = 0; i < layer_number; i++) {
        parameters += std::to_string(input_height[i]);
        parameters += " ";
        parameters += std::to_string(input_width[i]);
        parameters += " ";
        parameters += std::to_string(input_channel[i]);
        parameters += " ";

        parameters += std::to_string(kernel_dimension[i]);
        parameters += " ";
        parameters += std::to_string(kernel_size[i]);
        parameters += " ";
        parameters += std::to_string(kernel_channel[i]);
        parameters += " ";

        parameters += std::to_string(output_height[i]);
        parameters += " ";
        parameters += std::to_string(output_width[i]);
        parameters += " ";
        parameters += std::to_string(output_channel[i]);
        parameters += " ";
        parameters += "\n";
    }
    return parameters;
}


template<class T>
const std::vector<int> &Network<T>::getInput_height() const {
    return input_height;
}

template<class T>
void Network<T>::setInput_height(const std::vector<int> &input_height) {
    Network::input_height = input_height;
}

template<class T>
const std::vector<int> &Network<T>::getInput_width() const {
    return input_width;
}

template<class T>
void Network<T>::setInput_width(const std::vector<int> &input_width) {
    Network::input_width = input_width;
}

template<class T>
const std::vector<int> &Network<T>::getInput_channel() const {
    return input_channel;
}

template<class T>
void Network<T>::setInput_channel(const std::vector<int> &input_channel) {
    Network::input_channel = input_channel;
}

template<class T>
const std::vector<int> &Network<T>::getKernel_dimension() const {
    return kernel_dimension;
}

template<class T>
void Network<T>::setKernel_dimension(const std::vector<int> &kernel_dimension) {
    Network::kernel_dimension = kernel_dimension;
}

template<class T>
const std::vector<int> &Network<T>::getKernel_size() const {
    return kernel_size;
}

template<class T>
void Network<T>::setKernel_size(const std::vector<int> &kernel_size) {
    Network::kernel_size = kernel_size;
}

template<class T>
const std::vector<int> &Network<T>::getKernel_channel() const {
    return kernel_channel;
}

template<class T>
void Network<T>::setKernel_channel(const std::vector<int> &kernel_channel) {
    Network::kernel_channel = kernel_channel;
}

template<class T>
const std::vector<int> &Network<T>::getOutput_height() const {
    return output_height;
}

template<class T>
void Network<T>::setOutput_height(const std::vector<int> &output_height) {
    Network::output_height = output_height;
}

template<class T>
const std::vector<int> &Network<T>::getOutput_width() const {
    return output_width;
}

template<class T>
void Network<T>::setOutput_width(const std::vector<int> &output_width) {
    Network::output_width = output_width;
}

template<class T>
const std::vector<int> &Network<T>::getOutput_channel() const {
    return output_channel;
}

template<class T>
void Network<T>::setOutput_channel(const std::vector<int> &output_channel) {
    Network::output_channel = output_channel;
}

template<class T>
int Network<T>::getLayer_number() const {
    return layer_number;
}

template<class T>
void Network<T>::setLayer_number(int layer_number) {
    Network::layer_number = layer_number;
}

/***************************************************************/
/* Do not modify the above code.
   You are allowed to use the following global variables in your
   code. These are defined above.

   Begin your code here 	  			       */
/***************************************************************/

template <class T>
int Network<T>::obtain_parameters() {
    network_cfg_description = cfg_util->getFile_contents();

    layer_number = 0;

    input_height.clear();
    input_width.clear();
    input_channel.clear();

    kernel_dimension.clear();
    kernel_size.clear();
    kernel_channel.clear();

    output_height.clear();
    output_width.clear();
    output_channel.clear();
    /* Part I */
    /* Write your code here */

    //VARIABLES
    std::string curr_layer;
    int layertype = 0;
    int current_height = 0;
    int current_width = 0;
    int current_channels = 0;
    int filters = -1;
    int size = -1;
    int stride = -1;
    int pad = -1;
    
    std::map<std::string, std::string> params;
    for (const auto& str : network_cfg_description) {
        //DEBUG print statements
        //std::cout << str << " ";
        //std::cout << std::endl;
        
        if(str == "[net]"){
            layertype = 1;
//            printf("\nbegin parsing [net] layer\n");
            continue;
        }
        else if(str == "[convolutional]"){
            layer_number++;
//            printf("begin parsing [conv] layer, %d and counting\n", layer_number);
            layertype = 2;
            continue;
        }
        else if(str == "[maxpool]"){
            layertype = 3;
//            printf("begin parsing [pooling] layer\n");
            continue;
        }
        else{
            //printf("entering a layer\n");
            // Parse key=value pairs
            size_t eq_pos = str.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = str.substr(0, eq_pos);
                std::string value = str.substr(eq_pos + 1);
                params[key] = value;
                
                if(key == "filters"){
                    filters = std::stoi(value);
                }
                if(key == "size"){
                    size = std::stoi(value);
                }
                if(key == "stride"){
                    stride = std::stoi(value);
                }
                if(key == "pad"){
                    pad = std::stoi(value);
                }

                if(layertype == 1){
                    if (key == "height") {
                        current_height = std::stoi(value);
                    } else if (key == "width") {
                        current_width = std::stoi(value);
                    } else if (key == "channels") {
                        current_channels = std::stoi(value);
                    }
                }
            }
            if(layertype == 2){
                //DEBUG PRINT STATEMENTS
                /*
                printf("current_height = %d\n", current_height);
                printf("current_width = %d\n", current_width);
                printf("current_channel = %d\n", current_channels);
                */
//                printf("%d %d %d %d %d %d\n", filters, size, stride, pad, current_height, current_width);
                //layer_number = layer_number +2;
               
                /*
                int filters = std::stoi(params["filters"]);
                int size = std::stoi(params["size"]);
                int stride = std::stoi(params["stride"]);
                int pad = std::stoi(params["pad"]);
                */
                if(pad > -1){
                    int actualpad = (pad == 1) ? size/2 : 0;
                    int outH = (current_height+2*actualpad-size)/stride+1;
                    int outW = (current_width+2*actualpad-size)/stride+1;
    
                    input_height.push_back(current_height);
                    input_width.push_back(current_width);
                    input_channel.push_back(current_channels);
                    kernel_dimension.push_back(filters);
                    kernel_size.push_back(size);
                    kernel_channel.push_back(current_channels);
//                    printf("%d %d %d %d\n", current_height, current_width, outH, outW);
                    output_height.push_back(outH);
                    output_width.push_back(outW);
                    output_channel.push_back(filters);
                    
                    current_height = outH;
                    current_width = outW;
                    current_channels = filters;

                    filters = -1;
                    size = -1;
                    stride = -1;
                    pad = -1;
                }
                else{
//                    printf("not done parsing conv layer yet\n");
                }
            }
            if(layertype == 3){
                if(stride > -1){
                    int actualpad = 0;
                    int outH = (current_height+2*actualpad-size)/stride+1;
                    int outW = (current_width+2*actualpad-size)/stride+1;
                    //output_height.push_back(outH);
                    //output_width.push_back(outW);
                    current_height = outH;
                    current_width = outW;        
                }
            }
            if(layertype == 0){
                printf("empty file/unable to parse layers\n");
            }
        }
    }
    
    return 0;
}

template <class T>
int Network<T>::conv_convert(int layer_id, int padding, int stride, Array3D<T>& initial_input, Array4D<T>& initial_kernel,
                              Array2D<T>& input_matrix, Array2D<T>& kernel_matrix) {
    /* Part II */
    /*Write your code here*/
    
    //DEBUG PRINT STATEMENTS
    
    // printf("entering conv_convert()\n");
    // printf("input_height: %d ", initial_input.Size_3d()); 
    // //printf("\n");
    // printf("input_width: %d ", initial_input.Size_2d());
    // //printf("\n");
    // printf("input_channel: %d ", initial_input.Size_1d());
    // printf("\n");
    // printf("filters: %d kernel_height: %d kernel_width: %d kernel_channel: %d\n", initial_kernel.Size_4d(), initial_kernel.Size_3d(), initial_kernel.Size_2d(), initial_kernel.Size_1d());
    // printf("padding: %d stride: %d\n", padding, stride);    

    int input_height = initial_input.Size_3d();
    int input_width = initial_input.Size_2d();
    int input_channel = initial_input.Size_1d();

    int filters = initial_kernel.Size_4d();
    int kernel_height = initial_kernel.Size_3d();
    int kernel_width = initial_kernel.Size_2d(); //I think it doesn't matter here which one I use since height and width is the same b/c square kernel
    int kernel_channel = initial_kernel.Size_1d(); //should be the same as input_channel

    if(kernel_channel != input_channel){
        printf("kernel channels does not match input channels\n");
        return -1;
    }
    if(kernel_width != kernel_height){
        printf("kernel is not square, not supported\n");
        return -1;
    }
   
    int output_width = (input_width + 2 * padding - kernel_width) / stride + 1;
    int output_height = (input_height + 2 * padding - kernel_height) / stride + 1;

    if(output_width <= 0 || output_height <= 0){
        printf("invalid output dimension");
        return -1;
    }
    
    //pad the 3d input
    int padded_ow = input_width + padding*2;
    int padded_oh = input_height + padding*2;
    Array3D<T> padded_ii(padded_oh, padded_ow, input_channel);
//    printf("(%d %d) ", input_width, input_height);
//   printf("(%d %d)\n", padded_ow, padded_oh);

    //zero out the array first otherwise I get shit like -1170624351
    //channel -> width -> height
    for (int c = 0; c < input_channel; c++) {
        for (int w = 0; w < padded_ow; w++) {
            for (int h = 0; h < padded_oh; h++) {
//                printf("(%d, %d, %d)\n", h, w, c);
                padded_ii[h][w][c] = 0;
            }
        }
    }

    //copy elements over
    //channel -> width -> height
    for (int h = 0; h < input_height; h++) {
        for (int w = 0; w < input_width; w++) {
            for (int c = 0; c < input_channel; c++) {
//                printf("(%d, %d, %d)\n", h, w, c);
                padded_ii[h+padding][w+padding][c] = initial_input[h][w][c];

            }
        }
    }
    
    //input and kernel matrix dimensions
    int width = kernel_height * kernel_height * input_channel;
    int height = output_width * output_height;
    /*
    printf("output width: %d output height: %d\n", output_width, output_height);
    printf("padded output width: %d padded output height: %d\n", padded_ow, padded_oh);
    printf("output matrix width: %d output matrix height: %d\n", width, height);
    printf("kernel_matrix dimensions(%d, %d)\n", width, filters);
    */
    input_matrix.resize(height, width);
    //kernel_matrix.resize(filters, width);
    kernel_matrix.resize(width, filters);
    
    // //Construct input_matrix
    int col = 0;
    for (int h_out = 0; h_out < output_height; h_out++) {
        for (int w_out = 0; w_out < output_width; w_out++) {
            int row = 0;
            
            for (int h = 0; h < kernel_height; h++) {
                for (int w = 0; w < kernel_height; w++) {
                    for (int c = 0; c < input_channel; c++) {
                        int h_in = h_out * stride + h;
                        int w_in = w_out * stride + w;
                        input_matrix[col][row] = padded_ii[h_in][w_in][c];
                        row++;
                    }
                }
            }
            col++;
        }
    }

    // Construct kernel_matrix
    for (int i = 0; i < filters; i++) {
        int idx = 0;
        for (int h = 0; h < kernel_height; h++) {
            for (int w = 0; w < kernel_width; w++) {
                for (int c = 0; c < kernel_channel; c++) {
                    kernel_matrix[idx][i] = initial_kernel[i][h][w][c];
                    //printf("value = %d, idx=%d, i=%d, c=%d, w=%d, h=%d\n", kernel_matrix[idx][i], idx, i, c, w, h);
                    idx++;
                }
            }
        }
    }

    return 0;
}


template <class T>
int Network<T>::conv_convert_stream(int layer_id, int padding, int stride, Stream<T> &input, Stream<T> &output) {
    int buffer_size = (input_width[layer_id] + padding * 2) * input_channel[layer_id] * kernel_size[layer_id];
    T buffer[buffer_size];
    /* Part III */
    /*Write your code here*/

    //VARIABLES
    int input_w = input_width[layer_id];
    int input_h = input_height[layer_id];
    int input_c = input_channel[layer_id];
    int kernel_sz = kernel_size[layer_id];
    int padded_w = input_w + padding*2;
    int padded_h = input_h + padding*2;

    int output_w = (padded_w - kernel_sz)/stride + 1;
    int output_h = (padded_h - kernel_sz)/stride + 1;
    
    //printf("%d, %d, %d, %d\n", input_w, input_h, input_c, kernel_sz);

    /*
    printf("%d %d %d %d\n", input_w, input_h, input_c, kernel_sz);
    printf("(%d, %d)\n", padded_w, padded_h);
    printf("(%d, %d)\n", output_w, output_h);
    */

    //init
    int current_padded_row = 0;
    for (int r = 0; r < kernel_sz; r++) {
        if (current_padded_row < padding || current_padded_row >= (padding + input_h)) {
            for (int i = 0; i < padded_w * input_c; i++) {
                buffer[r * padded_w * input_c + i] = 0;
            }
        } else {
            for (int i = 0; i < padding * input_c; i++) {
                buffer[r * padded_w * input_c + i] = 0;
            }
            for (int i = 0; i < input_w * input_c; i++) {
                if (!input.empty())
                    buffer[r * padded_w * input_c + padding * input_c + i] = input.read();
                    //printf("input: %d\n", buffer[r * padded_w * input_c + padding * input_c + i]);
                else
                    buffer[r * padded_w * input_c + padding * input_c + i] = 0;
            }
            for (int i = padding * input_c + input_w * input_c; i < padded_w * input_c; i++) {
                buffer[r * padded_w * input_c + i] = 0;
            }
        }
        current_padded_row++;
    }

    //current window
    for (int i = 0; i < output_h; i++) {//Vertical
        for (int j = 0; j < output_w; j++) {//Horizontal
            int col_base = j * stride;
            for (int kr = 0; kr < kernel_sz; kr++) {
                for (int kc = 0; kc < kernel_sz; kc++) {
                    for (int ch = 0; ch < input_c; ch++) {
                        int index = kr * padded_w * input_c + ((col_base + kc) * input_c) + ch;
                        T value = buffer[index];
                        //printf("current buffer value: %d\n", buffer[index]);
                        output.write(value);
                    }
                }
            }
        }
        //slide down to next window
        if (i < output_h - 1) {
            for (int s = 0; s < stride; s++) {
                for (int r = 0; r < kernel_sz - 1; r++) {
                    for (int c = 0; c < padded_w * input_c; c++) {
                        buffer[r * padded_w * input_c + c] = buffer[(r + 1) * padded_w * input_c + c];
                    }
                }
                if (current_padded_row < padded_h) {
                    if (current_padded_row < padding || current_padded_row >= (padding + input_h)) {
                        for (int i = 0; i < padded_w * input_c; i++) {
                            buffer[(kernel_sz-1) * padded_w * input_c + i] = 0;
                        }
                    } else {
                        for (int i = 0; i < padding * input_c; i++) {
                            buffer[(kernel_sz-1) * padded_w * input_c + i] = 0;
                        }
                        for (int i = 0; i < input_w * input_c; i++) {
                            if (!input.empty())
                                buffer[(kernel_sz-1) * padded_w * input_c + padding * input_c + i] = input.read();
                                //printf("input: %d\n", buffer[r * padded_w * input_c + padding * input_c + i]);
                            else
                                buffer[(kernel_sz-1) * padded_w * input_c + padding * input_c + i] = 0;
                        }
                        for (int i = padding * input_c + input_w * input_c; i < padded_w * input_c; i++) {
                            buffer[(kernel_sz-1) * padded_w * input_c + i] = 0;
                        }
                    }
                } else {  
                    for (int c = 0; c < padded_w * input_c; c++) {
                        buffer[(kernel_sz - 1) * padded_w * input_c + c] = 0;
                    }
                }
                current_padded_row++;
            }
        }
    }
    return 0;
}

#endif //NETWORK_H
