#include "Halide.h"
#include "halide_image_io.h"
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace Halide;
using namespace Halide::Tools;

//forward declaration of the network function
Buffer<float> TrafficSignMPeemen(Buffer<float> input);


Buffer<float> load_input_image(const char* fname){


    //Load input image
    Buffer<uint8_t> img = load_and_convert_image(fname);

    //Sanity check on input
    assert(img.channels()==3 && "Code expects 3 channel input, please convert your input image");

    //convert char to float
    Var n("n"),m("m"),o("o");
    Func float_img("float_img");
    float_img(n,m,o)=cast<float>(img(n,m,o));

    //rgb to YUV conversion
    Func yuv_img("yuv_img");
    yuv_img(n,m,o) = cast<float>(0);
    yuv_img(n,m,0) =  16.0f +( 0.257f) * float_img(n,m,0) +( 0.504f) * float_img(n,m,1) + ( 0.098f) * float_img(n,m,2);
    yuv_img(n,m,1) =  128.0f +( -0.148f) * float_img(n,m,0) +( -0.291f) * float_img(n,m,1) + ( 0.439f) * float_img(n,m,2);
    yuv_img(n,m,2) =  128.0f +( 0.439f) * float_img(n,m,0) +( -0.368f) * float_img(n,m,1) + ( -0.071f) * float_img(n,m,2);

    //scaling to values in [0-1]
    Func scaled_img("scaled_img");
    scaled_img(n,m,o) = yuv_img(n,m,o)/(255.f);

    //return the scaled image buffer
    return scaled_img.realize(1280,720,3);
}

int main(int argc, char** argv){

    //sanity check arguments
    if(argc!=2){
        cerr << "Usage: "<< argv[0] <<" input_image.png"<< endl;
        return -1;
    }

    //load the input image
    Buffer<float> img = load_input_image((const char*) argv[1]);

    //evaluate the network on the specified input image
    Buffer<float> output =  TrafficSignMPeemen(img);

    //speed sign classes
    const int max_speed[8]={0, 30, 50, 60, 70, 80, 90, 100};

    //Check output agains threshold and report detected traffic signs
    for(int m=0;m<output.height();m++){
        for(int n=0;n<output.width();n++){
            if(output(n,m,0)>=1.6f){
                float max=-16.0f;
                int max_idx=0;
                for(int o=1;o<output.channels();o++){
                    if (output(n,m,o)>max){
                        max=output(n,m,o);
                        max_idx=o;
                    }
                }
                std::cout << n*4 <<", "<<m*4<<", "<< max_speed[max_idx]<<"km/h confidence: "<< 1.f/(1.f+exp(-1.f*max))<< endl;
            }
        }
    }

    return 0;
}
