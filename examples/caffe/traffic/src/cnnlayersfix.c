/****************************************************************************************
 * File:   cnnlayersfix.c
 * Author: OMITTED FOR BLIND REVIEW
 * Email:  OMITTED FOR BLIND REVIEW
 ****************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define LINE_BUFFER_SIZE 100  //Line buffer size for read write

#define YIN 720  //input frame size
#define XIN 1280

#define YL1 357 //CNN layer 1 size
#define XL1 637
#define YL2 177 //CNN layer 2 size
#define XL2	317
#define YL3	172 //CNN layer 3 size
#define XL3	312

void read_weight(const char filename[], int size, short matrix[])
{	/************************************************************************************
	* Function: void read_weight(const char filename[], int size, short matrix[])
	* Input   : char array containing the filename of the coefficient file, number of weights
    * Output  : matrix filled with weights for each feature map in a layer
    * Procedure: read all weights from file and strore in array
    ************************************************************************************/
	FILE* finput;

	finput = fopen(filename , "rb");
	if (finput==NULL) {printf("File error while loading %s\n\r", filename);}

	fread(matrix, sizeof(short), size, finput);
	fclose(finput);
}

void read_act(const char filename[], int size, unsigned char matrix[])
{	/************************************************************************************
	* Function: void read_act(const char filename[], int size, unsigned char matrix[])
	* Input   : char array containing the filename of the activation LUT, number of values
    * Output  : array filled with LUT values
    * Procedure: read all values from file and strore in array
    ************************************************************************************/
	FILE* finput;

	finput = fopen(filename , "rb" );
	if (finput==NULL) {printf("File error while loading %s\n\r", filename);}

	fread(matrix, sizeof(char), size, finput);
	fclose(finput);
}

void read_image_pgm(unsigned char image[],const char filename[], int imageWidth, int imageHeight)
{	/************************************************************************************
	* Function: void read_image_pgm(unsigned char image[], const char filename[], int imageWidth, int imageHeight)
	* Input   : uchar array pointer for output result, char array with frame filename, int frame width, int frame height
    * Output  : uchar frame array pointer
    * Procedure: if image dimensions and layout pgm is correct image is read from file to image array
    ************************************************************************************/
	int grayMax;
	int PGM_HEADER_LINES=3;
	FILE* input;

	int headerLines = 1;
	int scannedLines= 0;
	long int counter =0;

	//read header strings
	char *lineBuffer = (char *) malloc(LINE_BUFFER_SIZE);
	char *split;
	char *format = (char *) malloc(LINE_BUFFER_SIZE);
	char P5[]="P5";
	char comments[LINE_BUFFER_SIZE+1];

	//read the input PGM file header
	input=fopen(filename, "rb");

	//read the input PGM file header
	while(scannedLines < headerLines){
		fgets(lineBuffer, LINE_BUFFER_SIZE, input);
		//if not comments
		if(lineBuffer[0] != '#'){
			scannedLines += 1;
			//read the format
			if(scannedLines==1){
				split=strtok(lineBuffer, " \n");
				strcpy(format,split);
				if(strcmp(format,P5) == 0){
					//printf("FORMAT: %s\n",format);
					headerLines=PGM_HEADER_LINES;
				}
				else
				{
					printf("Only PGM P5 format is support. \n");
				}
			}
			//read width and height
			if (scannedLines==2)
			{
				split=strtok(lineBuffer, " \n");
				if(imageWidth == atoi(split)){ //check if width matches description
					//printf("WIDTH: %d, ", imageWidth);
				}
				else{
					printf("input frame has wrong width should be WIDTH: %d, ", imageWidth);
					exit(4);
				}
				split = strtok (NULL, " \n");
				if(imageHeight == atoi(split)){ //check if heigth matches description
					//printf("HEIGHT: %d\n", imageHeight);
				}
				else{
					printf("input frame has wrong height should be HEIGHT: %d, ", imageHeight);
					exit(4);
				}
			}
			// read maximum gray value
			if (scannedLines==3)
			{
				split=strtok(lineBuffer, " \n");
				grayMax = atoi(split);
				//printf("GRAYMAX: %d\n", grayMax);
			}
		}
		else
		{
			strcpy(comments,lineBuffer);
			//printf("comments: %s", comments);
		}
	}

	counter = fread(image, sizeof(unsigned char), imageWidth * imageHeight, input);
	//printf("pixels read: %d\n",counter);

	//close the input pgm file and free line buffer
	fclose(input);
	free(lineBuffer);
	free(format);
}

void write_image_pgm(unsigned char image[], const char filename[], int imageWidth, int imageHeight){
	/************************************************************************************
	* Function: void write_image_pgm(unsigned char image[], const char filename[], int imageWidth, int imageHeight)
	* Input   : uchar array pointer with video frame, char array with frame filename, int frame width, int frame height
    * Output  :
    * Procedure: if image dimensions and layout pgm is correct image is written to a file in pgm format
    ************************************************************************************/
    FILE * output;
    output = fopen(filename, "wb");
    fprintf(output, "P5\n");
    fprintf(output, "%d %d\n255\n",imageWidth, imageHeight);
    fwrite(image, sizeof(unsigned char), imageWidth * imageHeight, output);
    fclose(output);
}

void convolution_layer1(unsigned char in_layer[], unsigned char out_layer[], const short bias[], const short weight[], const unsigned char fixact[])
{	/************************************************************************************
	* Function: convolution_layer1(unsigned char in_layer[], unsigned char out_layer[], const short bias[], const short weight[], const char fixact[])
	* Input   : input image, pointer to output result, coefficients bias, weights, and activation function LUT
    * Output  : neuron outputs of the feature maps represented as an image
    * Procedure: perform feed forward computation through the feature extraction layers with subsampling S=2
    ************************************************************************************/
  int k,l,m,n,r;
  short index;
  float acc;
  int iacc;

  FILE* fp = fopen("layer1.txt", "wt");

  //loops over output feature maps
  for(r=0; r<6; r++){
    //convolve weight kernel with input image
    for(m=0; m<YL1; m++){//shift input window over image
      for(n=0; n<XL1; n++){//unroll 4 times improve locality reduce loop overhead
		//init values of feature map at bias value
		acc = (float)(bias[r])/2048.0f;

        //multiply input window with kernel
	    for(k=0; k<6; k++){
	      for(l=0; l<6; l++){
			acc += ((float)in_layer[(m*2+k)*XIN+n*2+l])/255.0f * ((float)(weight[6*(r*6+k)+l]))/2048.0f;
		  }
        }


        //scale back to work with this sigmoid function
        iacc = (int)(acc*512.0f);

        //fprintf(fp, "%f\n", ((float)acc));
		////saturate shift to scale back the result into 10 bit fixed-point
		if (iacc>=512){iacc=0x01FF;}
        else if (iacc<-512){iacc=0x0200;}

        index=iacc&0x000003FF;

        //debug printing
        //if(m==16 && n==34)
        //    printf("%f, acc[%d] = fixact[%d]: %f\n",acc, (int)((fmin(1.0f, fmax(-1.0f,acc))+1.0f)*511.0f),index, ((float)(fixact[index]))/255.0f);

		////lookup for correct activation function value
		out_layer[XL1*(r*YL1+m)+n]=fixact[index];
      }
    }
  }
  fclose(fp);
}

void convolution_layer2(unsigned char in_layer[], unsigned char out_layer[], const short bias[], const short weight[], const unsigned char fixact[])
{	/************************************************************************************
	* Function: void convolution_layer2(unsigned char in_layer[], unsigned char out_layer[], const short bias[], const short weight[], const char fixact[])
	* Input   : input image, pointer to output result, coefficients bias, weights, and activation function LUT
    * Output  : the neuron outputs computed from the input pattern
    * Procedure: perform feed forward computation through second layer of feature extraction layers with subsampling S=2
    ************************************************************************************/
  int k,l,m,n,q,r,qindex;
  short index;
  int acc;
  float facc;

  //feature maps are sparse connected therefore connection scheme is used
  const int qq[54]={0,1,2, 1,2,3, 2,3,4, 3,4,5, 0,4,5, 0,1,5,
		    0,1,2,3, 1,2,3,4, 2,3,4,5, 0,3,4,5, 0,1,4,5, 0,1,2,5, 0,1,3,4, 1,2,4,5, 0,2,3,5};

  //loops over output feature maps with 3 input feature maps
  for(r=0; r<6; r++){
    //convolve weight kernel with input image
    for(m=0; m<YL2; m++){//shift input window over image
      for(n=0; n<XL2; n++){
	    //init values of feature map at 0
		facc = (float)(bias[r])/2048.0f;

		for(q=0; q<3; q++){//connect with all connected 3 input feature maps
		  qindex=qq[r*3+q];//lookup connection address
		  //multiply input window with kernel
		  for(k=0; k<6; k++){
	        for(l=0; l<6; l++){
	          facc += in_layer[XL1*(qindex*YL1+(m*2+k))+n*2+l]/255.0f * weight[(r*3+q)*36+k*6+l]/2048.0f;
		    }
          }
		}

        acc = (int)(facc*512.0f);
        //debug printing
        //if(m==0 && n==0)
         //   printf("facc %f -> %d\n",facc, acc);

		//saturate shift to scale back the result into 10 bit fixed-point
		//if (acc>=524288){index=0x01FF;}
		//else if (acc<-524288){index=0x0200;}
		//else {index=(acc>>10)&0x000003FF;}

        if (acc>=512){acc=0x01FF;}
        else if (acc<-512){acc=0x200;}
        index=acc&0x000003FF;

        //debug printing
        //if(m==0 && n==0){
        //     printf("%f, acc[%d] = fixact[%d]: %f\n",facc, (int)((fmin(1.0f, fmax(-1.0f,facc))+1.0f)*511.0f),index, ((float)(fixact[index]))/255.0f);
        //}

        //lookup for correct activation function value
		out_layer[XL2*(r*YL2+m)+n]=fixact[index];
	  }
    }
  }
  for(r=0; r<9; r++){//loop over output feature maps with 4 input maps
  //convolve weight kernel with input image
    for(m=0; m<YL2; m++){//shift input window over image
      for(n=0; n<XL2; n++){
	    //init values of feature map at bias value
	    acc=bias[r+6]<<8;
		for(q=0; q<4; q++){//connect with all connected 4 input feature maps
		  qindex=qq[r*4+q+18];//lookup feature map adress
		  //multiply input window with kernel
		  for(k=0; k<6; k++){
	        for(l=0; l<6; l++){
				acc+=in_layer[XL1*(qindex*YL1+(m*2+k))+n*2+l] * weight[(r*4+q+18)*36+k*6+l];
		    }
          }
		}

		//saturate shift to scale back the result into fixed-point
		if (acc>=524288){index=0x01FF;}
		else if (acc<-524288){index=0x0200;}
		else {index=(acc>>10)&0x000003FF;}

        //debug printing
        //if(m==0 && n==0){
        //    printf("%f\n",((float)(fixact[index]))/255.0f);
        //}

		out_layer[XL2*((r+6)*YL2+m)+n]=fixact[index];
	  }
    }
  }
  //compute last feature map connected with all 6 input feature maps
  //convolve weight kernel with input image
  for(m=0; m<YL2; m++){//shift input window over image
    for(n=0; n<XL2; n++){
	  //init values of feature map at bias value
	  acc = bias[15]<<8;
	  for(q=0; q<6; q++){//connect with all input feature maps
		//multiply input window with kernel
		for(k=0; k<6; k++){
	      for(l=0; l<6; l++){
	        acc += in_layer[XL1*(q*YL1+(m*2+k))+n*2+l] * weight[(54+q)*36+k*6+l];
		  }
        }
	  }

	  //saturate shift to scale back the result into fixed-point
	  if (acc>=524288){index=0x01FF;}
	  else if (acc<-524288){index=0x0200;}
      else {index=(acc>>10)&0x000003FF;}

      //debug printing
      //if(m==0 && n==0)
      //      printf("%f\n",((float)(fixact[index]))/255.0f);

	  out_layer[XL2*(15*YL2+m)+n]=fixact[index];
	}
  }
}

void convolution_layer3(unsigned char in_layer[], unsigned char out_layer[], const short bias[], const short weight[], const unsigned char fixact[])
{	/************************************************************************************
	* Function: void convolution_layer3(unsigned char in_layer[], unsigned char out_layer[], const short bias[], const short weight[], const char fixact[])
	* Input   : input image, pointer to output result, coefficients bias, weights, and activation function LUT
    * Output  : the neuron outputs computed from the input pattern
    * Procedure: perform feed forward computation through the neural network classification layer
    ************************************************************************************/
  int k,l,m,n,r,q;
  int acc;
  short index;
  float facc;


  //loops over first 40 output feature maps
  for(r=0; r<40; r++){
    //convolve weight kernel with input image
    for(m=0; m<YL3; m++){//shift input window over image
      for(n=0; n<XL3; n++){//unroll 4 times to improve locality and reduce loop overhead

		//init values of feature map at bias value
		//acc = bias[r]<<8;
		facc = ((float)(bias[r]))/2048.0f;
		for(q=0; q<8; q++){
		  for(k=0; k<5; k++){//there is no subsampling in this layer
	        for(l=0; l<5; l++){//only 5x5 convolution
              facc += (in_layer[XL2*(q*YL2+(m+k))+n+l])/255.0f * weight[25*(r*8+q)+k*5+l]/2048.0f;
    	    }
		  }
        }
        acc=(int)(facc*512.0f);

        if (acc>=512){acc=0x01FF;}
        else if (acc<-512){acc=0x200;}
        index=acc&0x000003FF;

        out_layer[XL3*(r*YL3+m)+n]=fixact[index];

     //   if(m==0 && n==0)
     //      printf("%d: %f %f\n",r+1, facc, ((float)(fixact[index]))/255.0f);
	  }
	}
  }

  //loops over second 40 output feature maps
  for(r=40; r<80; r++){
    //convolve weight kernel with input image
    for(m=0; m<YL3; m++){//shift input window over image
      for(n=0; n<XL3; n++){//unroll 4 times to improve locality and reduce loop overhead

		//init values of feature map at bias value
		//acc = bias[r]<<8;
		facc = bias[r]/2048.0f;
		for(q=8; q<16; q++){
		  for(k=0; k<5; k++){//there is no subsampling in this layer
	        for(l=0; l<5; l++){//only 5x5 convolution
              facc += in_layer[XL2*(q*YL2+(m+k))+n+l]/255.0f * weight[25*(r*8+(q-8))+k*5+l]/2048.0f;
    	    }
		  }
        }

        acc=(int)(facc*255.0f*2048.0f);
		if (acc>=524288){index=0x01FF;}
		else if (acc<-524288){index=0x0200;}
		else {index=(acc>>10)&0x000003FF;}
		out_layer[XL3*(r*YL3+m)+n]=fixact[index];
        //if(m==0 && n==0)
        //    printf("%d: %f %f\n",r+1, facc, ((float)(fixact[index]))/255.0f);
	  }
	}
  }

}

int convolution_layer4(unsigned char in_layer[], const short bias[], const short weight[], const unsigned char fixact[], unsigned short detect[])
{	/************************************************************************************
	* Function: void convolution_layer4(unsigned char in_layer[], const short bias[], const short weight[], const unsigned char fixact[], unsigned short detect[])
	* Input   : input image, coefficients bias and weights, activation function LUT
    * Output  : array with detection info (position x, y, sign class, and confidence) for each detection, number of detections
    * Procedure: perform feed forward computation through the neural network layer threshold with neuron output
	             to detect signs at pixel positions. compute detection position in the original frame
    ************************************************************************************/
  int m,n,q,r;
  int detections=0;
  short index;
  int acc, acc0;
  float facc0, facc;
  int max;
  int set=0;

  //convolve weight kernel with input image
  for(m=0; m<YL3; m++){//shift input window over image
    for(n=0; n<XL3; n++){
      //init values of feature map at bias value
	  //acc0 = bias[0]<<8;
	  facc0 = bias[0]/2048.0f;
	  for(q=0; q<80; q++){
	    facc0 +=  ((float)(in_layer[q*YL3*XL3+m*XL3+n]))/255.0f *  weight[q]/2048.0f;
	  }

     // if(n==10 && m==5){
     //     printf("L41 %f\n", facc0);
     // }

      acc0 = (int)(facc0*2048.0f*255.0f);

	  //detection threshold confidence 0.5 inverse of activation function is 0
	  //if(acc0>=0){// if sign detected figure out which sign
	    max=0;
        for(r=1; r<8; r++){// check other 7 maps for the strongest sign
          //acc = bias[r]<<8;
          facc = bias[r]/2048.0f;
          for(q=0; q<80; q++){
            facc += in_layer[q*YL3*XL3+m*XL3+n]/255.0f * weight[r*80+q]/2048.0f;
          }
     //     if(n==10 && m==5){
     //         printf("L4 %f\n", facc);
     //     }
          acc = (int)(facc*2048.0f*255.0f);
	      if (acc>=0 && acc>max && acc0>=0){
		    max=acc;
			detect[detections*4]=n*4;
			detect[detections*4+1]=m*4;
			detect[detections*4+2]=r;

			if (acc>=524288){index=0x01FF;}
 	        else {index=(acc>>10)&0x000003FF;}

			detect[detections*4+3]=fixact[index];
			set=1;
          }
        }
        if (set==1){//this means that a sign is found
          detections=detections+1;
	      set=0;
        }
	  //}
	}
  }
  return detections;
}

void annotate_img(unsigned char img[], unsigned short detectarray[], int detections)
/************************************************************************************
	* Function: void annotate_img(unsigned char img[], unsigned short detectarray[], int detections)
	* Input   : original input frame, array with detection info, number of detections
    * Output  : annotated frame
    * Procedure: draw detection boxes in the video frame according to the information in the dection array
    ************************************************************************************/
{
  int i,x,y,posx,posy;

  for(i=0; i<detections; i++){//loop over the obtained detections
    posx=detectarray[i*4];
	posy=detectarray[i*4+1];
    for(x=0; x<32; x++){ //draw the x lines of the box
	  img[posy*1280+posx+x]=255;
	  img[(posy+31)*1280+posx+x]=255;
	}
    for(y=1; y<31; y++){ //draw the y lines of the box
      img[(posy+y)*1280+posx]=255;
	  img[(posy+y)*1280+posx+31]=255;
    }
  }
}

int main(void)
{
    /************************************************************************************
	 * Function: int main (void)
	 * Input   :
    * Output  : 0 if succesfull
    * Procedure: load network coefficients, load images, detect signs in images, update maximum speed
    ************************************************************************************/

	// variable initialization
	int i;
	const int max_speed[8]={0, 30, 50, 60, 70, 80, 90, 100};//speed sign classes

	// image and feature map initialization
	static unsigned char in_image[YIN*XIN];//for input image
	static unsigned char fixnet_layer1[6*YL1*XL1];//intermediat feature map results
	static unsigned char fixnet_layer2[16*YL2*XL2];
 	static unsigned char fixnet_layer3[80*YL3*XL3];

	// fixed-point bias and weight coefficients
	static short fixbias1[6];
	static short fixweight1[6*36];
	static short fixbias2[16];
	static short fixweight2[(6*3+9*4+6)*36];
	static short fixbias3[80];
	static short fixweight3[25*8*80];
	static short fixbias4[8];
	static short fixweight4[80*8];

	// LUT for fixed-point signoid/activation function initialization
	static unsigned char fixact[1024];

	static unsigned short detectarray[4*10];//detection array with space for 10 detections
	int detections;

	 clock_t starttime, endtime; //vars to measure computation time

	// read the fixed-point look-up table
	printf("Reading the coefficient files\n");

	read_weight("bias01.bin", 6, fixbias1);
	read_weight("weight01.bin", 6*36, fixweight1);

	read_weight("bias02.bin", 16, fixbias2);
	read_weight("weight02.bin", 2160, fixweight2);

	read_weight("bias03.bin", 80, fixbias3);
	read_weight("weight03.bin", 25*8*80, fixweight3);

	read_weight("bias04.bin", 8, fixbias4);
	read_weight("weight04.bin", 80*8, fixweight4);

	//read the activation function
	read_act("fixact01.bin", 1024, fixact);

	//read image from file
	printf("Reading the image ...\n");

	read_image_pgm(in_image, "new.pgm", 1280, 720);

	printf("Calculation CNN...\n");


	//perform feed forward operation thourgh the network
	//start timer
    starttime=clock();
	convolution_layer1(in_image, fixnet_layer1, fixbias1, fixweight1, fixact);
    //stop timer
    endtime=clock();
    printf("  Elapsed time layer 1 is %f s\n", 1.0*(endtime-starttime)/CLOCKS_PER_SEC);

	//start timer
    starttime=clock();
	convolution_layer2(fixnet_layer1, fixnet_layer2, fixbias2, fixweight2,fixact);
	//stop timer
    endtime=clock();
    printf("  Elapsed time layer 2 is %f s\n", 1.0*(endtime-starttime)/CLOCKS_PER_SEC);

	//start timer
    starttime=clock();
	convolution_layer3(fixnet_layer2, fixnet_layer3, fixbias3, fixweight3,fixact);
	//stop timer
    endtime=clock();
    printf("  Elapsed time layer 3 is %f s\n", 1.0*(endtime-starttime)/CLOCKS_PER_SEC);

    //start timer
    starttime=clock();
	detections=convolution_layer4(fixnet_layer3, fixbias4, fixweight4, fixact, detectarray);
	//stop timer
    endtime=clock();
    printf("  Elapsed time layer 4 is %f s\n", 1.0*(endtime-starttime)/CLOCKS_PER_SEC);

	printf("number of detections = %d\n",detections);
    for(i=0; i<detections; i++){
      printf("detection nr %d = %d km/h, box pos= x %d, y %d, confidence = %f\n",i,max_speed[detectarray[i*4+2]], detectarray[i*4],detectarray[i*4+1],((float)detectarray[i*4+3])/255.0f);
    }

    annotate_img(in_image, detectarray, detections);

    write_image_pgm(in_image, "output.pgm", 1280, 720);
    printf("done!\n");
	return 0;
}
