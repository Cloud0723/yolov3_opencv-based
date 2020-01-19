/**
 * file CameraCtl.hpp
 * The CameraCtl Class for using hikvision camera in opencv easily.
 * by Dinger
 * complement: hqy
 * last change: 2019.12.13
 */

#ifndef __CAMERA_CTL_HPP
#define __CAMERA_CTL_HPP

#include <stdio.h>
#include <string.h>
#include "MvCameraControl.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <unistd.h>

using namespace cv;

#define MAX_IMAGE_DATA_SIZE   (40*1024*1024)

enum GAIN_MODE{NONE=0, ONCE=1, CONTINUOUS=2};
enum BITRATE{               //kbps
    DEFAULT =4096,          //默认码率4096kbps
    HIGH    =6144,           
    ULTRA   =8192, 
    MAXIMUM =16383,         //Hikvision sdk能接受的最大码率
    MEDIUM  =2048, 
    LOW     =1024,
    MINIMUM =128            //最小码率
    };

class CameraCtl
{
private:
    int nRet;
    void* handle;
    unsigned char * pData;
    bool grabing;
    bool printDeviceInfo(MV_CC_DEVICE_INFO* pstMVDevInfo);
    VideoWriter writer;
public:
    CameraCtl();
    ~CameraCtl();
    int startGrabbing();
    int stopGrabbing();
    Mat getOpencvMat(int Msec=1000);
    int setExposureTime(const float t){             //设置曝光时间
        nRet=MV_CC_SetExposureTime(handle, t);
        if(nRet != MV_OK){
            printf("Failed to set exposure time. nRet [%x]\n", nRet);
            return -1;
        }
        return 0;
    }

    int setGainMode(GAIN_MODE enumMode=NONE){               //设置自动增益
        nRet=MV_CC_SetGainMode(handle, enumMode);
        if(nRet != MV_OK){
            printf("Failed to set gain mode. nRet [%x]\n", nRet);
            return -1;
        }
        return 0;
    }               

    int getGainMode(bool disp=false){                       //获取自动增益信息,正常值0,1,2
        //disp选项：是否打印信息
        MVCC_ENUMVALUE* pstValue=new MVCC_ENUMVALUE;
        nRet=MV_CC_GetGainMode(handle, pstValue);
        int res=0;
        if(nRet != MV_OK){
            printf("Get gain mode failed. nRet [%x]\n", nRet);
            printf("Exception: %x.\n", nRet);
            delete pstValue;
            return -1;
        }
        if(disp){
            printf("Current mode:%d. ", pstValue->nCurValue);
            switch(pstValue->nCurValue){
                case 0: printf("Auto gain is not set up.\n");break;
                case 1: printf("Auto gain is triggered once.\n");break;
                case 2: printf("Continuous auto gain.\n");break;
                default: printf("Unknown parameter.\n"); delete pstValue; return -1;
            }
        }
        res = pstValue->nCurValue;
        delete pstValue;
        return res;
    }       

    int setGain(const float gain){                          //设置固定值自动增益
        nRet=MV_CC_SetGain(handle, gain);
        if(nRet != MV_OK){
            printf("Failed to set gain. nRet [%x]\n", nRet);
            return -1;
        }
        return 0;
    }       

    int setFrameRate(const float rate){                //设置相机帧率
        nRet=MV_CC_SetFrameRate(handle, rate);
        if(nRet != MV_OK){
            printf("Failed to set frame rate. nRet [%x]\n", nRet);
            return -1;
        }
        return 0;
    }

    int setAutoExposure(const unsigned int mode){           //设置自动曝光
        if(mode>2){
            printf("Input param error: digit should be in range(0,3).\n");
            return -1;
        }
        nRet=MV_CC_SetExposureAutoMode(handle, mode);
        if(nRet != MV_OK){
            printf("Failed to set auto exposure mode. nRet [%x]\n", nRet);
            return -1;
        }
        return 0;
    }

    int getAutoExposure(bool disp=false){                   //获取自动曝光信息，正常为0,1,2
        //disp选项：是否打印信息
        MVCC_ENUMVALUE* tmp=new MVCC_ENUMVALUE;
        int res;
        nRet=MV_CC_GetExposureAutoMode(handle, tmp);
        if(nRet != MV_OK){
            printf("Fail to get auto exposure mode info. nRet [%x]\n", nRet);
            delete tmp;
            return -1;
        }
        if(disp){
            printf("Current mode:%d. ", tmp->nCurValue);
            if(tmp->nCurValue==1){
                printf("Auto exposure: ONCE.\n");
            }
            else if(!tmp->nCurValue){
                printf("Auto exposure: OFF.\n");
            }
            else if(tmp->nCurValue==2){
                printf("Auto exposure: CONTINUOUS.\n");
            }
            else printf("Unknown status.\n");
        }
        res = tmp->nCurValue;
        delete tmp;
        return res;
    }

    int loadFeature(const char *path){                       //加载相机参数设置  
        nRet=MV_CC_FeatureLoad(handle, path);
        if(nRet != MV_OK){
            printf("Failed to load camera features. You may check your filepath. nRet [%x]\n", nRet);
            }
        return 0;
    }

    int saveFeature(const char *path){                       //保存相机参数设置
        nRet=MV_CC_FeatureSave(handle, path);
        if(nRet != MV_OK){
            printf("Failed to save camera features. You may check your filepath. nRet [%x]\n", nRet);
            return -1;
        }
        return 0;
    }

    float getExposureTime(bool disp=false){                 //返回曝光时间,正常值为正数
        //disp选项：是否打印信息
        MVCC_FLOATVALUE *tmpF=new MVCC_FLOATVALUE;
        nRet=MV_CC_GetExposureTime(handle, tmpF);
        float res;
        if(nRet != MV_OK){
            printf("Fail to get exposure time. nRet [%x]\n", nRet);
            delete tmpF;
            return -1;
        }
        if(disp){
            printf("Current exposure time: %f.\n", tmpF->fCurValue);
            printf("With minimum value:%f, and maxinum value:%f.\n",
                tmpF->fMin, tmpF->fMax);
        }
        res=tmpF->fCurValue;
        delete tmpF;
        return res;
    }     

    float getGain(bool disp=false){                         //返回当前增益值,正常值非负
        //disp选项：是否打印信息    
        MVCC_FLOATVALUE *tmpF=new MVCC_FLOATVALUE;
        nRet=MV_CC_GetGain(handle, tmpF);
        float res;
        if(nRet != MV_OK){
            printf("Fail to get gain. nRet [%x]\n", nRet);
            delete tmpF;
            return -1;
        }
        if(disp){
            printf("Current gain: %f.\n", tmpF->fCurValue);
            printf("With minimum value:%f, and maxinum value:%f.\n",
                tmpF->fMin, tmpF->fMax);
        }
        res=tmpF->fCurValue;
        delete tmpF;
        return res;
    }

    float getFrameRate(bool disp=false){                    //获取相机帧率,正常为正数
        //disp选项：是否打印信息
        MVCC_FLOATVALUE *tmpF=new MVCC_FLOATVALUE;
        nRet=MV_CC_GetFrameRate(handle, tmpF);
        float res;
        if(nRet != MV_OK){
            printf("Fail to get frame rate. nRet [%x]\n", nRet);
            delete tmpF;
            return -1;
        }
        if(disp){
            printf("Current frame rate: %f.\n", tmpF->fCurValue);
            printf("With minimum value:%f, and maxinum value:%f.\n",
                tmpF->fMin, tmpF->fMax);
        }
        res=tmpF->fCurValue;
        delete tmpF;
        return res;
    } 
};

#endif // __CAMERA_CTL_HPP
