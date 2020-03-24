#include <cstdlib>
#include "lime/LimeSuite.h"
#include <iostream>
#include "math.h"
#include <thread>
#include <memory.h>
using namespace std;
 
#define N_RADIOS 4
 
static lms_device_t* m_lms_device;
static int m_n_devices;
 
static int error()
{
    //print last error message
    cout << "ERROR:" << LMS_GetLastErrorMessage();
    if (m_lms_device != NULL)
        LMS_Close(m_lms_device);
    exit(-1);
}
 
void lime_terminate(void)
{
    //Close device
    LMS_Close(m_lms_device);
}

 
int main( void )
{
    // This blocks
    // Set all device handles to NULL
    m_lms_device = NULL;
 
    //Find devices
    int n;
    lms_info_str_t list[N_RADIOS]; //should be large enough to hold all detected devices
 
    if ((m_n_devices = LMS_GetDeviceList(list)) < 0) error();//NULL can be passed to only get number of devices
 
    cout << "Devices found: " << m_n_devices << endl; //print number of devices
 
    if (m_n_devices < 1) return -1; 

    for(int i = 0; i<N_RADIOS; i++){
	    //open the first device
	    if (LMS_Open(&m_lms_device, list[i], NULL)) error();
	 
	    //Initialize device with default configuration
	    //Do not use if you want to keep existing configuration
	    if (LMS_Init(m_lms_device) != 0) error();
	    //if (LMS_Reset(m_lms_device) != 0) error();

	    // just set EXTREF
	    if (LMS_SetClockFreq(m_lms_device, LMS_CLOCK_EXTREF, 10000000) != 0) error();
    
 
    cout << "EXTREF set" << endl;
    lime_terminate();
 }
    return 0;
}
