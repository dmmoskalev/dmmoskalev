//============================================================================
// Name        : ql.cpp
// Author      : dmitry moskalev
// Version     :
// Copyright   : ask me to get it
// Description : quick labeling on base of openCV find contours tool
//============================================================================

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <unistd.h>
#include <pthread.h>

#define HRC std::chrono::high_resolution_clock
#define VP std::vector<std::vector<cv::Point>>
#define V4i std::vector<cv::Vec4i>
#define LOGD(...)  do {printf(__VA_ARGS__);printf("\n");} while (0)
#define DBG(fmt, args...) LOGD("%s:%d, " fmt, __FUNCTION__, __LINE__, ##args);
#define ASSERT(b) \
do \
{ \
    if (!(b)) \
    { \
        LOGD("error on %s:%d", __FUNCTION__, __LINE__); \
        return 0; \
    } \
} while (0)

#define  NAME          "quick labeling"
#define  MAINVERSION    (  0)  /**<  Main Version: X.-.-   */
#define  VERSION        (  1)  /**<       Version: -.X.-   */
#define  SUBVERSION     (  1)  /**<    Subversion: -.-.X   */

struct Coordinates {
    int latch_X;
    int latch_Y;
    int circle_X;
    int circle_Y;
    int label_X;
    int label_Y;
};
struct ContourArgs{
    VP mask_contours;
    double threshold_value;
    int min_area;
    int max_area;
    double distance;
    double dist_in_loop;
};
int OPT_GRAPH = 0;
static Coordinates get_circle_crds(Coordinates &crd,
                                   const cv::Mat &input,
                                   ContourArgs &cnt_args)// find circle shape to detect the center
{
    cv::Mat thresh;
    VP contours;
    V4i hierarchy;
    int contour_index = 0;
    double _distance = cnt_args.distance;
    double _dist_in_loop = cnt_args.dist_in_loop;
    HRC::time_point begin = HRC::now();
    cv::threshold(input, thresh, cnt_args.threshold_value, 255, cv::THRESH_BINARY);
    cv::findContours(thresh, contours, hierarchy, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE); // attention: RETR_EXTERNAL - retrieves only the extreme outer contours to save time
    std::vector<cv::Moments> cmu(contours.size());

    for (size_t i = 0; i < contours.size(); i++)
    {
        cmu[i] = moments(contours[i]);
        if (cnt_args.min_area < cmu[i].m00 && cmu[i].m00 < cnt_args.max_area)
        {
            // detect distance between shapes of mask and just detected using I2 method
            _dist_in_loop = matchShapes(cnt_args.mask_contours[0], contours[i], cv::CONTOURS_MATCH_I2, 0);
            if (_dist_in_loop < _distance)
            {
                crd.circle_X = int(cmu[i].m10 / cmu[i].m00);
                crd.circle_Y = int(cmu[i].m01 / cmu[i].m00);
                _distance = _dist_in_loop;
                contour_index = i;
                std::cout << "distance = " << _distance <<" index = " << i << std::endl;
                std::cout << "    cnt x = " << crd.circle_X << ", cnt y = " << crd.circle_Y << std::endl;
            }
        }
    }
    std::vector<cv::Moments>().swap(cmu);
    HRC::time_point end = HRC::now();
    std::cout << "**** circle crds find time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    if (OPT_GRAPH){
        cv::drawContours(input, contours, contour_index, cv::Scalar(0, 255, 0), 2);
        cv::circle(input, cv::Point(crd.circle_X, crd.circle_Y), 7, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_8);

    }
    return crd;
}

static Coordinates get_latch_crds(Coordinates &crd,
                                   const cv::Mat &input,
                                   ContourArgs &cnt_args)
{
    cv::Mat thresh;
    VP latch_contours;
    V4i latch_hierarchy;
    int contour_index = 0;
    double _distance = cnt_args.distance;
    double _dist_in_loop = cnt_args.dist_in_loop;
    double angle = 80 * CV_PI / 180;
    double scale = 1.9;
    double alfa = scale * cos(angle);
    double betta = scale * sin(angle);

    HRC::time_point begin = HRC::now();

    threshold(input, thresh, cnt_args.threshold_value, 255, cv::THRESH_BINARY);
    cv::findContours(thresh, latch_contours, latch_hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Moments> mu(latch_contours.size());
    for (size_t i = 0; i < latch_contours.size(); i++)
    {
        mu[i] = moments(latch_contours[i]);
        if (cnt_args.min_area < mu[i].m00 && mu[i].m00 < cnt_args.max_area)
        {
            // detect distance between shapes of mask and just detected using I2 method
            _dist_in_loop = matchShapes(cnt_args.mask_contours[0], latch_contours[i], cv::CONTOURS_MATCH_I2, 0);
            if (_dist_in_loop < _distance)
            {
                crd.latch_X = int(mu[i].m10 / mu[i].m00);
                crd.latch_Y = int(mu[i].m01 / mu[i].m00);
                _distance = _dist_in_loop;
                contour_index = i;
                crd.label_X = crd.latch_X * alfa + crd.latch_Y * betta + (1 - alfa) * crd.circle_X - betta * crd.circle_Y;
                crd.label_Y = crd.latch_Y * alfa - crd.latch_X * betta + betta * crd.circle_X + (1 - alfa) * crd.circle_Y;
                std::cout << "distance = " << _distance << " index = " << i << std::endl;
                std::cout << "    latch x = " << crd.latch_X << ", latch y = " << crd.latch_Y << std::endl;
                std::cout << "    label x = " << crd.label_X << ", label y = " << crd.label_Y << std::endl;


            }
        }
    }

    HRC::time_point end = HRC::now();
    std::cout << "**** latch crds find time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    if (OPT_GRAPH){
        cv::drawContours(input, latch_contours, contour_index, cv::Scalar(0, 255, 0), 2);
        cv::circle(input, cv::Point(crd.latch_X, crd.latch_Y), 7, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_8);
        /*cv::putText(input, std::to_string(mu[contour_index].m00), cv::Point(crd.latch_X - 10, crd.latch_X - 10), cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 0, 0), 2);*/

        cv::circle(input, cv::Point(crd.label_X, crd.label_Y), 7, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_8);
        cv::putText(input, "[*]label", cv::Point(crd.label_X - 15, crd.label_Y - 15), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 2);
    }
    std::vector<cv::Moments>().swap(mu);
    return crd;

}

int  change_options_by_commandline(int argc, char *argv[], int *mode, int *graph, size_t *loopn)
{
    int  opt;

    while((opt =  getopt(argc, argv, "m:g:n:")) != -1)
    {
        switch(opt)
        {
            default:
                printf("_______________________________________________________________________________\n");
                printf("                                                                               \n");
                printf("  %s v.%d.%d.%d.\n", NAME, MAINVERSION, VERSION, SUBVERSION);
                printf("  -----------------------------------------------------------------------------\n");
                printf("                                                                               \n");
                printf("  Usage: %s [-m mode] [-g graph] [-n loop number]\n", argv[0]);
                printf("                                                                               \n");
                printf("  -m,  Mode (0- cane gallery, 1- cane live, 2- bottle gallery, 3- bottle live)  \n");
                printf("  -g,  Graphical output (0-no, 1-yes)                                           \n");
                printf("  -n,  Loop number                                                          \n");
                printf("_______________________________________________________________________________\n");
                printf("                                                                               \n");
                return(+1);
            case 'm':  *mode   = atol(optarg);  printf("Setting Mode Value to %d.\n",*mode);  break;
            case 'g':  *graph  = atol(optarg);  printf("Setting Graphical output to %d.\n", *graph  );  break;
            case 'n':  *loopn  = atol(optarg);  printf("Setting Loop number to %zu.\n", *loopn  );  break;

        }
    }

    if(argc<2)
    {
        printf("  Hint: Incorrect command line option (see:  %s -? )\n", argv[0]);
    }

    return(0);
}

int cane_gallery(std::string *base, size_t *loopn, int *period){
    int drops = 0;
    char fileindex[6];
    double totaltime = 0;
    double meantime = 0;
    double min_total_time = 100000;
    double max_total_time = 0;
    std::string problem_loop;
    VP mask_latch_contours;
    V4i mask_latch_hierarchy;
    VP mask_circle_contours;
    V4i mask_circle_hierarchy;
    Coordinates crd = {
            0,0,0,0,0,0
    };
    const std::string latchmaskfile = *base + "/images/mask.bmp";
    const std::string circlemaskfile = *base + "/images/round_mask.bmp";
    cv::Mat blurred;
    // find mask sample contours
    cv::Mat mask = cv::imread(latchmaskfile, cv::IMREAD_GRAYSCALE);
    cv::findContours(mask, mask_latch_contours, mask_latch_hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat mask_circle = cv::imread(circlemaskfile, cv::IMREAD_GRAYSCALE);
    cv::findContours(mask_circle, mask_circle_contours, mask_circle_hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    ContourArgs circle_args = {
            mask_circle_contours,
            45,
            150000,
            250000,
            1.2,
            10
    };
    ContourArgs latch_args = {
            mask_latch_contours,
            135,
            4000,
            5800,
            1.2,
            10
    };
    std::cout << "Performing " << *loopn << " iterations...\n" << std::flush;

    for (size_t k = 0; k < *loopn; k++)
    {
        std::cout <<"loop: " << k << std::endl;
        crd={0,0,0,0,0,0};
        HRC::time_point begin = HRC::now();

        sprintf(fileindex,"%05zu",k);
        std::string filename = *base + std::string("/images/") + fileindex + std::string(".bmp");

        cv::Mat img_gray=cv::imread(filename,cv::IMREAD_GRAYSCALE);
        cv::GaussianBlur(img_gray, blurred, cv::Size(5, 5), 0, 0, cv::BORDER_DEFAULT);

        crd = get_circle_crds(crd,blurred,circle_args);
        if ((crd.circle_X > 0) & (crd.circle_Y > 0)){
            get_latch_crds(crd,blurred,latch_args);
        }

        HRC::time_point end = HRC::now();
        std::cout << "**************** loop time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
        totaltime = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        if((totaltime>*period) | (crd.circle_X == 0) | (crd.circle_Y == 0) | (crd.label_X == 0) | (crd.label_Y == 0))
        {
            drops+=1;
            problem_loop += "loop #" + std::to_string(k) + " -> " + std::to_string(totaltime) + "ms\n";
        }
        else
        {
            meantime+=totaltime;
            min_total_time = totaltime < min_total_time? totaltime:min_total_time;
            usleep((*period-int(totaltime))*1000);
        }
        max_total_time = totaltime > max_total_time ? totaltime : max_total_time;
        if (OPT_GRAPH){
            cv::imshow("Output image with contours", blurred);
            cv::waitKey(1);

        }
    }

    meantime/=(*loopn-drops);
    std::cout << "Done!" << std::endl;
    std::cout << "Average for " << *loopn << " CPU runs: " << meantime << "ms" << std::endl;
    std::cout << "Minimum CPU loop time: " << min_total_time      << "ms" << std::endl;
    std::cout << "Maximum CPU loop time: " << max_total_time      << "ms" << std::endl;
    std::cout << "Drops: "                 << drops                       << std::endl;
    std::cout << "Problem loops are: "     << std::endl <<problem_loop.c_str() << std::endl;
    cv::destroyAllWindows();
    return drops;
}

int main(int argc, char **argv) {
    //global variables
    int optMode = 0;
    int optGraph = 0;
    int optPeriod = 40; // loop period in milliseconds for gallery mode
    size_t optLoopN = 300; // loop number

    int ret = 0;
    ret =  change_options_by_commandline(argc, argv, &optMode, &optGraph, &optLoopN);
    ASSERT(ret==0);

    OPT_GRAPH = optGraph;
    std::string argv_str(argv[0]);
    std::string base = argv_str.substr(0, argv_str.find_last_of("/"));

    switch(optMode)
    {
        default:
        case 0:
            ret = cane_gallery(&base, &optLoopN, &optPeriod);
            ASSERT(ret>=0);
            break;
        case 1:
            break;
        case 2:
            break;
        case 3:
            break;

    }


    return 0;
}




