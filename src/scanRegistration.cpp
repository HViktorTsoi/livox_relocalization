// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cmath>
#include <vector>

#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <eigen3/Eigen/Core>
// 这部分代码是对点云提取特征

typedef pcl::PointXYZI PointType;
int scanID;

// CloudFeatureFlag存储的是当前这个点属于哪一类特征
// 0: 默认
// 1: 平面特征
// 100: 前景遮挡物的边缘角点
// 150: 平面相交形成的角点
// 250: 噪点

int CloudFeatureFlag[32000];

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubLaserCloud_temp;
std::vector<sensor_msgs::PointCloud2ConstPtr> msg_window;
cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));

/**
 * 判断point_list中的点是否构成平面(因为提特征的时候是在一条线上搜索，所以实际上是判断是否构成平坦的直线)，
 * 方法是计算协方差矩阵的特征值，判断最大的特征值是否远大于其他的两个
 * @param point_list
 * @param plane_threshold
 * @return
 */
bool plane_judge(const std::vector<PointType>& point_list,const int plane_threshold)
{
  int num = point_list.size();
  float cx = 0;
  float cy = 0;
  float cz = 0;
  for (int j = 0; j < num; j++) {
      cx += point_list[j].x;
      cy += point_list[j].y;
      cz += point_list[j].z;
  }
  cx /= num;
  cy /= num;
  cz /= num;
  //mean square error
  float a11 = 0;
  float a12 = 0;
  float a13 = 0;
  float a22 = 0;
  float a23 = 0;
  float a33 = 0;
  for (int j = 0; j < num; j++) {
      float ax = point_list[j].x - cx;
      float ay = point_list[j].y - cy;
      float az = point_list[j].z - cz;

      a11 += ax * ax;
      a12 += ax * ay;
      a13 += ax * az;
      a22 += ay * ay;
      a23 += ay * az;
      a33 += az * az;
  }
  a11 /= num;
  a12 /= num;
  a13 /= num;
  a22 /= num;
  a23 /= num;
  a33 /= num;

  matA1.at<float>(0, 0) = a11;
  matA1.at<float>(0, 1) = a12;
  matA1.at<float>(0, 2) = a13;
  matA1.at<float>(1, 0) = a12;
  matA1.at<float>(1, 1) = a22;
  matA1.at<float>(1, 2) = a23;
  matA1.at<float>(2, 0) = a13;
  matA1.at<float>(2, 1) = a23;
  matA1.at<float>(2, 2) = a33;

  cv::eigen(matA1, matD1, matV1);
  if (matD1.at<float>(0, 0) > plane_threshold * matD1.at<float>(0, 1)) {
    return true;
  }
  else{
    return false;
  }
}

void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
{
  pcl::PointCloud<PointType> laserCloudIn;
  pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);

  int cloudSize = laserCloudIn.points.size();

  if(cloudSize > 32000) cloudSize = 32000;
  
  int count = cloudSize;
  PointType point;
  pcl::PointCloud<PointType> Allpoints;

  for (int i = 0; i < cloudSize; i++) {

    point.x = laserCloudIn.points[i].x;
    point.y = laserCloudIn.points[i].y;
    point.z = laserCloudIn.points[i].z;
    // 此处是针对mid40model的旋转方式，通过在yOz方向的横滚角，判断是第几根线
    double theta = std::atan2(laserCloudIn.points[i].y,laserCloudIn.points[i].z) / M_PI * 180 + 180;

    // 这个scanID没有用到
    scanID = std::floor(theta / 9);
    float dis = point.x * point.x
                + point.y * point.y
                + point.z * point.z;

    double dis2 = laserCloudIn.points[i].z * laserCloudIn.points[i].z + laserCloudIn.points[i].y * laserCloudIn.points[i].y;
    double theta2 = std::asin(sqrt(dis2/dis)) / M_PI * 180;

    point.intensity = scanID+(laserCloudIn.points[i].intensity/10000);
    //point.intensity = scanID+(double(i)/cloudSize);

    // 抛弃测距为inf的点
    if (!pcl_isfinite(point.x) ||
        !pcl_isfinite(point.y) ||
        !pcl_isfinite(point.z)) {
      continue;
    }

    Allpoints.push_back(point);
  }

  pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
  *laserCloud += Allpoints;
  cloudSize = laserCloud->size();

  for (int i = 0; i < cloudSize; i++) {
    CloudFeatureFlag[i] = 0;
  }
    
  pcl::PointCloud<PointType> cornerPointsSharp;

  pcl::PointCloud<PointType> surfPointsFlat;

  pcl::PointCloud<PointType> laserCloudFull;

  int debugnum1 = 0;
  int debugnum2 = 0;
  int debugnum3 = 0;
  int debugnum4 = 0;
  int debugnum5 = 0;

  int count_num = 1;
  bool left_surf_flag = false;
  bool right_surf_flag = false;
  Eigen::Vector3d surf_vector_current(0,0,0);
  Eigen::Vector3d surf_vector_last(0,0,0);
  int last_surf_position = 0;
  double depth_threshold = 0.1;
  //********************************************************************************************************************************************
  for (int i = 5; i < cloudSize - 5; i += count_num ) {
    float depth = sqrt(laserCloud->points[i].x * laserCloud->points[i].x +
                       laserCloud->points[i].y * laserCloud->points[i].y +
                       laserCloud->points[i].z * laserCloud->points[i].z);

    // if(depth < 2) depth_threshold = 0.05;
    // if(depth > 30) depth_threshold = 0.1;
    //left curvature
    // 求当前点左侧邻域的曲率
    float ldiffX = 
                laserCloud->points[i - 4].x + laserCloud->points[i - 3].x
                - 4 * laserCloud->points[i - 2].x
                + laserCloud->points[i - 1].x + laserCloud->points[i].x;

    float ldiffY = 
                laserCloud->points[i - 4].y + laserCloud->points[i - 3].y
                - 4 * laserCloud->points[i - 2].y
                + laserCloud->points[i - 1].y + laserCloud->points[i].y;

    float ldiffZ = 
                laserCloud->points[i - 4].z + laserCloud->points[i - 3].z
                - 4 * laserCloud->points[i - 2].z
                + laserCloud->points[i - 1].z + laserCloud->points[i].z;

    float left_curvature = ldiffX * ldiffX + ldiffY * ldiffY + ldiffZ * ldiffZ;

    // 如果左侧曲率较小，则做一次标记
    if(left_curvature < 0.01){

      std::vector<PointType> left_list;

      for(int j = -4; j < 0; j++){
        left_list.push_back(laserCloud->points[i+j]);
      }

      // 如果左侧曲率足够小，则顺便将左侧邻域最中间的那个点标记为平面点
      if( left_curvature < 0.001) CloudFeatureFlag[i-2] = 1; //surf point flag  && plane_judge(left_list,1000)

      left_surf_flag = true;
    }
    else{
      left_surf_flag = false;
    }

    // 计算右侧邻域的曲率
    //right curvature
    float rdiffX = 
                laserCloud->points[i + 4].x + laserCloud->points[i + 3].x
                - 4 * laserCloud->points[i + 2].x
                + laserCloud->points[i + 1].x + laserCloud->points[i].x;

    float rdiffY = 
                laserCloud->points[i + 4].y + laserCloud->points[i + 3].y
                - 4 * laserCloud->points[i + 2].y
                + laserCloud->points[i + 1].y + laserCloud->points[i].y;

    float rdiffZ = 
                laserCloud->points[i + 4].z + laserCloud->points[i + 3].z
                - 4 * laserCloud->points[i + 2].z
                + laserCloud->points[i + 1].z + laserCloud->points[i].z;

    float right_curvature = rdiffX * rdiffX + rdiffY * rdiffY + rdiffZ * rdiffZ;

    // 同理，如果右侧曲率较小，则做一次标记
    if(right_curvature < 0.01){
      std::vector<PointType> right_list;

      for(int j = 1; j < 5; j++){
        right_list.push_back(laserCloud->points[i+j]);
      }
        if(right_curvature < 0.001 ) CloudFeatureFlag[i+2] = 1; //surf point flag  && plane_judge(right_list,1000)


      count_num = 4;
      right_surf_flag = true;
    }
    else{
      count_num = 1;
      right_surf_flag = false;
    }

    // 如果左右邻域的都较小，判断是不是由两个平面构成的交线(在laser线上就是两条直线构成的交点)
    //surf-surf corner feature
    if(left_surf_flag && right_surf_flag){
     debugnum4 ++;

     // 计算当前点到左右邻域点方向向量的平均
     Eigen::Vector3d norm_left(0,0,0);
     Eigen::Vector3d norm_right(0,0,0);
     for(int k = 1;k<5;k++){
         Eigen::Vector3d tmp = Eigen::Vector3d(laserCloud->points[i-k].x-laserCloud->points[i].x,
                            laserCloud->points[i-k].y-laserCloud->points[i].y,
                            laserCloud->points[i-k].z-laserCloud->points[i].z);
        tmp.normalize();
        norm_left += (k/10.0)* tmp;
     }
     for(int k = 1;k<5;k++){
         Eigen::Vector3d tmp = Eigen::Vector3d(laserCloud->points[i+k].x-laserCloud->points[i].x,
                            laserCloud->points[i+k].y-laserCloud->points[i].y,
                            laserCloud->points[i+k].z-laserCloud->points[i].z);
        tmp.normalize();
        norm_right += (k/10.0)* tmp;
     }

      // 计算左右平均方向向量的夹角
      //calculate the angle between this group and the previous group
      double cc = fabs( norm_left.dot(norm_right) / (norm_left.norm()*norm_right.norm()) );
      //calculate the maximum distance, the distance cannot be too small
      Eigen::Vector3d last_tmp = Eigen::Vector3d(laserCloud->points[i-4].x-laserCloud->points[i].x,
                                                 laserCloud->points[i-4].y-laserCloud->points[i].y,
                                                 laserCloud->points[i-4].z-laserCloud->points[i].z);
      Eigen::Vector3d current_tmp = Eigen::Vector3d(laserCloud->points[i+4].x-laserCloud->points[i].x,
                                                    laserCloud->points[i+4].y-laserCloud->points[i].y,
                                                    laserCloud->points[i+4].z-laserCloud->points[i].z);
      double last_dis = last_tmp.norm();
      double current_dis = current_tmp.norm();

      // 夹角足够接近90度，并且左右邻域足够长，认为是两平面夹角的边缘点
      if(cc < 0.5 && last_dis > 0.05 && current_dis > 0.05 ){ //
        debugnum5 ++;
        CloudFeatureFlag[i] = 150;
      }
    }
  }
  for(int i = 5; i < cloudSize - 5; i ++){
    float diff_left[2];
    float diff_right[2];
    float depth = sqrt(laserCloud->points[i].x * laserCloud->points[i].x +
                        laserCloud->points[i].y * laserCloud->points[i].y +
                        laserCloud->points[i].z * laserCloud->points[i].z);

    // 计算左右邻域的点到当前点的梯度
    for(int count = 1; count < 3; count++ ){
      float diffX1 = laserCloud->points[i + count].x - laserCloud->points[i].x;
      float diffY1 = laserCloud->points[i + count].y - laserCloud->points[i].y;
      float diffZ1 = laserCloud->points[i + count].z - laserCloud->points[i].z;
      diff_right[count - 1] = sqrt(diffX1 * diffX1 + diffY1 * diffY1 + diffZ1 * diffZ1);

      float diffX2 = laserCloud->points[i - count].x - laserCloud->points[i].x;
      float diffY2 = laserCloud->points[i - count].y - laserCloud->points[i].y;
      float diffZ2 = laserCloud->points[i - count].z - laserCloud->points[i].z;
      diff_left[count - 1] = sqrt(diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2);
    }

    float depth_right = sqrt(laserCloud->points[i + 1].x * laserCloud->points[i + 1].x +
                    laserCloud->points[i + 1].y * laserCloud->points[i + 1].y +
                    laserCloud->points[i + 1].z * laserCloud->points[i + 1].z);
    float depth_left = sqrt(laserCloud->points[i - 1].x * laserCloud->points[i - 1].x +
                    laserCloud->points[i - 1].y * laserCloud->points[i - 1].y +
                    laserCloud->points[i - 1].z * laserCloud->points[i - 1].z);

    // 左右梯度都非常大的点被认为是噪点
     //outliers
    if( (diff_right[0] > 0.1*depth && diff_left[0] > 0.1*depth) ){
      debugnum1 ++;  
      CloudFeatureFlag[i] = 250;
      continue;
    }

    // 左右邻域距离非常远，即不属于平面交点的边缘点，比如前景遮挡的情况，这里提取的就是前景的边缘点
    // 分为左右两种情况(在laser线上的左右，实际上由于mid40的旋转这里可以检测到各个方向的边缘点，但是适配到其他的雷达上效果可能很差)
    //break points
    if(fabs(diff_right[0] - diff_left[0])>0.1){
      if(diff_right[0] > diff_left[0]){

        Eigen::Vector3d surf_vector = Eigen::Vector3d(laserCloud->points[i-4].x-laserCloud->points[i].x,
                                                  laserCloud->points[i-4].y-laserCloud->points[i].y,
                                                  laserCloud->points[i-4].z-laserCloud->points[i].z);
        Eigen::Vector3d lidar_vector = Eigen::Vector3d(laserCloud->points[i].x,
                                                      laserCloud->points[i].y,
                                                      laserCloud->points[i].z);
        double left_surf_dis = surf_vector.norm();
        //calculate the angle between the laser direction and the surface
        double cc = fabs( surf_vector.dot(lidar_vector) / (surf_vector.norm()*lidar_vector.norm()) );

        // 判断左侧的邻域是否构成平面，通过协方差的特征值大小
        std::vector<PointType> left_list;
        double min_dis = 10000;
        double max_dis = 0;
        for(int j = 0; j < 4; j++){   //TODO: change the plane window size and add thin rod support
          left_list.push_back(laserCloud->points[i-j]);
          Eigen::Vector3d temp_vector = Eigen::Vector3d(laserCloud->points[i-j].x-laserCloud->points[i-j-1].x,
                                                  laserCloud->points[i-j].y-laserCloud->points[i-j-1].y,
                                                  laserCloud->points[i-j].z-laserCloud->points[i-j-1].z);

          if(j == 3) break;
          double temp_dis = temp_vector.norm();
          if(temp_dis < min_dis) min_dis = temp_dis;
          if(temp_dis > max_dis) max_dis = temp_dis;
        }
        bool left_is_plane = plane_judge(left_list,100);

        // 仅当左侧为平面，并且最近和最远距离差较小，并且左侧邻域与当前点距离较小，且夹角足够接近90度的点才被认为是特征角点
        if(left_is_plane && (max_dis < 2*min_dis) && left_surf_dis < 0.05 * depth  && cc < 0.8){//
          if(depth_right > depth_left){
            CloudFeatureFlag[i] = 100;
          }
          else{
            if(depth_right == 0) CloudFeatureFlag[i] = 100;
          }
        }
      }
      else{

        // 右侧的情况完全相同，这里考虑可以重构下代码
        Eigen::Vector3d surf_vector = Eigen::Vector3d(laserCloud->points[i+4].x-laserCloud->points[i].x,
                                                      laserCloud->points[i+4].y-laserCloud->points[i].y,
                                                      laserCloud->points[i+4].z-laserCloud->points[i].z);
        Eigen::Vector3d lidar_vector = Eigen::Vector3d(laserCloud->points[i].x,
                                                      laserCloud->points[i].y,
                                                      laserCloud->points[i].z);
        double right_surf_dis = surf_vector.norm();
        //calculate the angle between the laser direction and the surface
        double cc = fabs( surf_vector.dot(lidar_vector) / (surf_vector.norm()*lidar_vector.norm()) );

        std::vector<PointType> right_list;
        double min_dis = 10000;
        double max_dis = 0;
        for(int j = 0; j < 4; j++){ //TODO: change the plane window size and add thin rod support
          right_list.push_back(laserCloud->points[i-j]);
          Eigen::Vector3d temp_vector = Eigen::Vector3d(laserCloud->points[i+j].x-laserCloud->points[i+j+1].x,
                                                  laserCloud->points[i+j].y-laserCloud->points[i+j+1].y,
                                                  laserCloud->points[i+j].z-laserCloud->points[i+j+1].z);

          if(j == 3) break;
          double temp_dis = temp_vector.norm();
          if(temp_dis < min_dis) min_dis = temp_dis;
          if(temp_dis > max_dis) max_dis = temp_dis;
        }
        bool right_is_plane = plane_judge(right_list,100);

        if(right_is_plane && (max_dis < 2*min_dis) && right_surf_dis < 0.05 * depth && cc < 0.8){ // 

          if(depth_right < depth_left){
            CloudFeatureFlag[i] = 100;
          }
          else{
            if(depth_left == 0) CloudFeatureFlag[i] = 100;       
          }
        }
      }
    }

    // 重复验证，保证边缘点的左侧和右侧邻域的夹角接近90度
    // (这里有问题，如果按照这种逻辑，那么前景和背景都是平面的情况下，这样的边缘就会被排除掉)，
    // 可以这样考虑：如果断点处左右都是平面，如果这断点在背景中，那么说明这个断点是由于前景遮挡造成的“阴影边缘”，当雷达移动时阴影边缘点当然会移动
    // 但是这样的移动是由于视差引起的，与雷达的移动无关，无法用作特征，因此这里考虑将其排除掉
    // break point select
    if(CloudFeatureFlag[i] == 100){
      debugnum2++;
      std::vector<Eigen::Vector3d> front_norms;
      Eigen::Vector3d norm_front(0,0,0);
      Eigen::Vector3d norm_back(0,0,0);
      for(int k = 1;k<4;k++){
          Eigen::Vector3d tmp = Eigen::Vector3d(laserCloud->points[i-k].x-laserCloud->points[i].x,
                            laserCloud->points[i-k].y-laserCloud->points[i].y,
                            laserCloud->points[i-k].z-laserCloud->points[i].z);
        tmp.normalize();
        front_norms.push_back(tmp);
        norm_front += (k/6.0)* tmp;
      }
      std::vector<Eigen::Vector3d> back_norms;
      for(int k = 1;k<4;k++){
          Eigen::Vector3d tmp = Eigen::Vector3d(laserCloud->points[i+k].x-laserCloud->points[i].x,
                            laserCloud->points[i+k].y-laserCloud->points[i].y,
                            laserCloud->points[i+k].z-laserCloud->points[i].z);
        tmp.normalize();
        back_norms.push_back(tmp);
        norm_back += (k/6.0)* tmp;
      }
      double cc = fabs( norm_front.dot(norm_back) / (norm_front.norm()*norm_back.norm()) );
        if(cc < 0.8){
        debugnum3++;
      }else{
        CloudFeatureFlag[i] = 0;
      }

      continue;

    }
  }

  // 按照上述特征提取过程对点云进行segmentation之后，分别保存边缘和平面两类特征点
  //push_back feature
  for(int i = 0; i < cloudSize; i++){
    //laserCloud->points[i].intensity = double(CloudFeatureFlag[i]) / 10000;
    float dis = laserCloud->points[i].x * laserCloud->points[i].x
                + laserCloud->points[i].y * laserCloud->points[i].y
                + laserCloud->points[i].z * laserCloud->points[i].z;
    float dis2 = laserCloud->points[i].y * laserCloud->points[i].y + laserCloud->points[i].z * laserCloud->points[i].z;
    float theta2 = std::asin(sqrt(dis2/dis)) / M_PI * 180;
    //std::cout<<"DEBUG theta "<<theta2<<std::endl;
    // if(theta2 > 34.2 || theta2 < 1){
    //    continue;
    // }
    //if(dis > 30*30) continue;

    if(CloudFeatureFlag[i] == 1){
      surfPointsFlat.push_back(laserCloud->points[i]);
      continue;
    }

    if(CloudFeatureFlag[i] == 100 || CloudFeatureFlag[i] == 150){
        cornerPointsSharp.push_back(laserCloud->points[i]);
    } 
  }


  // 发布相关的topic
  std::cout<<"ALL point: "<<cloudSize<<" outliers: "<< debugnum1 << std::endl
            <<" break points: "<< debugnum2<<" break feature: "<< debugnum3 << std::endl
            <<" normal points: "<< debugnum4<<" surf-surf feature: " << debugnum5 << std::endl;

  sensor_msgs::PointCloud2 laserCloudOutMsg;
  pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
  laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
  laserCloudOutMsg.header.frame_id = "/livox";
  pubLaserCloud.publish(laserCloudOutMsg);

  sensor_msgs::PointCloud2 cornerPointsSharpMsg;
  pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
  cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsSharpMsg.header.frame_id = "/livox";
  pubCornerPointsSharp.publish(cornerPointsSharpMsg);

  sensor_msgs::PointCloud2 surfPointsFlat2;
  pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
  surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsFlat2.header.frame_id = "/livox";
  pubSurfPointsFlat.publish(surfPointsFlat2);

}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "scanRegistration");
  ros::NodeHandle nh;

  ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>
                                  ("/livox/lidar", 100, laserCloudHandler);
  pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>
                                 ("/livox_cloud", 20);

  pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>
                                        ("/laser_cloud_sharp", 20);


  pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>
                                       ("/laser_cloud_flat", 20);


  ros::spin();

  return 0;
}
