#include <iostream>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>


#include <string>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>
 using namespace std;
 
const char* path = "/home/jhua/hba_data/avia1/pcd_ds005_er01/";
void GetFileNames(string path,vector<string>& filenames)
{
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str()))){
        std::cout<<"Folder doesn't Exist!"<<std::endl;
        return;
    }
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
            filenames.push_back(path + "/" + ptr->d_name);
            //std::cout<<path + "/" + ptr->d_name<<endl;
    }
    }
    closedir(pDir);
}



int main(int argc, char** argv) {
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloudadd(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
	//pcl::io::loadPCDFile("./test1.pcd", *cloud);


	DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir("/home/jhua/hba_data/avia1/pcd_ds005_er01/"))){
        std::cout<<"Folder doesn't Exist!"<<std::endl;
        return -1;
    }
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
            //filenames.push_back(path + "/" + ptr->d_name);
			pcl::io::loadPCDFile("/home/jhua/hba_data/avia1/pcd_ds005_er01/" + string(ptr->d_name), *cloud);
			*cloudadd = *cloudadd + *cloud;
			cloud->clear();
            printf("Processing %s\n", ptr->d_name);
            //std::cout<<path + "/" + ptr->d_name<<endl;
    }
    }
    closedir(pDir);

	pcl::io::savePCDFileBinary("ds005_er01.pcd", *cloudadd);
	return 0;
 
}


