// Offline adapter around unmodified official Nav2 AMCL libraries (LGPL-2.1+).
// This adapter controls IO/update scheduling; it is not the full ROS lifecycle node.
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include "nav2_amcl/map/map.hpp"
#include "nav2_amcl/pf/pf.hpp"
#include "nav2_amcl/motion_model/omni_motion_model.hpp"
#include "nav2_amcl/sensors/laser/laser.hpp"

static pf_vector_t random_pose(void * ptr) {
  auto * m=static_cast<map_t *>(ptr); pf_vector_t p{};
  for (int k=0;k<1000000;++k) {
    int x=static_cast<int>(drand48()*m->size_x), y=static_cast<int>(drand48()*m->size_y);
    if(m->cells[MAP_INDEX(m,x,y)].occ_state==-1) {
      p.v[0]=MAP_WXGX(m,x); p.v[1]=MAP_WYGY(m,y); p.v[2]=(drand48()*2-1)*M_PI; return p;
    }
  }
  throw std::runtime_error("No sampled free map cell");
}
static double wrap(double a) {return std::atan2(std::sin(a),std::cos(a));}
int main(int argc,char **argv) {
  if(argc<7) {
    std::cerr<<"usage: offline_amcl MAP FRAMES OUT INIT_X INIT_Y INIT_YAW [SEED=0] [MODEL=likelihood_field] [MIN=1000] [MAX=10000] [ALPHA=0.2] [SIGMA_HIT=0.2] [MAX_BEAMS=100] [INIT_STD_XY=0.2] [INIT_STD_YAW_DEG=10]\n";
    return argc==2 && std::string(argv[1])=="--help"?0:2;
  }
  try {
    const long seed=argc>7?std::stol(argv[7]):0; srand48(seed);
    const std::string model=argc>8?argv[8]:"likelihood_field";
    const int nmin=argc>9?std::stoi(argv[9]):1000,nmax=argc>10?std::stoi(argv[10]):10000;
    const double alpha=argc>11?std::stod(argv[11]):0.2,sigma=argc>12?std::stod(argv[12]):0.2;
    const int beams=argc>13?std::stoi(argv[13]):100;
    const double std_xy=argc>14?std::stod(argv[14]):0.2;
    const double std_yaw=(argc>15?std::stod(argv[15]):10.0)*M_PI/180.0;
    if(nmin<1 || nmax<nmin || alpha<0 || sigma<=0 || beams<2 || std_xy<=0 || std_yaw<=0) throw std::runtime_error("Invalid parameters");
    std::ifstream mi(argv[1]); if(!mi) throw std::runtime_error("Cannot read map");
    auto * map=map_alloc(); double ox,oy;
    mi>>map->size_x>>map->size_y>>map->scale>>ox>>oy;
    if(!mi || map->size_x<1 || map->size_y<1 || map->scale<=0) throw std::runtime_error("Invalid map header");
    // Match Nav2 AmclNode::convertMap: ROS lower-left corner -> AMCL center.
    map->origin_x=ox+(map->size_x/2)*map->scale;
    map->origin_y=oy+(map->size_y/2)*map->scale;
    map->cells=static_cast<map_cell_t *>(calloc(map->size_x*map->size_y,sizeof(map_cell_t)));
    for(int k=0;k<map->size_x*map->size_y;++k) {
      int c; mi>>c; if(!mi || c< -1 || c>1) throw std::runtime_error("Invalid/missing map state"); map->cells[k].occ_state=c;
    }
    auto * pf=pf_alloc(nmin,nmax,0.0,0.0,random_pose);
    // pf_alloc seeds with wall time. Override before initialization for reproducibility.
    srand48(seed);
    pf->pop_err=0.05; pf->pop_z=0.99;
    pf_vector_t initial{}; initial.v[0]=std::stod(argv[4]); initial.v[1]=std::stod(argv[5]); initial.v[2]=std::stod(argv[6]);
    pf_matrix_t covariance{}; covariance.m[0][0]=std_xy*std_xy; covariance.m[1][1]=std_xy*std_xy; covariance.m[2][2]=std_yaw*std_yaw;
    pf_init(pf,initial,covariance);
    nav2_amcl::OmniMotionModel motion; motion.initialize(alpha,alpha,alpha,alpha,alpha);
    std::unique_ptr<nav2_amcl::Laser> laser;
    if(model=="likelihood_field") laser=std::make_unique<nav2_amcl::LikelihoodFieldModel>(0.5,0.5,sigma,2.0,beams,map);
    else if(model=="likelihood_field_prob") laser=std::make_unique<nav2_amcl::LikelihoodFieldModelProb>(0.5,0.5,sigma,2.0,true,0.5,0.3,0.9,beams,map);
    else if(model=="likelihood_field_prob_noskip") laser=std::make_unique<nav2_amcl::LikelihoodFieldModelProb>(0.5,0.5,sigma,2.0,false,0.5,0.3,0.9,beams,map);
    else throw std::runtime_error("Unknown model");
    std::ifstream frames(argv[2]); std::ofstream out(argv[3]); if(!frames || !out) throw std::runtime_error("Cannot open frames/output");
    pf_vector_t extrinsic{}; double range_max;
    frames>>extrinsic.v[0]>>extrinsic.v[1]>>extrinsic.v[2]>>range_max;
    if(!frames || range_max<=0) throw std::runtime_error("Invalid laser header");
    laser->SetLaserPose(extrinsic);
    out<<"timestamp,x,y,yaw,n_particles,cluster_weight,cov_xx,cov_yy,cov_yawyaw\n"<<std::setprecision(17);
    pf_vector_t previous{}; bool first=true; double time, last_time=-INFINITY;
    while(frames>>time) {
      pf_vector_t odom{}; int n;
      frames>>odom.v[0]>>odom.v[1]>>odom.v[2]>>n;
      if(!frames || n<2 || time<=last_time) throw std::runtime_error("Invalid frame header/time");
      last_time=time;
      nav2_amcl::LaserData data; data.laser=laser.get(); data.range_count=n; data.range_max=range_max; data.ranges=new double[n][2];
      for(int i=0;i<n;++i) {
        frames>>data.ranges[i][0]>>data.ranges[i][1];
        if(!frames || !std::isfinite(data.ranges[i][0]) || !std::isfinite(data.ranges[i][1]) || data.ranges[i][0]<0) throw std::runtime_error("Invalid range/bearing");
      }
      if(!first) {pf_vector_t delta{}; for(int j=0;j<3;++j) delta.v[j]=odom.v[j]-previous.v[j]; delta.v[2]=wrap(delta.v[2]); motion.odometryUpdate(pf,odom,delta);}
      previous=odom; first=false;
      laser->sensorUpdate(pf,&data); pf_update_resample(pf,map);
      pf_vector_t pose{}; pf_matrix_t cov{}; double max_weight=-1;
      auto *set=&pf->sets[pf->current_set];
      for(int c=0;c<set->cluster_count;++c) {double w; pf_vector_t p; pf_matrix_t cv; if(pf_get_cluster_stats(pf,c,&w,&p,&cv) && w>max_weight) {max_weight=w;pose=p;cov=cv;}}
      if(max_weight<0) throw std::runtime_error("No particle cluster");
      out<<time<<','<<pose.v[0]<<','<<pose.v[1]<<','<<pose.v[2]<<','<<set->sample_count<<','<<max_weight<<','<<cov.m[0][0]<<','<<cov.m[1][1]<<','<<cov.m[2][2]<<std::endl;
    }
    if(!frames.eof()) throw std::runtime_error("Malformed frames");
    laser.reset(); pf_free(pf); map_free(map);
  }catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
