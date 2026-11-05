#include <cstdlib>
#include <iostream>
#include "nav2_amcl/map/map.hpp"
#include "nav2_amcl/pf/pf.hpp"
#include "nav2_amcl/sensors/laser/laser.hpp"
pf_vector_t rp(void*) {return pf_vector_t{};}
void run(int n,bool skip){
 map_t* m=map_alloc();m->size_x=m->size_y=100;m->scale=.1;m->origin_x=m->origin_y=0;m->cells=(map_cell_t*)calloc(10000,sizeof(map_cell_t));
 for(int k=0;k<10000;k++)m->cells[k].occ_state=-1;
 m->cells[0].occ_state=1;
 pf_t* p=pf_alloc(2,2,0,0,rp);pf_vector_t pose{};pf_matrix_t cov{};cov.m[0][0]=cov.m[1][1]=cov.m[2][2]=.001;pf_init(p,pose,cov);
 auto* set=&p->sets[p->current_set];set->sample_count=2;set->samples[0].pose=set->samples[1].pose=pose;set->samples[0].weight=.75;set->samples[1].weight=.25;set->converged=1;
 {nav2_amcl::LikelihoodFieldModelProb l(.5,.5,100,2,skip,.5,.3,.9,100,m);l.SetLaserPose(pose);nav2_amcl::LaserData d;d.laser=&l;d.range_count=n;d.range_max=30;d.ranges=new double[n][2];for(int i=0;i<n;i++){d.ranges[i][0]=1;d.ranges[i][1]=0;}l.sensorUpdate(p,&d);std::cout<<"n="<<n<<" skip="<<skip<<" weight0="<<set->samples[0].weight<<" weight1="<<set->samples[1].weight<<"\n";}
 pf_free(p);map_free(m);
}
int main(){run(100,true);run(811,true);run(811,false);}
