// Read-only event instrumentation of recorded Karto scans. No truth input.
#include <karto_sdk/Mapper.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

struct Events : karto::MapperLoopClosureListener {
  karto::Mapper *mapper = nullptr;
  karto::LocalizedRangeScan *scan = nullptr;
  std::ofstream events;
  int current = -1;
  double time = -1;
  int checked = -1;
  bool captured = false;
  explicit Events(const std::string &path):events(path) {
    events << std::setprecision(17)
      << "scan\ttime\tradius\tevent\tother\tseparation_m\tchain_first\tchain_last\tchain_size\tx\ty\tyaw\tdetail\n";
  }
  void line(double radius,const std::string &kind,int other=-1,double separation=-1,
            int first=-1,int last=-1,int size=0,double x=0,double y=0,double yaw=0,std::string detail="") {
    std::replace(detail.begin(),detail.end(),'\t',' ');
    std::replace(detail.begin(),detail.end(),'\n',' ');
    events << current << '\t' << time << '\t' << radius << '\t' << kind << '\t'
      << other << '\t' << separation << '\t' << first << '\t' << last << '\t'
      << size << '\t' << x << '\t' << y << '\t' << yaw << '\t' << detail << '\n';
  }
  // Mirrors FindPossibleLoopClosure's distance, linked-scan and chain-length
  // tests using its public graph traversal and scan manager. The actual matcher
  // calls are still made by the installed library and logged below.
  void capture() {
    if (captured || !mapper || !scan || !mapper->GetGraph()) return;
    captured = true;
    auto graph=mapper->GetGraph();
    auto pose=scan->GetReferencePose(true);
    line(0,"POSE_BEFORE_MATCH",-1,-1,-1,-1,0,pose.GetX(),pose.GetY(),pose.GetHeading());
    auto mgr=mapper->GetMapperSensorManager();
    auto name=scan->GetSensorName();
    // One sensor with contiguous accepted state IDs in these recordings.
    const int n=scan->GetStateId()+1;
    for (double radius:{1.5,3.0}) {
      auto near=graph->FindNearLinkedScans(scan,radius);
      std::unordered_set<karto::LocalizedRangeScan*> linked(near.begin(),near.end());
      std::vector<std::pair<int,double>> chain;
      auto emit=[&]() {
        if (chain.size()>=10) {
          double low=1e9,high=0;
          for (auto q:chain){low=std::min(low,q.second);high=std::max(high,q.second);}
          line(radius,"MATCH_CHAIN",-1,low,chain.front().first,chain.back().first,
               static_cast<int>(chain.size()),pose.GetX(),pose.GetY(),pose.GetHeading(),
               "max_separation_m="+std::to_string(high));
        }
        chain.clear();
      };
      for (int k=0;k<n;k++) {
        auto prior=mgr->GetScan(name,k);
        if (!prior) continue;
        auto q=prior->GetReferencePose(true);
        double d=std::hypot(q.GetX()-pose.GetX(),q.GetY()-pose.GetY());
        bool inside=d<radius+1e-8;
        bool is_linked=linked.count(prior)>0;
        if (k<scan->GetStateId() && d<=3.0+1e-8) {
          line(radius,inside?(is_linked?"LINKED_EXCLUSION":"DISTANCE_ADMITTED"):
               "DISTANCE_EXCLUDED",k,d);
        }
        if (inside) {
          if (is_linked) chain.clear();
          else chain.emplace_back(k,d);
        } else emit();
      }
      // The installed function returns the terminal chain even when short.
      if (!chain.empty()) {
        double low=1e9,high=0;
        for (auto q:chain){low=std::min(low,q.second);high=std::max(high,q.second);}
        line(radius,"MATCH_CHAIN_TERMINAL",-1,low,chain.front().first,
             chain.back().first,static_cast<int>(chain.size()),
             pose.GetX(),pose.GetY(),pose.GetHeading(),
             "max_separation_m="+std::to_string(high));
      }
    }
  }
  void LoopClosureCheck(const std::string &s) override {
    capture();
    if (s.find("COARSE RESPONSE:")==0) ++checked;
    line(0,"MATCHER_CHECK",checked,-1,-1,-1,0,0,0,0,s);
    std::cout << "CHECK," << current << ',' << time << ',' << s << '\n';
  }
  void BeginLoopClosure(const std::string &s) override {
    line(0,"ACCEPT_LOOP",checked,-1,-1,-1,0,0,0,0,s);
    std::cout << "ACCEPT," << current << ',' << time << ',' << s << '\n';
  }
};

int main(int argc, char **argv) {
  if (argc != 5) {std::cerr << "usage: replay_events session.data radius output_edges.csv events.tsv\n";return 2;}
  karto::Dataset data; data.LoadFromFile(argv[1]);
  if (data.GetLasers().size()!=1) {std::cerr << "expected one lidar\n";return 3;}
  auto laser=dynamic_cast<karto::LaserRangeFinder*>(data.GetLasers()[0]);
  if (!laser) return 4;
  karto::SensorManager::GetInstance()->RegisterSensor(laser,true);
  std::cout << std::setprecision(17) << "SENSOR," << laser->GetName().ToString() << ','
            << laser->GetMinimumRange() << ',' << laser->GetMaximumRange() << ','
            << laser->GetRangeThreshold() << ',' << laser->GetMinimumAngle() << ','
            << laser->GetMaximumAngle() << ',' << laser->GetAngularResolution() << '\n';
  karto::Mapper mapper;
  mapper.setParamUseScanMatching(true);mapper.setParamUseScanBarycenter(true);
  mapper.setParamMinimumTimeInterval(.1);mapper.setParamMinimumTravelDistance(.1);
  mapper.setParamMinimumTravelHeading(.1);mapper.setParamScanBufferSize(10);
  mapper.setParamScanBufferMaximumScanDistance(10);
  mapper.setParamLinkMatchMinimumResponseFine(.1);mapper.setParamLinkScanMaximumDistance(1.5);
  mapper.setParamLoopSearchMaximumDistance(std::stod(argv[2]));mapper.setParamDoLoopClosing(true);
  mapper.setParamLoopMatchMinimumChainSize(10);mapper.setParamLoopMatchMaximumVarianceCoarse(3.0);
  mapper.setParamLoopMatchMinimumResponseCoarse(.35);mapper.setParamLoopMatchMinimumResponseFine(.45);
  mapper.setParamCorrelationSearchSpaceDimension(.5);mapper.setParamCorrelationSearchSpaceResolution(.01);
  mapper.setParamCorrelationSearchSpaceSmearDeviation(.1);
  mapper.setParamLoopSearchSpaceDimension(8.0);mapper.setParamLoopSearchSpaceResolution(.05);
  mapper.setParamLoopSearchSpaceSmearDeviation(.03);
  mapper.setParamDistanceVariancePenalty(.5);mapper.setParamAngleVariancePenalty(1.0);
  mapper.setParamFineSearchAngleOffset(.00349);mapper.setParamCoarseSearchAngleOffset(.349);
  mapper.setParamCoarseAngleResolution(.0349);mapper.setParamMinimumAnglePenalty(.9);
  mapper.setParamMinimumDistancePenalty(.5);mapper.setParamUseResponseExpansion(true);
  mapper.setParamMinPassThrough(2);mapper.setParamOccupancyThreshold(.1);
  Events listener(argv[4]);listener.mapper=&mapper;mapper.AddListener(&listener);
  int processed=0;
  for (auto const &[id,obj] : data.GetData()) {
    auto scan=dynamic_cast<karto::LocalizedRangeScan*>(obj);
    if (!scan) continue;
    listener.current=id;listener.time=scan->GetTime();
    listener.scan=scan;listener.captured=false;listener.checked=-1;
    // Process() predicts from the recorded wheel pose. The saved corrected pose is
    // deliberately discarded; it contains information from the original result.
    scan->SetCorrectedPose(scan->GetOdometricPose());
    const auto old_edges=mapper.GetGraph()?mapper.GetGraph()->GetEdges().size():0;
    bool ok=mapper.Process(scan);
    if (ok && !listener.captured) listener.capture();
    if (ok) ++processed;
    if (ok) {
      auto p=scan->GetCorrectedPose();std::cout << "POSE," << id << ',' << scan->GetTime()
        << ',' << p.GetX() << ',' << p.GetY() << ',' << p.GetHeading() << '\n';
      listener.line(0,"POSE_AFTER",-1,-1,-1,-1,0,p.GetX(),p.GetY(),p.GetHeading());
      const auto &links=mapper.GetGraph()->GetEdges();
      for (size_t k=old_edges;k<links.size();++k) {
        auto edge=links[k];auto label=dynamic_cast<karto::LinkInfo*>(edge->GetLabel());
        if (!label) continue;
        auto d=label->GetPoseDifference();
        listener.line(0,"EDGE_ADDED",edge->GetSource()->GetObject()->GetUniqueId(),-1,-1,-1,0,
                      d.GetX(),d.GetY(),d.GetHeading(),
                      "target="+std::to_string(edge->GetTarget()->GetObject()->GetUniqueId()));
      }
    }
  }
  auto graph=mapper.GetGraph();
  std::ofstream out(argv[3]);out<<std::setprecision(17)<<"source,target,dx,dy,dyaw,var_x,var_y,var_yaw\n";
  for (auto edge:graph->GetEdges()) {
    auto link=dynamic_cast<karto::LinkInfo*>(edge->GetLabel());if (!link) continue;
    auto p=link->GetPoseDifference();auto c=link->GetCovariance();
    out<<edge->GetSource()->GetObject()->GetUniqueId()<<','<<edge->GetTarget()->GetObject()->GetUniqueId()
      <<','<<p.GetX()<<','<<p.GetY()<<','<<p.GetHeading()<<','<<c(0,0)<<','<<c(1,1)<<','<<c(2,2)<<'\n';
  }
  std::cerr << "processed="<<processed<<" edges="<<graph->GetEdges().size()<<'\n';
}
