// Diagnostic replay of recorded Karto scans. No simulator or truth input.
#include <karto_sdk/Mapper.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

struct Events : karto::MapperLoopClosureListener {
  int current = -1;
  double time = -1;
  void LoopClosureCheck(const std::string &s) override {
    std::cout << "CHECK," << current << ',' << time << ',' << s << '\n';
  }
  void BeginLoopClosure(const std::string &s) override {
    std::cout << "ACCEPT," << current << ',' << time << ',' << s << '\n';
  }
};

int main(int argc, char **argv) {
  if (argc != 4) {std::cerr << "usage: replay_karto session.data radius output_edges.csv\n";return 2;}
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
  Events listener;mapper.AddListener(&listener);
  int processed=0;
  for (auto const &[id,obj] : data.GetData()) {
    auto scan=dynamic_cast<karto::LocalizedRangeScan*>(obj);
    if (!scan) continue;
    listener.current=id;listener.time=scan->GetTime();
    // Process() predicts from the recorded wheel pose. The saved corrected pose is
    // deliberately discarded; it contains information from the original result.
    scan->SetCorrectedPose(scan->GetOdometricPose());
    bool ok=mapper.Process(scan);
    if (ok) ++processed;
    if (ok) {
      auto p=scan->GetCorrectedPose();std::cout << "POSE," << id << ',' << scan->GetTime()
        << ',' << p.GetX() << ',' << p.GetY() << ',' << p.GetHeading() << '\n';
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
