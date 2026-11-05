// Read-only evaluator diagnostic using the installed Karto serialization API.
// Does not initialize ROS, optimize the graph, or modify saved mapper data.
#include <karto_sdk/Mapper.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  if (argc != 3) { std::cerr << "inspect_graph session.posegraph output_prefix\n"; return 2; }
  karto::Mapper mapper;
  mapper.LoadFromFile(argv[1]);
  auto graph = mapper.GetGraph();
  if (!graph) return 3;
  std::ofstream nodes(std::string(argv[2]) + "_nodes.csv");
  std::ofstream edges(std::string(argv[2]) + "_edges.csv");
  nodes << std::setprecision(17) << "id,time,x,y,yaw,odom_x,odom_y,odom_yaw\n";
  edges << std::setprecision(17) << "source,target,dx,dy,dyaw,var_x,var_y,var_yaw\n";
  for (const auto &device : graph->GetVertices()) {
    for (const auto &vertex : device.second) {
      auto scan = vertex.second->GetObject();
      auto pose = scan->GetCorrectedPose(); auto odom = scan->GetOdometricPose();
      nodes << scan->GetUniqueId() << ',' << scan->GetTime() << ','
            << pose.GetX() << ',' << pose.GetY() << ',' << pose.GetHeading() << ','
            << odom.GetX() << ',' << odom.GetY() << ',' << odom.GetHeading() << '\n';
    }
  }
  for (auto edge : graph->GetEdges()) {
    auto link = dynamic_cast<karto::LinkInfo *>(edge->GetLabel());
    if (!link) continue;
    auto d = link->GetPoseDifference(); auto cov = link->GetCovariance();
    edges << edge->GetSource()->GetObject()->GetUniqueId() << ','
          << edge->GetTarget()->GetObject()->GetUniqueId() << ','
          << d.GetX() << ',' << d.GetY() << ',' << d.GetHeading() << ','
          << cov(0,0) << ',' << cov(1,1) << ',' << cov(2,2) << '\n';
  }
}
