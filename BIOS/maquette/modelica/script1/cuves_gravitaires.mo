model cuves_gravitaires
  replaceable package Water = Modelica.Media.Water.ConstantPropertyLiquidWater;
  inner Modelica.Fluid.System system(energyDynamics = Modelica.Fluid.Types.Dynamics.FixedInitial)  annotation(
    Placement(visible = true, transformation(origin = {74, 82}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));

  Modelica.Fluid.Vessels.OpenTank CuvePleine(
      redeclare package Medium = Water,
      crossArea = 1,
      height = 2,
      level_start = 1,
      nPorts = 3,
      portsData = {Modelica.Fluid.Vessels.BaseClasses.VesselPortsData(diameter = 0.1),
                   Modelica.Fluid.Vessels.BaseClasses.VesselPortsData(diameter = 0.1),
                   Modelica.Fluid.Vessels.BaseClasses.VesselPortsData(diameter = 0.01)}) annotation(
    Placement(visible = true, transformation(origin = {-50, 34}, extent = {{-20, -20}, {20, 20}}, rotation = 0)));

  Modelica.Fluid.Vessels.OpenTank CuveVide(
      redeclare package Medium = Water,
      crossArea = 1,
      height = 2,
      level_start = 0,
      nPorts= 2,
      portsData = {Modelica.Fluid.Vessels.BaseClasses.VesselPortsData(diameter = 0.1),
                   Modelica.Fluid.Vessels.BaseClasses.VesselPortsData(diameter = 0.1)}) annotation(
    Placement(visible = true, transformation(origin = {54, 32}, extent = {{-20, -20}, {20, 20}}, rotation = 0)));

  Modelica.Fluid.Pipes.DynamicPipe pipeGo(
      redeclare package Medium = Water,
      diameter = 0.1,
      length = 100)  annotation(
    Placement(visible = true, transformation(origin = {0, -54}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Fluid.Pipes.DynamicPipe PipeReturn(
      redeclare package Medium = Water,
      diameter = 0.1,
      length = 100) annotation(
    Placement(visible = true, transformation(origin = {0, -24}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Fluid.Sensors.RelativePressure relativePressure(
      redeclare package Medium = Water) annotation(
    Placement(visible = true, transformation(origin = {-10, 8}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Fluid.Sources.FixedBoundary cstPressure(
      redeclare package Medium = Water, nPorts = 1) annotation(
    Placement(visible = true, transformation(origin = {-8, 34}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Fluid.Sensors.Pressure pressure(
      redeclare package Medium = Water) annotation(
    Placement(visible = true, transformation(origin = {-32, -8}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
equation
  connect(pipeGo.port_b, CuveVide.ports[1]) annotation(
    Line(points = {{10, -54}, {68, -54}, {68, 12}, {54, 12}}, color = {0, 127, 255}));
  connect(CuvePleine.ports[1], pipeGo.port_a) annotation(
    Line(points = {{-50, 14}, {-54, 14}, {-54, -54}, {-10, -54}}, color = {0, 127, 255}));
  connect(CuvePleine.ports[2], relativePressure.port_a) annotation(
    Line(points = {{-50, 14}, {-33, 14}, {-33, 8}, {-20, 8}}, color = {0, 127, 255}));
  connect(relativePressure.port_b, cstPressure.ports[1]) annotation(
    Line(points = {{0, 8}, {8, 8}, {8, 38}, {2, 38}, {2, 34}}, color = {0, 127, 255}));
  connect(pressure.port, CuvePleine.ports[3]) annotation(
    Line(points = {{-32, -18}, {-48, -18}, {-48, 14}, {-50, 14}}, color = {0, 127, 255}));
  connect(pressure.port, PipeReturn.port_a) annotation(
    Line(points = {{-32, -18}, {-22, -18}, {-22, -24}, {-10, -24}, {-10, -24}}, color = {0, 127, 255}));
  connect(PipeReturn.port_b, CuveVide.ports[2]) annotation(
    Line(points = {{10, -24}, {48, -24}, {48, 12}, {54, 12}}, color = {0, 127, 255}));
  annotation(
    uses(Modelica(version = "3.2.3"), BuildSysPro(version = "3.3.0"), Buildings(version = "6.0.0")));
end cuves_gravitaires;
