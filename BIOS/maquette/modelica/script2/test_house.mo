model test_house
  BuildSysPro.BoundaryConditions.Weather.Meteofile meteofile annotation(
    Placement(visible = true, transformation(origin = {-148, -17}, extent = {{-30, -27}, {30, 27}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Components.HeatCapacitor inertia(C = 1.59e9)  annotation(
    Placement(visible = true, transformation(origin = {-38, 60}, extent = {{-34, -34}, {34, 34}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor TROutdoor(R = 1.96e-4)  annotation(
    Placement(visible = true, transformation(origin = {-76, 0}, extent = {{-20, -20}, {20, 20}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor TRindoor(R = 2.12e-1)  annotation(
    Placement(visible = true, transformation(origin = {9, 9}, extent = {{-19, -19}, {19, 19}}, rotation = 0)));
  BuildSysPro.Building.AirFlow.HeatTransfer.AirNode airNode(V = 1000)  annotation(
    Placement(visible = true, transformation(origin = {149, -11}, extent = {{-27, -27}, {27, 27}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Celsius.TemperatureSensor Wall annotation(
    Placement(visible = true, transformation(origin = {-22, -52}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
  Modelica.Thermal.HeatTransfer.Celsius.TemperatureSensor Indoor annotation(
    Placement(visible = true, transformation(origin = {86,44}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
equation
  connect(TROutdoor.port_b, inertia.port) annotation(
    Line(points = {{-56, 0}, {-56, 1}, {-38, 1}, {-38, 26}}, color = {191, 0, 0}));
  connect(inertia.port, TRindoor.port_a) annotation(
    Line(points = {{-38, 26}, {-38, 9}, {-10, 9}}, color = {191, 0, 0}));
  connect(meteofile.T_dry, TROutdoor.port_a) annotation(
    Line(points = {{-121, -8.9}, {-110, -8.9}, {-110, 0}, {-96, 0}}, color = {191, 0, 0}));
  connect(TRindoor.port_b, airNode.port_a) annotation(
    Line(points = {{28, 9}, {75.5, 9}, {75.5, -22}, {149, -22}}, color = {191, 0, 0}));
  connect(Wall.port, TRindoor.port_a) annotation(
    Line(points = {{-26, -54}, {-26, 9}, {-10, 9}}, color = {191, 0, 0}));
  connect(TRindoor.port_b, Indoor.port) annotation(
    Line(points = {{28, 9}, {28, 20.5}, {76, 20.5}, {76, 44}}, color = {191, 0, 0}));
  annotation(
    uses(Modelica(version = "3.2.2")),
    Documentation,
  Diagram(coordinateSystem(extent = {{-200, -100}, {200, 100}})),
  Icon(coordinateSystem(extent = {{-200, -100}, {200, 100}})),
  version = "");
end test_house;
