model V3V
    Real R=50;
    Integer n=100;
    Real l=0.75;
    // delta is the deviation to the equal percentage law
    // 0.01 - 0.9 > different results
    Real delta0=0.9;
    Modelica.Blocks.Sources.Ramp ramp(duration = 10, height = 1)  annotation(
    Placement(visible = true, transformation(origin = {-48, 60}, extent = {{-10, -10}, {10, 10}}, rotation = 0)));
    Real phi;
    package Medium = IDEAS.Media.Water "Medium in the component";
equation
  phi = IDEAS.Fluid.Actuators.BaseClasses.equalPercentage(ramp.y, R, l, delta0);
protected
    annotation(
    uses(Modelica(version = "3.2.2"), IDEAS(version = "2.1.0")));
end V3V;