% scale(1000) import("torso_leg_connector.stl");

// Sketch PureShapes 230
multmatrix([[-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 80.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 230.000000;
translate([0, 0, -thickness]) {
  translate([0.000000, 0.000000, 0]) {
    cylinder(r=30.000000,h=thickness);
  }
}
}
