% scale(1000) import("lower_forearm.stl");

// Sketch PureShapes 150
multmatrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 75.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 150.000000;
translate([0, 0, -thickness]) {
  translate([0.000000, 0.000000, 0]) {
    cylinder(r=40.000000,h=thickness);
  }
}
}
