% scale(1000) import("lower_leg_bolt.stl");

// Sketch PureShapes 140
multmatrix([[0.0, 0.0, 1.0, 70.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 140.000000;
translate([0, 0, -thickness]) {
  translate([0.000000, 0.000000, 0]) {
    cylinder(r=15.000000,h=thickness);
  }
}
}
