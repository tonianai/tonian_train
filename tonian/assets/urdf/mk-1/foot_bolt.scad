% scale(1000) import("foot_bolt.stl");

// Sketch PureShapes 70
multmatrix([[0.0, 0.0, 1.0, 35.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) {
thickness = 70.000000;
translate([0, 0, -thickness]) {
  translate([0.000000, 0.000000, 0]) {
    cylinder(r=15.000000,h=thickness);
  }
}
}
